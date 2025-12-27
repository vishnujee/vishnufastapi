import aiohttp
import feedparser
import re
import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import os
import pytz
import time

from urllib.parse import urlencode  
from urllib.parse import urlparse, parse_qs, quote

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('job_scoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === GEMINI SAFE SESSION MANAGEMENT ===
import google.generativeai as genai
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Global shared session for Gemini (will be created once and reused)
_gemini_session: aiohttp.ClientSession | None = None

def validate_and_clean_url_standalone(url: str) -> str:
    """Standalone version of URL validation for use outside classes"""
    if not url:
        return "#"
    
    # Convert Google News RSS URLs to proper article URLs
    if 'news.google.com/rss/articles/' in url:
        try:
            article_match = re.search(r'/articles/([^?]+)', url)
            if article_match:
                article_id = article_match.group(1)
                cleaned_url = f"https://news.google.com/articles/{article_id}?hl=en-IN&gl=IN&ceid=IN:en"
                # logger.info(f"✅ Converted RSS URL: {cleaned_url}")
                return cleaned_url
        except Exception as e:
            logger.warning(f"URL conversion failed for {url}: {e}")
    
    # Return original URL if no conversion needed
    return url

def is_search_results_url(url: str) -> bool:
    """
    Simple detection for search results/job listing pages.
    Returns True for BAD URLs, False for individual job URLs.
    """
    if not url:
        return False
    
    url_lower = url.lower()
    
    # ONLY check for CLEAR patterns
    if '-jobs' in url_lower:
        # Exception: "job-listings-" is GOOD
        if 'job-listings-' in url_lower:
            return False
        return True
    
    if '?k=' in url_lower:
        return True
    
    if 'linkedin.com/jobs/search/' in url_lower:
        return True
    
    if 'jobs?' in url_lower:
        return True
    
    return False

# Add this function near the top of your file
def create_url_mapping(sources: List[Dict]) -> Dict[str, str]:
    """Create short URL -> full URL mapping to avoid LLM truncation"""
    url_map = {}
    for i, source in enumerate(sources):
        short_id = f"URL_{i+1}"
        url_map[short_id] = source['url']
    return url_map

def restore_full_urls(content: str, url_map: Dict[str, str]) -> str:
    """Restore full URLs from short IDs in LLM response"""
    for short_id, full_url in url_map.items():
        content = content.replace(short_id, full_url)
    return content

def get_gemini_model(model_name: str = "gemini-2.0-flash-lite"):
    return genai.GenerativeModel(model_name)

ist = pytz.timezone('Asia/Kolkata')

def safe_job_title(title: str, max_length: int = 150) -> str:
    """Ensure job titles don't break formatting"""
    if len(title) > max_length:
        return title[:max_length-3] + "..."
    return title

def get_source_name(url: str) -> str:
    """Extract source name from URL with better error handling"""
    try:
        
        domain = urlparse(url).netloc
        
        if not domain:
            return "News Source"
            
        domain = domain.replace('www.', '').replace('news.', '')
        
        # Handle Google News URLs specifically
        if 'news.google.com' in domain:
            return "Google News"
            
        # Extract main domain
        main_domain_parts = domain.split('.')
        if len(main_domain_parts) >= 2:
            # Get the actual domain name (not TLD)
            main_domain = main_domain_parts[-2] if main_domain_parts[-2] not in ['co', 'com'] else main_domain_parts[-3]
            return main_domain.title()
        else:
            return domain.title()
            
    except Exception as e:
        print(f"Error extracting source name from {url}: {e}")
        return "News Source"

class EnergyNewsSearcher:
    def __init__(self):
        self.session = None
        self.trusted_sources = [
            "economic times", "moneycontrol", "reuters", "bloomberg", 
            "ndtv", "business standard", "financial express", "the hindu",
            "times of india", "hindustan times", "livemint", "cnbc",
            "espn", "cricbuzz", "espncricinfo", "the guardian",
            "linkedin", "naukri.com"
            "techcrunch", "theverge", "wired", "techradar"
        ]

        
        
        self.top_companies_list = [

            # =========================
            # Power / Energy / Utilities
            # =========================
            "NTPC", "Power Grid", "NHPC", "NPCIL",
            "Tata Power", "Tata Renewable",
            "Adani Power", "Adani Green", "Adani Transmission",
            "ReNew", "Torrent Power", "JSW Energy", "SJVN",
            "GMR Energy", "Reliance Power", "RattanIndia",
            "BHEL", "Siemens Energy", "GE Power", "GE Vernova",
            "ABB", "Schneider", "Hitachi Energy",
            "L&T Power", "Sterling & Wilson",
            "Suzlon", "Inox Wind",
            "Thermax", "ISGEC",
            "KEC International", "Kalpataru Power",
            "Techno Electric",
            "Waaree Energies", "Vikram Solar",
            "Azure Power", "Greenko",
            "Amara Raja", "Exide",
            "Jindal Power",

            # =========================
            # Construction / Infrastructure / EPC
            # =========================
            "Larsen & Toubro", "L&T",
            "Shapoorji Pallonji",
            "Afcons Infrastructure",
            "Tata Projects",
            "NCC Limited", "NCC",
            "HCC",
            "IRCON International", "IRCON",
            "RITES",
            "RVNL", "Rail Vikas Nigam",
            "Gammon India", "Gammon",
            "Simplex Infrastructure", "Simplex",
            "GMR Infrastructure",
            "GVK",
            "Dilip Buildcon",
            "Ashoka Buildcon",
            "J Kumar Infraprojects", "J Kumar",
            "Patel Engineering",
            "Montecarlo",
            "Sadbhav Engineering", "Sadbhav",
            "PNC Infratech",
            "Megha Engineering", "MEIL",
            "ITD Cementation",
            "Ahluwalia Contracts", "Ahluwalia",
            "KNR Constructions", "KNR",
            "Capacite Infraprojects", "Capacite",
            "Som Projects",
            "Hindustan Prefab",
            "NBCC",
            "Engineers India", "EIL",

            # =========================
            # Electrical / Industrial Manufacturing
            # =========================
            "Havells",
            "Polycab",
            "KEI Industries",
            "Apar Industries",
            "Crompton Greaves",
            "Bajaj Electricals",
            "Luminous",
            "Finolex Cables",
            "CG Power",
            "Voltamp",

            # =========================
            # IT / Software (≤ 20)
            # =========================
            "TCS", "Tata Consultancy",
            "Infosys",
            "Wipro",
            "HCL",
            "Tech Mahindra",
            "LTIMindtree",
            "Accenture",
            "IBM",
            "Capgemini",
            "Cognizant",
            "Oracle",
            "SAP",
            "Microsoft",
            "Google",
            "Amazon",
            "Zoho",
            "Freshworks",
            "Persistent",
            "Tata Elxsi",
            "Siemens Digital"
        ]



    def clean_html_content(self, html_text: str, max_length: int = 500) -> str:
        """Remove HTML tags and clean text for LLM processing"""
        if not html_text:
            return ""
        
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', html_text)
        
        # Remove common HTML entities
        clean_text = re.sub(r'&[a-z]+;', ' ', clean_text)
        
        # Remove excessive whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Remove URLs if they slipped through
        clean_text = re.sub(r'https?://\S+', '', clean_text)
        
        # Remove common RSS artifacts
        clean_text = re.sub(r'\s*-\s*[A-Za-z\s]+$', '', clean_text)  # Remove source attribution
        clean_text = re.sub(r'\d+\s*(hours?|days?|minutes?)\s*ago', '', clean_text)  # Remove timestamps
        
        # Truncate to max length
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length] + "..."
        
        return clean_text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()

    async def close_session(self):
        """Close aiohttp session properly"""
        if self.session:
            await self.session.close()
            self.session = None

    def deduplicate_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Remove duplicate jobs before LLM processing"""
        seen_titles = set()
        unique_jobs = []
        
        for job in jobs:
            title_key = re.sub(r'[^\w\s]', '', job["title"].lower()).strip()
            title_key = ' '.join(title_key.split()[:8])
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_jobs.append(job)
        
        logger.info(f"Deduplication: {len(jobs)} → {len(unique_jobs)} jobs")
        return unique_jobs
    
    def expand_job_keywords(self, base_keywords: List[str]) -> List[str]:
        """Smart keyword expansion for job searches"""
        if not base_keywords:
            return []
            
        expanded = set()
        
        job_roles_map = {
            "software": ["developer", "engineer", "programmer", "full stack", "backend", "frontend"],
            "engineer": ["engineering", "technical", "development", "design"],
            "manager": ["management", "lead", "head", "supervisor", "director"],
            "analyst": ["analysis", "business analyst", "data analyst", "research"],
            "power": ["energy", "electrical", "utility", "grid", "transmission","manager","engineer","supervisor","distribution","transmission","substation","site","construction","erection","stringing"],
            "renewable": ["solar", "wind", "green energy", "sustainability"],
            "junior": ["fresher", "entry level", "trainee", "associate"],
            "civil": ["construction", "structural", "infrastructure", "site engineer", "project engineer"],
        }
        
        for keyword in base_keywords:
            keyword_lower = keyword.lower()
            expanded.add(keyword_lower)
            
            for base, variations in job_roles_map.items():
                if base in keyword_lower:
                    for variation in variations:
                        expanded.add(variation)
            
            expanded.add(f"{keyword_lower} jobs")
            expanded.add(f"{keyword_lower} hiring")
            expanded.add(f"{keyword_lower} vacancy")
        
        return list(expanded)
    
    def _format_jobs_for_llm_with_categories(self, jobs: List[Dict]) -> str:
        """Format jobs with category labels for LLM"""
        formatted = []
        for i, job in enumerate(jobs[:250]):  # Limit to avoid token overflow
            category = job.get('category', 'unknown')
            source = job.get('source', 'Unknown')
            formatted.append(f"JOB {i+1} [{category.upper()}] ({source}): {job['title'][:80]}")
        return "\n".join(formatted)
    

    def debug_job_order(self, filtered_jobs: List[Dict], limit: int = 50):
        """Debug the order of jobs in filtered list"""
        logger.info("🔍 DEBUGGING JOB ORDER IN FILTERED LIST:")
        
        linkedin_count = 0
        naukri_count = 0
        
        for i, job in enumerate(filtered_jobs[:limit]):
            source = job.get('source', 'unknown')
            if source == 'LinkedIn':
                linkedin_count += 1
            elif source == 'Naukri.com':
                naukri_count += 1
            
            logger.info(f"  {i+1}. Source: {source} | Category: {job.get('category', 'unknown')} | {job['title'][:40]}...")
        
        logger.info(f"📊 First {limit} jobs: LinkedIn={linkedin_count}, Naukri={naukri_count}")
        
        # Check overall distribution
        total_linkedin = sum(1 for j in filtered_jobs if j.get('source') == 'LinkedIn')
        total_naukri = sum(1 for j in filtered_jobs if j.get('source') == 'Naukri.com')
        logger.info(f"📊 Total in filtered_jobs: LinkedIn={total_linkedin}, Naukri={total_naukri}")
    
    async def llm_batch_filter_jobs(self, all_jobs: List[Dict], keywords: List[str] = None, max_output_jobs: int = 35):
        """LLM filtering with dynamic numbering - NO CAPS"""
        try:
            # Separate jobs by source
            linkedin_jobs = [job for job in all_jobs if job['source'] == 'LinkedIn']
            naukri_jobs = [job for job in all_jobs if job['source'] == 'Naukri.com']
            
            logger.info(f"📊 TOTAL JOBS AVAILABLE: {len(all_jobs)}")
            logger.info(f"📊 SOURCE COUNTS: LinkedIn={len(linkedin_jobs)}, Naukri={len(naukri_jobs)}")
            
            # **NEW: Filter out personal profiles and low-quality jobs BEFORE sorting**
            logger.info("🔍 FILTERING OUT PERSONAL PROFILES AND LOW-QUALITY JOBS...")
            
            def is_quality_job(job):
                # Filter out personal profiles
                if job.get('is_personal_profile', False):
                    logger.debug(f"   ❌ Filtered personal profile: {job['title'][:50]}...")
                    return False
                
                # Filter out bulk postings with very low scores
                if job.get('is_bulk_posting', False) and job.get('combined_score', 0) < 0.3:
                    logger.debug(f"   ❌ Filtered low-quality bulk posting: {job['title'][:50]}...")
                    return False
                
                # Filter out "other" category with low scores
                if job.get('category') == 'other' and job.get('combined_score', 0) < 0.4:
                    logger.debug(f"   ❌ Filtered low-score 'other' category: {job['title'][:50]}...")
                    return False
                
                # Minimum score threshold
                if job.get('combined_score', 0) < 0.4:
                    logger.debug(f"   ❌ Filtered low overall score: {job['title'][:50]}...")
                    return False
                
                return True
            
            # Apply filtering
            linkedin_filtered = [job for job in linkedin_jobs if is_quality_job(job)]
            naukri_filtered = [job for job in naukri_jobs if is_quality_job(job)]
            
            logger.info(f"📊 AFTER FILTERING: LinkedIn {len(linkedin_jobs)}→{len(linkedin_filtered)}, Naukri {len(naukri_jobs)}→{len(naukri_filtered)}")
            
            # Sort by score (already penalized for profiles/bulk)
            linkedin_sorted = sorted(linkedin_filtered, 
                                    key=lambda x: x.get('combined_score', 0.5), 
                                    reverse=True)
            naukri_sorted = sorted(naukri_filtered, 
                                key=lambda x: x.get('combined_score', 0.5), 
                                reverse=True)
            
            # DEBUG: Log distribution
            logger.info("📊 CATEGORY DISTRIBUTION:")
            for source_name, job_list in [("LinkedIn", linkedin_sorted), ("Naukri", naukri_sorted)]:
                cats = {}
                for job in job_list:
                    cat = job.get('category', 'unknown')
                    cats[cat] = cats.get(cat, 0) + 1
                logger.info(f"   {source_name}: {cats}")
            
            # === DYNAMIC NUMBERING ===
            # Build job display with clear source sections
            jobs_by_source = []
            display_mapping = []  # (display_idx, source, actual_idx)
            
            # 1. LinkedIn jobs
            linkedin_start = 1
            for i, job in enumerate(linkedin_sorted):
                display_idx = linkedin_start + i
                jobs_by_source.append((display_idx, "LinkedIn", i, job))
                display_mapping.append((display_idx, "LinkedIn", i))
            
            # 2. Naukri jobs (continue numbering)
            naukri_start = len(linkedin_sorted) + 1
            for i, job in enumerate(naukri_sorted):
                display_idx = naukri_start + i
                jobs_by_source.append((display_idx, "Naukri.com", i, job))
                display_mapping.append((display_idx, "Naukri.com", i))
            
            logger.info(f"🔢 DISPLAY NUMBERING: LinkedIn items 1-{len(linkedin_sorted)}, Naukri items {naukri_start}-{len(display_mapping)}")
            
            # Prepare jobs text
            linkedin_text = []
            naukri_text = []
            
            for display_idx, source, actual_idx, job in jobs_by_source:
                title_display = job['title'][:150]  # Increased
                content_preview = job.get('content', '')[:350]  # Add content preview
                score = job.get('combined_score', 0.5)
                category = job.get('category', 'unknown')
                
                # Include content in the display
                job_line = f"{display_idx}. [{category.upper()}] Score: {score:.2f} | {title_display}"
                if content_preview:
                    job_line += f"\n   Description: {content_preview}"
                
                if source == "LinkedIn":
                    linkedin_text.append(job_line)
                else:
                    naukri_text.append(job_line)
            
            # Build the prompt text

            jobs_text = f"=== LINKEDIN JOBS (Items 1-{len(linkedin_sorted)}) ===\n"
            jobs_text += "\n".join(linkedin_text)
            
            jobs_text += f"\n\n=== NAUKRI.COM JOBS (Items {naukri_start}-{len(display_mapping)}) ===\n"
            jobs_text += "\n".join(naukri_text)
            
            logger.info(f"📋 PRESENTING TO LLM: {len(linkedin_sorted)} LinkedIn + {len(naukri_sorted)} Naukri = {len(display_mapping)} total jobs")
            
#===============================================================
# Details log before presenting to LLM
#===============================================================

            logger.info("🔍 DETAILED JOB LIST SENT TO LLM:")

            # Log LinkedIn jobs being sent
            logger.info(f"=== LINKEDIN JOBS BEING SENT TO LLM (First 10 of {len(linkedin_sorted)}):")
            for i, (display_idx, source, actual_idx, job) in enumerate(jobs_by_source[:10]):
                if source == "LinkedIn":
                    cat = job.get('category', 'unknown')
                    score = job.get('combined_score', 0.5)
                    title = job['title']
                    logger.info(f"  Item {display_idx}. [{cat}] Score: {score:.2f} | {title}...")

            # Log Naukri jobs being sent  
            logger.info(f"=== NAUKRI JOBS BEING SENT TO LLM (First 10 of {len(naukri_sorted)}):")
            naukri_jobs_in_list = [(d, s, a, j) for d, s, a, j in jobs_by_source if s == "Naukri.com"]
            for i, (display_idx, source, actual_idx, job) in enumerate(naukri_jobs_in_list[:10]):
                cat = job.get('category', 'unknown')
                score = job.get('combined_score', 0.5)
                title = job['title']
                logger.info(f"  Item {display_idx}. [{cat}] Score: {score:.2f} | {title}...")

            # Log category distribution of what's being sent
            logger.info("📊 CATEGORY DISTRIBUTION BEING SENT TO LLM:")
            sent_categories = {'electrical': 0, 'civil': 0, 'software': 0, 'other': 0}
            for display_idx, source, actual_idx, job in jobs_by_source:
                cat = job.get('category', 'other')
                if cat in sent_categories:
                    sent_categories[cat] += 1
            logger.info(f"   Electrical: {sent_categories['electrical']}")
            logger.info(f"   Civil: {sent_categories['civil']}")
            logger.info(f"   Software: {sent_categories['software']}")
            logger.info(f"   Other: {sent_categories['other']}")

            # Log personal profile count in what's being sent
            personal_profiles = sum(1 for _, _, _, job in jobs_by_source if job.get('is_personal_profile', False))
            bulk_postings = sum(1 for _, _, _, job in jobs_by_source if job.get('is_bulk_posting', False))
            logger.info(f"⚠️  WARNING: {personal_profiles} personal profiles and {bulk_postings} bulk postings still in LLM input")

#===============================================================
# Details log before presenting to LLM
#===============================================================

            model = get_gemini_model("gemini-2.0-flash")
            
            # In the llm_batch_filter_jobs method, modify the prompt's SELECTION ALGORITHM section:

            prompt = f"""
                CRITICAL INSTRUCTIONS - READ CAREFULLY:
                1. You MUST select EXACTLY {max_output_jobs} jobs
                2. SOURCE DISTRIBUTION IS MANDATORY AND STRICTLY ENFORCED:
                    - LinkedIn: 20-25 jobs **(items 1-{len(linkedin_sorted)})**
                    - Naukri.com: 10-15 jobs **(items {naukri_start}-{len(display_mapping)})**
                
                3. CATEGORY DISTRIBUTION IS MANDATORY:
                    - Electrical: 16-18 jobs
                    - Civil: 8-10 jobs
                    - Software: 7-9 jobs
                    
                SELECTION ALGORITHM (FOLLOW EXACTLY):
                Step 1: FIRST select 10-15 Naukri.com jobs (electrical, civil, software mix)
                Step 2: THEN select 20-25 LinkedIn jobs (complete the categories)
                Step 3: Ensure final distribution matches targets
                
                PAY ATTENTION TO SOURCE IN EACH ITEM: Each job shows "Source: LinkedIn" or "Source: Naukri.com"
                
                FAILURE CONDITIONS (DO NOT DO):
                - If you select 0 Naukri jobs = COMPLETE FAILURE
                - If you select less than 10 Naukri jobs = FAILURE
                - If you ignore category distribution = FAILURE
                
                IMPORTANT: Naukri jobs have score indicators - use them but prioritize meeting distribution requirements.
                
                
                DEBUG CHECK: Before outputting, count:
                - LinkedIn jobs selected: ___
                - Naukri.com jobs selected: ___ (MUST BE 10-15)
                - Total jobs: ___ (MUST BE {max_output_jobs})
            
            
                EXCLUSION CRITERIA - DO NOT SELECT:
                - Bulk job posting
                - Marketing, HR, finance, admin roles
                - Individual profiles, resume services, career advice
                - Overseas jobs not in India
                - Sales engineer, business development roles
                - Call center, BPO, customer service

                REJECT THESE SPECIFIC PATTERNS:
                - Job search results pages (e.g., "Civil Engineer jobs in Champa Chhattisgarh")
                - Multiple job listings (e.g., "22 Civil Engineer Site Engineer jobs in Davangere")
                - Any title with "jobs in [location]" pattern
                - Any title with numbers followed by "jobs" (e.g., "15 jobs found", "10+ openings")
                - Electrical Engineer - Naukri.com



              JOB LISTINGS:
                {jobs_text}
                
                RESPONSE FORMAT: 
                ONLY comma-separated numbers like "1,5,12,8,3,17,22,9,14,6,19,11,25,2,7,16,20,4,13,18,10,21,15,23,24,26,27,28,29,30,31,32,33,34,35"
                EXACTLY {max_output_jobs} numbers. NO other text.

            """
            
            logger.info(f"📝 Prompt length: {len(prompt)} characters")
            logger.info("🤖 Sending prompt to LLM...")
            
            response = model.generate_content(prompt)
            selected_numbers_text = response.text.strip()
            
            # DEBUG: Log raw response
            logger.info(f"📄 LLM Raw Response (first 300 chars): {selected_numbers_text[:300]}...")
            
            # Parse selected numbers
            selected_display_indices = []
            for num_str in re.findall(r'\d+', selected_numbers_text):
                idx = int(num_str)
                selected_display_indices.append(idx)
                if len(selected_display_indices) >= max_output_jobs:
                    break
            
            logger.info(f"🔢 Parsed {len(selected_display_indices)} display indices from LLM")
            
            # Create lookup for quick mapping
            display_to_job = {}
            for display_idx, source, actual_idx, job in jobs_by_source:
                display_to_job[display_idx] = (source, actual_idx, job)
            
            # Map display indices to actual jobs
            selected_jobs = []
            linkedin_count = 0
            naukri_count = 0
            category_counts = {'electrical': 0, 'civil': 0, 'software': 0, 'other': 0}
            invalid_indices = []
            
            for display_idx in selected_display_indices:
                if display_idx in display_to_job:
                    source, actual_idx, job = display_to_job[display_idx]
                    selected_jobs.append(job)
                    
                    if source == "LinkedIn":
                        linkedin_count += 1
                    else:
                        naukri_count += 1
                    
                    cat = job.get('category', 'other')
                    if cat in category_counts:
                        category_counts[cat] += 1
                    else:
                        category_counts['other'] += 1
                    
                    logger.debug(f"✅ Selected {source}: {job['title'][:50]}... (Cat: {cat}, Score: {job.get('combined_score', 0):.2f})")
                else:
                    invalid_indices.append(display_idx)
                    logger.warning(f"⚠️ Invalid display index: {display_idx}")
            
            if invalid_indices:
                logger.warning(f"⚠️ {len(invalid_indices)} invalid indices: {invalid_indices}")
            
            logger.info("📊 SELECTION RESULTS:")
            logger.info(f"   Total selected: {len(selected_jobs)}")
            logger.info(f"   Sources: LinkedIn={linkedin_count}, Naukri={naukri_count}")
            logger.info(f"   Categories: {category_counts}")
            
            # Calculate average scores
            if selected_jobs:
                avg_score = sum(j.get('combined_score', 0) for j in selected_jobs) / len(selected_jobs)
                logger.info(f"   Average score: {avg_score:.2f}")
            
            # Validate distribution
            # === SIMPLIFIED VALIDATION WITH YOUR CRITERIA ===
            def is_distribution_valid():
                """Check if selection meets all criteria"""
                # 1. Source validation
                if not (10 <= linkedin_count <= 29):
                    logger.warning(f"❌ LinkedIn count {linkedin_count} outside range 10-29")
                    return False
                
                if not (5 <= naukri_count <= 15):
                    logger.warning(f"❌ Naukri count {naukri_count} outside range 5-15")
                    return False
                
                # 2. Category validation
                if not (10 <= category_counts['electrical'] <= 27):
                    logger.warning(f"❌ Electrical count {category_counts['electrical']} outside range 10-27")
                    return False
                
                if not (4 <= category_counts['civil'] <= 10):
                    logger.warning(f"❌ Civil count {category_counts['civil']} outside range 4-10")
                    return False
                
                if not (3 <= category_counts['software'] <= 9):
                    logger.warning(f"❌ Software count {category_counts['software']} outside range 3-9")
                    return False
                
                # # 3. Total count check
                # if len(selected_jobs) != max_output_jobs:
                #     logger.warning(f"❌ Total jobs {len(selected_jobs)} not equal to {max_output_jobs}")
                #     return False

                # 3. Total count check with variance of ±10
                allowed_variance = 10
                if not (max_output_jobs - allowed_variance <= len(selected_jobs) <= max_output_jobs + allowed_variance):
                    logger.warning(f"❌ Total jobs {len(selected_jobs)} not within allowed variance ±{allowed_variance} of {max_output_jobs}")
                    return False

                
                logger.info("✅ Distribution meets all criteria")
                return True
            
            # Check distribution
            if not is_distribution_valid():
                logger.error("🚨 LLM selection doesn't meet criteria - using fallback")
                return self.fallback_distributed_selection(all_jobs, max_output_jobs)
            
            # If we get here, distribution is valid
            logger.info(f"✅ FINAL SELECTION by LLM: {len(selected_jobs)} jobs")
            logger.info(f"   Sources: LinkedIn={linkedin_count}, Naukri={naukri_count}")
            logger.info(f"   Categories: Electrical={category_counts['electrical']}, Civil={category_counts['civil']}, Software={category_counts['software']}")
            
            return selected_jobs
            
        except Exception as e:
            logger.error(f"LLM batch filtering failed: {e}")
            import traceback
            logger.error(f"🔍 Traceback:\n{traceback.format_exc()}")
            return self.fallback_distributed_selection(all_jobs, max_output_jobs)


    def rebalance_selection(self, selected_jobs: List[Dict], linkedin_jobs: List[Dict], 
                       naukri_jobs: List[Dict], max_output: int) -> List[Dict]:
        """Proper rebalancing that ensures ALL categories and sources"""
        logger.info("🔄 REBALANCING with STRICT category and source enforcement")
        
        # DEBUG: Log what we're starting with
        logger.info(f"📊 INPUT: {len(selected_jobs)} selected jobs")
        logger.info(f"📊 AVAILABLE: {len(linkedin_jobs)} LinkedIn, {len(naukri_jobs)} Naukri jobs")
        
        # Analyze current selection
        current_sources = {'LinkedIn': 0, 'Naukri.com': 0}
        current_categories = {'electrical': 0, 'civil': 0, 'software': 0, 'other': 0}
        
        for job in selected_jobs:
            source = job.get('source', 'unknown')
            category = job.get('category', 'other')
            
            if source in current_sources:
                current_sources[source] += 1
            
            if category in current_categories:
                current_categories[category] += 1
        
        logger.info(f"📊 Current selection analysis:")
        logger.info(f"   Sources: {current_sources}")
        logger.info(f"   Categories: {current_categories}")
        
        # TARGET DISTRIBUTION (NON-NEGOTIABLE)
        target_sources = {
            'LinkedIn': 20,      # 20-25 range, target 20
            'Naukri.com': 15     # 10-15 range, target 15
        }
        
        target_categories = {
            'electrical': 16,    # 15-18 range, target 16
            'civil': 9,          # 8-10 range, target 9
            'software': 10       # 7-9 range, target 10 (adjusted for balance)
        }
        
        logger.info(f"🎯 TARGET DISTRIBUTION:")
        logger.info(f"   Sources: {target_sources}")
        logger.info(f"   Categories: {target_categories}")
        
        # Categorize available jobs by source and category
        available_by_source_cat = {
            'LinkedIn': {'electrical': [], 'civil': [], 'software': [], 'other': []},
            'Naukri.com': {'electrical': [], 'civil': [], 'software': [], 'other': []}
        }
        
        # Fill LinkedIn jobs
        for job in linkedin_jobs:
            if job not in selected_jobs:  # Only consider jobs not already selected
                cat = job.get('category', 'other')
                if cat in available_by_source_cat['LinkedIn']:
                    available_by_source_cat['LinkedIn'][cat].append(job)
        
        # Fill Naukri jobs
        for job in naukri_jobs:
            if job not in selected_jobs:  # Only consider jobs not already selected
                cat = job.get('category', 'other')
                if cat in available_by_source_cat['Naukri.com']:
                    available_by_source_cat['Naukri.com'][cat].append(job)
        
        # Sort each list by score (highest first)
        for source in available_by_source_cat:
            for category in available_by_source_cat[source]:
                available_by_source_cat[source][category].sort(
                    key=lambda x: x.get('combined_score', 0), 
                    reverse=True
                )
        
        # DEBUG: Log available jobs
        logger.info("📊 AVAILABLE JOBS BY SOURCE & CATEGORY:")
        for source in ['LinkedIn', 'Naukri.com']:
            logger.info(f"   {source}:")
            for cat in ['electrical', 'civil', 'software']:
                count = len(available_by_source_cat[source][cat])
                if count > 0:
                    top_score = available_by_source_cat[source][cat][0].get('combined_score', 0) if count > 0 else 0
                    logger.info(f"     {cat}: {count} jobs (top score: {top_score:.2f})")
        
        # START REBUILDING SELECTION FROM SCRATCH
        final_selection = []
        
        # STRATEGY: Build category by category, ensuring source distribution
        
        # 1. FIRST - ENSURE SOFTWARE JOBS (most critical, usually fewest)
        logger.info("🎯 STEP 1: Ensuring software jobs...")
        software_needed = target_categories['software']
        
        # Try to get software from Naukri first (usually more)
        naukri_software = available_by_source_cat['Naukri.com']['software'][:software_needed]
        final_selection.extend(naukri_software)
        software_added = len(naukri_software)
        
        # If still need software, get from LinkedIn
        if software_added < software_needed:
            remaining = software_needed - software_added
            linkedin_software = available_by_source_cat['LinkedIn']['software'][:remaining]
            final_selection.extend(linkedin_software)
            software_added += len(linkedin_software)
        
        logger.info(f"✅ Added {software_added} software jobs")
        
        # 2. SECOND - ENSURE CIVIL JOBS
        logger.info("🎯 STEP 2: Ensuring civil jobs...")
        civil_needed = target_categories['civil']
        
        # Distribute between sources (prefer LinkedIn for civil)
        linkedin_civil_needed = min(civil_needed // 2 + 1, len(available_by_source_cat['LinkedIn']['civil']))
        naukri_civil_needed = civil_needed - linkedin_civil_needed
        
        # Take from LinkedIn
        if linkedin_civil_needed > 0 and len(available_by_source_cat['LinkedIn']['civil']) >= linkedin_civil_needed:
            linkedin_civil = available_by_source_cat['LinkedIn']['civil'][:linkedin_civil_needed]
            final_selection.extend(linkedin_civil)
        
        # Take from Naukri
        if naukri_civil_needed > 0 and len(available_by_source_cat['Naukri.com']['civil']) >= naukri_civil_needed:
            naukri_civil = available_by_source_cat['Naukri.com']['civil'][:naukri_civil_needed]
            final_selection.extend(naukri_civil)
        
        logger.info(f"✅ Added {linkedin_civil_needed} LinkedIn + {naukri_civil_needed} Naukri civil jobs")
        
        # 3. THIRD - ADD ELECTRICAL JOBS
        logger.info("🎯 STEP 3: Adding electrical jobs...")
        electrical_needed = target_categories['electrical']
        
        # Calculate how many electrical jobs we can add based on remaining slots
        remaining_slots = max_output - len(final_selection)
        electrical_to_add = min(electrical_needed, remaining_slots)
        
        # Distribute between sources
        linkedin_electrical_needed = min(electrical_to_add // 2, len(available_by_source_cat['LinkedIn']['electrical']))
        naukri_electrical_needed = electrical_to_add - linkedin_electrical_needed
        
        # Take from LinkedIn
        if linkedin_electrical_needed > 0 and len(available_by_source_cat['LinkedIn']['electrical']) >= linkedin_electrical_needed:
            linkedin_electrical = available_by_source_cat['LinkedIn']['electrical'][:linkedin_electrical_needed]
            final_selection.extend(linkedin_electrical)
        
        # Take from Naukri
        if naukri_electrical_needed > 0 and len(available_by_source_cat['Naukri.com']['electrical']) >= naukri_electrical_needed:
            naukri_electrical = available_by_source_cat['Naukri.com']['electrical'][:naukri_electrical_needed]
            final_selection.extend(naukri_electrical)
        
        logger.info(f"✅ Added {linkedin_electrical_needed} LinkedIn + {naukri_electrical_needed} Naukri electrical jobs")
        
        # 4. CHECK AND ADJUST SOURCE DISTRIBUTION
        logger.info("🎯 STEP 4: Adjusting source distribution...")
        
        # Count current sources
        current_sources_final = {'LinkedIn': 0, 'Naukri.com': 0}
        for job in final_selection:
            source = job.get('source')
            if source in current_sources_final:
                current_sources_final[source] += 1
        
        logger.info(f"📊 Current source distribution: {current_sources_final}")
        
        # Adjust if needed
        linkedin_diff = target_sources['LinkedIn'] - current_sources_final['LinkedIn']
        naukri_diff = target_sources['Naukri.com'] - current_sources_final['Naukri.com']
        
        if linkedin_diff > 0:
            # Need more LinkedIn jobs
            logger.info(f"➕ Need {linkedin_diff} more LinkedIn jobs")
            # Find best LinkedIn jobs not yet selected (any category)
            available_linkedin = []
            for cat in ['electrical', 'civil', 'software']:
                for job in available_by_source_cat['LinkedIn'][cat]:
                    if job not in final_selection:
                        available_linkedin.append(job)
            
            available_linkedin.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            to_add = min(linkedin_diff, len(available_linkedin))
            final_selection.extend(available_linkedin[:to_add])
            logger.info(f"✅ Added {to_add} additional LinkedIn jobs")
        
        elif naukri_diff > 0:
            # Need more Naukri jobs
            logger.info(f"➕ Need {naukri_diff} more Naukri jobs")
            # Find best Naukri jobs not yet selected (any category)
            available_naukri = []
            for cat in ['electrical', 'civil', 'software']:
                for job in available_by_source_cat['Naukri.com'][cat]:
                    if job not in final_selection:
                        available_naukri.append(job)
            
            available_naukri.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            to_add = min(naukri_diff, len(available_naukri))
            final_selection.extend(available_naukri[:to_add])
            logger.info(f"✅ Added {to_add} additional Naukri jobs")
        
        # 5. FINAL TOP-UP TO REACH EXACT COUNT
        if len(final_selection) < max_output:
            logger.info(f"🎯 STEP 5: Filling {max_output - len(final_selection)} remaining slots...")
            
            # Find all remaining high-scoring jobs
            all_remaining = []
            for source in ['LinkedIn', 'Naukri.com']:
                for cat in ['electrical', 'civil', 'software']:
                    for job in available_by_source_cat[source][cat]:
                        if job not in final_selection:
                            all_remaining.append(job)
            
            # Sort by score
            all_remaining.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            
            # Add to reach exact count
            needed = max_output - len(final_selection)
            to_add = min(needed, len(all_remaining))
            final_selection.extend(all_remaining[:to_add])
            logger.info(f"✅ Added {to_add} top-scoring jobs to reach {max_output}")
        
        # 6. FINAL VALIDATION AND EMERGENCY FIXES
        logger.info("🎯 STEP 6: Final validation...")
        
        # Trim if over
        if len(final_selection) > max_output:
            final_selection = final_selection[:max_output]
            logger.info(f"🔪 Trimmed to {max_output} jobs")
        
        # Final analysis
        final_source_counts = {'LinkedIn': 0, 'Naukri.com': 0}
        final_category_counts = {'electrical': 0, 'civil': 0, 'software': 0, 'other': 0}
        
        for job in final_selection:
            source = job.get('source')
            category = job.get('category', 'other')
            
            if source in final_source_counts:
                final_source_counts[source] += 1
            
            if category in final_category_counts:
                final_category_counts[category] += 1
        
        logger.info("📊 FINAL REBALANCED SELECTION:")
        logger.info(f"   Total jobs: {len(final_selection)}")
        logger.info(f"   Sources: {final_source_counts}")
        logger.info(f"   Categories: {final_category_counts}")
        
        # EMERGENCY CHECK: If still no software, force add
        if final_category_counts['software'] == 0:
            logger.error("🚨 CRITICAL: Still 0 software jobs after rebalancing!")
            
            # Find ANY software jobs from original lists
            all_software = []
            for job in linkedin_jobs + naukri_jobs:
                if job.get('category') == 'software' and job not in final_selection:
                    all_software.append(job)
            
            if all_software:
                all_software.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
                # Replace lowest scoring non-software jobs
                final_selection.sort(key=lambda x: x.get('combined_score', 0))
                replace_count = min(5, len(all_software), len(final_selection))
                
                for i in range(replace_count):
                    final_selection[i] = all_software[i]
                
                logger.info(f"🚨 EMERGENCY: Replaced {replace_count} jobs with software")
                
                # Recalculate
                final_category_counts['software'] = replace_count
                logger.info(f"   Updated categories: {final_category_counts}")
        
        # Calculate average score
        if final_selection:
            avg_score = sum(j.get('combined_score', 0) for j in final_selection) / len(final_selection)
            logger.info(f"   Average score: {avg_score:.2f}")
        
        # Check if distribution is acceptable
        source_ok = (
            final_source_counts['LinkedIn'] >= 18 and final_source_counts['LinkedIn'] <= 25 and
            final_source_counts['Naukri.com'] >= 10 and final_source_counts['Naukri.com'] <= 15
        )
        
        category_ok = (
            final_category_counts['electrical'] >= 15 and final_category_counts['electrical'] <= 18 and
            final_category_counts['civil'] >= 8 and final_category_counts['civil'] <= 10 and
            final_category_counts['software'] >= 7 and final_category_counts['software'] <= 9
        )
        
        if source_ok and category_ok:
            logger.info("✅ REBALANCING SUCCESSFUL: All targets met!")
        else:
            logger.warning(f"⚠️  Rebalancing partially successful:")
            if not source_ok:
                logger.warning(f"   Source distribution off: LinkedIn={final_source_counts['LinkedIn']}, Naukri={final_source_counts['Naukri.com']}")
            if not category_ok:
                logger.warning(f"   Category distribution off: Electrical={final_category_counts['electrical']}, Civil={final_category_counts['civil']}, Software={final_category_counts['software']}")
        
        return final_selection

    def fallback_distributed_selection(self, all_jobs: List[Dict], max_output_jobs: int):
        """Improved fallback that GUARANTEES category balance"""
        logger.info("🛠️ Using IMPROVED fallback with category enforcement")
        
        # Separate by source
        linkedin_jobs = [j for j in all_jobs if j['source'] == 'LinkedIn']
        naukri_jobs = [j for j in all_jobs if j['source'] == 'Naukri.com']
        
        logger.info(f"📊 Fallback input: {len(linkedin_jobs)} LinkedIn, {len(naukri_jobs)} Naukri")
        
        # CRITICAL: Count available categories
        linkedin_cats = {'electrical': [], 'civil': [], 'software': []}
        naukri_cats = {'electrical': [], 'civil': [], 'software': []}
        
        for job in linkedin_jobs:
            cat = job.get('category')
            if cat in linkedin_cats:
                linkedin_cats[cat].append(job)
        
        for job in naukri_jobs:
            cat = job.get('category')
            if cat in naukri_cats:
                naukri_cats[cat].append(job)
        
        logger.info(f"📊 LinkedIn categories: Electrical={len(linkedin_cats['electrical'])}, Civil={len(linkedin_cats['civil'])}, Software={len(linkedin_cats['software'])}")
        logger.info(f"📊 Naukri categories: Electrical={len(naukri_cats['electrical'])}, Civil={len(naukri_cats['civil'])}, Software={len(naukri_cats['software'])}")
        
        # Sort each category by score
        for cat in linkedin_cats:
            linkedin_cats[cat].sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        for cat in naukri_cats:
            naukri_cats[cat].sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        # TARGET DISTRIBUTION (35 total)
        target_categories = {
            'electrical': 18,  # 18 electrical
            'civil': 10,       # 10 civil  
            'software': 7      # 7 software
        }
        
        target_sources = {
            'LinkedIn': 20,
            'Naukri.com': 15
        }
        
        selected_jobs = []
        
        # STRATEGY: First ensure minimum software (MOST CRITICAL)
        logger.info("🎯 Prioritizing software jobs first...")
        
        # Try to get software from Naukri first (usually more)
        software_needed = target_categories['software']
        naukri_software = naukri_cats['software'][:min(software_needed, len(naukri_cats['software']))]
        selected_jobs.extend(naukri_software)
        software_selected = len(naukri_software)
        
        # If still need software, get from LinkedIn
        if software_selected < software_needed:
            additional_needed = software_needed - software_selected
            linkedin_software = linkedin_cats['software'][:min(additional_needed, len(linkedin_cats['software']))]
            selected_jobs.extend(linkedin_software)
            software_selected += len(linkedin_software)
        
        logger.info(f"✅ Selected {software_selected} software jobs")
        
        # Then ensure civil jobs
        logger.info("🎯 Adding civil jobs...")
        civil_needed = target_categories['civil']
        
        # Mix of LinkedIn and Naukri civil jobs
        linkedin_civil_count = min(civil_needed // 2, len(linkedin_cats['civil']))
        naukri_civil_count = min(civil_needed - linkedin_civil_count, len(naukri_cats['civil']))
        
        selected_jobs.extend(linkedin_cats['civil'][:linkedin_civil_count])
        selected_jobs.extend(naukri_cats['civil'][:naukri_civil_count])
        
        # Fill remaining with electrical jobs
        logger.info("🎯 Adding electrical jobs...")
        remaining_slots = max_output_jobs - len(selected_jobs)
        
        # Calculate how many electrical from each source
        linkedin_electrical_needed = min(
            remaining_slots // 2, 
            len(linkedin_cats['electrical']),
            target_sources['LinkedIn'] - sum(1 for j in selected_jobs if j['source'] == 'LinkedIn')
        )
        
        naukri_electrical_needed = min(
            remaining_slots - linkedin_electrical_needed,
            len(naukri_cats['electrical']),
            target_sources['Naukri.com'] - sum(1 for j in selected_jobs if j['source'] == 'Naukri.com')
        )
        
        selected_jobs.extend(linkedin_cats['electrical'][:linkedin_electrical_needed])
        selected_jobs.extend(naukri_cats['electrical'][:naukri_electrical_needed])
        
        # If still have slots, fill with highest scoring remaining jobs
        if len(selected_jobs) < max_output_jobs:
            logger.info(f"🔄 Filling {max_output_jobs - len(selected_jobs)} remaining slots")
            
            # Get all jobs not yet selected
            remaining_jobs = []
            for job in all_jobs:
                if job not in selected_jobs:
                    remaining_jobs.append(job)
            
            # Sort by score and add
            remaining_jobs.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            additional_needed = max_output_jobs - len(selected_jobs)
            selected_jobs.extend(remaining_jobs[:additional_needed])
        
        # Final validation
        final_categories = {'electrical': 0, 'civil': 0, 'software': 0}
        final_sources = {'LinkedIn': 0, 'Naukri.com': 0}
        
        for job in selected_jobs:
            cat = job.get('category', 'other')
            if cat in final_categories:
                final_categories[cat] += 1
            final_sources[job['source']] += 1
        
        logger.info(f"✅ FINAL FALLBACK SELECTION: {len(selected_jobs)} jobs")
        logger.info(f"   Categories: {final_categories}")
        logger.info(f"   Sources: {final_sources}")
        
        # EMERGENCY: If still no software, force add at least 2
        if final_categories['software'] == 0:
            logger.warning("🚨 EMERGENCY: Still 0 software jobs - forcing addition")
            # Find ANY software jobs
            all_software = []
            for job in all_jobs:
                if job.get('category') == 'software' and job not in selected_jobs:
                    all_software.append(job)
            
            if all_software:
                # Replace lowest scoring non-software jobs
                selected_jobs.sort(key=lambda x: x.get('combined_score', 0))
                for i in range(min(2, len(all_software))):
                    if i < len(selected_jobs):
                        selected_jobs[i] = all_software[i]
                logger.info("   Added 2 software jobs via emergency replacement")
        
        return selected_jobs[:max_output_jobs]


    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
        return self.session
    

############################################  NEW APPROACH FOR JOB SEARCH  ############################################


    # async def _analyze_job_with_llm(self, title: str, content: str, source: str) -> Dict:
    #     """FIXED LLM call - No aggressive software forcing, adds proper company detection"""
    #     try:
    #         model = get_gemini_model("gemini-2.0-flash-lite")
            
    #         # Prepare top companies list for prompt
    #         top_companies_str = "\n".join([f"- {company}" for company in self.top_companies_list[:50]])
            
    #         prompt = f"""
    #         **CRITICAL TASK**: Analyze this job posting for category AND detect if it's from a top company.
            
    #         JOB TITLE: {title}
    #         DESCRIPTION: {content[:400]}
    #         SOURCE: {source}
            
    #         **CRITICAL FIRST CHECK**: Is this a REAL job posting or just search results?
            
    #         **REJECT IF ANY OF THESE (These are NOT real jobs):**
    #         1. Search results pages: "13215 Construction Engineer Job Vacancies In India"
    #         2. Personal profiles: "NUNAKANI RAMANJANEYULU - Senior Piping Engineer"
    #         3. Generic listings: "Construction Engineer Jobs In India" (no company)
    #         4. Multiple openings: "Diploma Electrical Abb Jobs - 910 Diploma Electrical Abb Job Vacancies"
            
    #         **ACCEPT ONLY REAL JOB POSTINGS:**
    #         - Specific role with company name: "Engineer at Adani Power"
    #         - Contains experience: "4 to 8 years of experience"
    #         - Specific location: "in Chennai, Tamil Nadu"
            
    #         **TOP 100 INDIAN COMPANIES TO CHECK**:
    #         {top_companies_str}
            
    #         **CATEGORY RULES (MUST FOLLOW)**:
    #         1. **SOFTWARE**: ONLY if role involves SOFTWARE DEVELOPMENT/CODING
    #         - "Software Engineer", "Software Developer", "Programmer", "Full Stack Developer"
    #         - NOT: "Electrical Engineer with IT skills", "Maintenance Engineer in IT department"
            
    #         2. **ELECTRICAL**: Power/Electrical engineering roles
    #         - "Electrical Engineer", "Power Engineer", "Transmission Engineer", "Substation Engineer"
    #         - Renewable energy engineering roles: "Solar Engineer", "Wind Engineer"
            
    #         3. **CIVIL**: Construction/Infrastructure roles
    #         - "Civil Engineer", "Site Engineer", "Construction Engineer", "Structural Engineer"
            
    #         4. **OTHER**: Non-engineering or ambiguous roles
    #         - Sales, HR, Marketing, Admin, Business Development
    #         - Management roles without clear engineering focus
    #         - **ALSO USE "other" FOR SEARCH RESULTS/PROFILES** (score them 0.1)
            
    #         **COMPANY DETECTION RULES**:
    #         - Look for company name IN THE TITLE
    #         - Must match EXACTLY or be very close variation of companies in list above
    #         - Common patterns: "at [Company]", "- [Company]", "hiring for [Company]"
    #         - If company is NOT in top list → is_top_company = false
    #         - "LinkedIn India", "Naukri.com" are NOT top companies
            
    #         **CRITICAL EXAMPLES**:
    #         ✅ REAL JOBS (ANALYZE NORMALLY):
    #         - "Electrical Engineer at Adani Power" → Category: ELECTRICAL, Company: Adani Power, Top: YES
    #         - "Software Developer at Infosys" → Category: SOFTWARE, Company: Infosys, Top: YES
    #         - "Civil Site Engineer - L&T Construction" → Category: CIVIL, Company: L&T Construction, Top: YES
            
    #         ❌ REJECT THESE (Category: OTHER, Score: 0.1):
    #         - "13215 Construction Engineer Job Vacancies" → Category: OTHER, Score: 0.1 (search results)
    #         - "NUNAKANI RAMANJANEYULU - Senior Piping Engineer" → Category: OTHER, Score: 0.1 (profile)
    #         - "Construction Engineer Jobs In India" → Category: OTHER, Score: 0.1 (generic listing)
    #         - "Electrical Engineer - Naukri.com" → Category: OTHER, Score: 0.1 (no company)
            
    #         **SCORING**:
    #         - Engineering role (electrical/civil/software): relevance_score = 0.7-1.0
    #         - Top company bonus: +0.2 to relevance_score
    #         - Non-engineering role: relevance_score = 0.3-0.6
    #         - **Search results/profiles: relevance_score = 0.1** (This will filter them out)
            
    #         **RESPONSE (JSON ONLY)**:
    #         {{
    #             "category": "electrical/civil/software/other",
    #             "company_detected": "EXACT COMPANY NAME or 'Not Found'",
    #             "is_top_company": true/false,
    #             "relevance_score": 0.0-1.0,
    #             "confidence": 0.0-1.0,
    #             "reasoning": "Brief explanation of category and company match"
    #         }}
    #         """
            
    #         response = model.generate_content(prompt)
    #         text = response.text.strip()
    #         logger.debug(f"📄 Raw LLM response: {text}")
            
    #         # Clean JSON response
    #         text = self._clean_json_response(text)
    #         logger.debug(f"📄 Cleaned JSON: {text}")
            
    #         try:
    #             result = json.loads(text)
                
    #             # Ensure category is lowercase
    #             category = result.get("category", "other").lower()
    #             result["category"] = category
                
    #             # Extract and validate company
    #             company = result.get("company_detected", "Not Found")
                
    #             # CRITICAL FIX: Clean up company name
    #             company = company.replace("LinkedIn India", "").replace("Naukri.com", "").strip()
    #             if not company or company in ["LinkedIn", "Naukri", "Not Found", "Unknown", "Search Results"]:
    #                 company = "Not Found"
    #                 result["is_top_company"] = False
    #             else:
    #                 # Check if it's actually a top company
    #                 is_top = False
    #                 for top_company in self.top_companies_list:
    #                     if top_company.lower() in company.lower() or company.lower() in top_company.lower():
    #                         is_top = True
    #                         # Use the standardized company name
    #                         company = top_company
    #                         break
    #                 result["is_top_company"] = is_top
                
    #             result["company_detected"] = company
                
    #             # Adjust scoring based on findings
    #             relevance = float(result.get("relevance_score", 0.5))
                
    #             # Apply top company bonus
    #             if result["is_top_company"]:
    #                 relevance = min(1.0, relevance + 0.2)
    #                 logger.info(f"✅ TOP COMPANY DETECTED: {company} - {title[:50]}...")
                
    #             # Penalize "other" category
    #             if category == "other":
    #                 # If it's likely search results/profiles, give very low score
    #                 title_lower = title.lower()
    #                 if any(pattern in title_lower for pattern in [
    #                     'job vacancies', 'job openings', ' vacancies', 
    #                     'jobs in', '- jobs', ' job ', 'profile', '- linkedin', '- naukri'
    #                 ]):
    #                     relevance = 0.1
    #                     result["reasoning"] = "Search results/generic listing"
    #                 else:
    #                     relevance = max(0.3, relevance - 0.2)
                
    #             result["relevance_score"] = relevance
    #             result["confidence"] = float(result.get("confidence", 0.7))
    #             result["reasoning"] = result.get("reasoning", "No reasoning")
                
    #             # Calculate combined score
    #             result["combined_score"] = (
    #                 result["relevance_score"] * 0.6 +
    #                 result["confidence"] * 0.4
    #             )
                
    #             # Log if low score (likely bulk posting)
    #             if result["relevance_score"] < 0.3:
    #                 logger.info(f"🔍 Low score job detected: {title[:50]}... (Score: {result['relevance_score']:.2f})")
                
    #             return result
                
    #         except json.JSONDecodeError as e:
    #             logger.error(f"❌ JSON parse failed! Raw: {text[:200]}...")
    #             logger.error(f"❌ Error: {e}")
    #             return self._fallback_categorization(title, content)
                
    #     except Exception as e:
    #         logger.error(f"LLM analysis failed: {e}")
    #         return self._fallback_categorization(title, content)
    async def _analyze_job_with_llm(self, title: str, content: str, source: str) -> Dict:
        """FIXED LLM call - No aggressive software forcing, adds proper company detection"""
        try:
            model = get_gemini_model("gemini-2.0-flash-lite")
            
            # Prepare top companies list for prompt
            top_companies_str = "\n".join([f"- {company}" for company in self.top_companies_list[:50]])
            
            prompt = f"""
            **CRITICAL TASK**: Analyze this job posting for category AND detect if it's from a top company.
            
            JOB TITLE: {title}
            DESCRIPTION: {content[:400]}
            SOURCE: {source}
            
            **TOP 100 INDIAN COMPANIES TO CHECK**:
            {top_companies_str}
            
            **CATEGORY RULES (MUST FOLLOW)**:
            1. **SOFTWARE**: ONLY if role involves SOFTWARE DEVELOPMENT/CODING
            - "Software Engineer", "Software Developer", "Programmer", "Full Stack Developer"
            - NOT: "Electrical Engineer with IT skills", "Maintenance Engineer in IT department"
            
            2. **ELECTRICAL**: Power/Electrical engineering roles
            - "Electrical Engineer", "Power Engineer", "Transmission Engineer", "Substation Engineer"
            - Renewable energy engineering roles: "Solar Engineer", "Wind Engineer"
            
            3. **CIVIL**: Construction/Infrastructure roles
            - "Civil Engineer", "Site Engineer", "Construction Engineer", "Structural Engineer"
            
            4. **OTHER**: Non-engineering or ambiguous roles
            - Sales, HR, Marketing, Admin, Business Development
            - Management roles without clear engineering focus
            
            **COMPANY DETECTION RULES**:
            - Look for company name IN THE TITLE
            - Must match EXACTLY or be very close variation of companies in list above
            - Common patterns: "at [Company]", "- [Company]", "hiring for [Company]"
            - If company is NOT in top list → is_top_company = false
            - "LinkedIn India", "Naukri.com" are NOT top companies
            
            **CRITICAL EXAMPLES**:
            - "Electrical Engineer at Adani Power" → Category: ELECTRICAL, Company: Adani Power, Top: YES
            - "Software Developer at Infosys" → Category: SOFTWARE, Company: Infosys, Top: YES
            - "Civil Site Engineer - L&T Construction" → Category: CIVIL, Company: L&T Construction, Top: YES
            - "Electrical Maintenance Engineer - LinkedIn India" → Category: ELECTRICAL, Company: Not Found, Top: NO
            - "HR Recruiter at TCS" → Category: OTHER, Company: TCS, Top: YES (but category is other)
            
            **SCORING**:
            - Engineering role (electrical/civil/software): relevance_score = 0.7-1.0
            - Top company bonus: +0.2 to relevance_score
            - Non-engineering role: relevance_score = 0.3-0.6
            
            **RESPONSE (JSON ONLY)**:
            {{
                "category": "electrical/civil/software/other",
                "company_detected": "EXACT COMPANY NAME or 'Not Found'",
                "is_top_company": true/false,
                "relevance_score": 0.0-1.0,
                "confidence": 0.0-1.0,
                "reasoning": "Brief explanation of category and company match"
            }}
            """
            
            response = model.generate_content(prompt)
            text = response.text.strip()
            logger.debug(f"📄 Raw LLM response: {text}")
            
            # Clean JSON response
            text = self._clean_json_response(text)
            logger.debug(f"📄 Cleaned JSON: {text}")
            
            try:
                result = json.loads(text)
                
                # Ensure category is lowercase
                category = result.get("category", "other").lower()
                result["category"] = category
                
                # Extract and validate company
                company = result.get("company_detected", "Not Found")
                
                # CRITICAL FIX: Clean up company name
                company = company.replace("LinkedIn India", "").replace("Naukri.com", "").strip()
                if not company or company in ["LinkedIn", "Naukri", "Not Found", "Unknown"]:
                    company = "Not Found"
                    result["is_top_company"] = False
                else:
                    # Check if it's actually a top company
                    is_top = False
                    for top_company in self.top_companies_list:
                        if top_company.lower() in company.lower() or company.lower() in top_company.lower():
                            is_top = True
                            # Use the standardized company name
                            company = top_company
                            break
                    result["is_top_company"] = is_top
                
                result["company_detected"] = company
                
                # Adjust scoring based on findings
                relevance = float(result.get("relevance_score", 0.5))
                
                # Apply top company bonus
                if result["is_top_company"]:
                    relevance = min(1.0, relevance + 0.2)
                    logger.info(f"✅ TOP COMPANY DETECTED: {company} - {title[:50]}...")
                
                # Penalize "other" category
                if category == "other":
                    relevance = max(0.3, relevance - 0.2)
                
                result["relevance_score"] = relevance
                result["confidence"] = float(result.get("confidence", 0.7))
                result["reasoning"] = result.get("reasoning", "No reasoning")
                
                # Calculate combined score
                result["combined_score"] = (
                    result["relevance_score"] * 0.6 +
                    result["confidence"] * 0.4
                )
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parse failed! Raw: {text[:200]}...")
                logger.error(f"❌ Error: {e}")
                return self._fallback_categorization(title, content)
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_categorization(title, content)


    def _fallback_categorization(self, title: str, content: str) -> Dict:
        """Fallback categorization when LLM fails"""
        title_lower = title.lower()
        
        # Check for bulk posting patterns in fallback too
        bulk_patterns = [
            r'\d{3,}\s+(job|vacancy|opening)',
            r'jobs?\s+in\s+[a-z\s]+$',
            r'\d+\s+jobs?\s+(found|available|vacancies)',
        ]
        
        for pattern in bulk_patterns:
            if re.search(pattern, title_lower):
                logger.info(f"🔍 Fallback: Detected bulk posting - {title[:50]}...")
                return {
                    "category": "other",
                    "relevance_score": 0.1,
                    "confidence": 0.8,
                    "reasoning": f"Fallback: Detected as bulk/search results",
                    "company_detected": "Search Results",
                    "is_top_company": False,
                    "combined_score": 0.1 * 0.6 + 0.8 * 0.4  # = 0.38
                }
        
        # Check for personal profiles
        if re.match(r'^[a-z]+\s+[a-z]+\s+-', title_lower):
            logger.info(f"🔍 Fallback: Detected personal profile - {title[:50]}...")
            return {
                "category": "other",
                "relevance_score": 0.1,
                "confidence": 0.8,
                "reasoning": "Fallback: Detected as personal profile",
                "company_detected": "Personal Profile",
                "is_top_company": False,
                "combined_score": 0.1 * 0.6 + 0.8 * 0.4  # = 0.38
            }
        
        # Original fallback logic for real jobs
        software_keywords = ['software', 'developer', 'programmer', 'full stack', 'frontend', 
                            'backend', 'web', 'mobile', 'app', 'qa', 'tester', 'automation',
                            'devops', 'cloud', 'data', 'ai', 'ml', 'it ', 'information technology']
        
        electrical_keywords = ['electrical', 'power', 'transmission', 'distribution', 'substation',
                            'grid', 'switchgear', 'transformer', 'motor', 'drive', 'solar panel',
                            'wind turbine', 'hv', 'ehv', 'power system', 'electrical engineer']
        
        civil_keywords = ['civil', 'structural', 'construction', 'site engineer', 'building',
                        'infrastructure', 'road', 'bridge', 'highway', 'rcc', 'concrete']
        
        # Check categories in priority order
        if any(keyword in title_lower or keyword in content.lower() for keyword in software_keywords):
            category = "software"
            relevance = 0.7
        elif any(keyword in title_lower or keyword in content.lower() for keyword in electrical_keywords):
            category = "electrical"
            relevance = 0.7
        elif any(keyword in title_lower or keyword in content.lower() for keyword in civil_keywords):
            category = "civil"
            relevance = 0.7
        else:
            category = "other"
            relevance = 0.3
        
        return {
            "category": category,
            "relevance_score": relevance,
            "confidence": 0.6,
            "reasoning": f"Fallback categorization: {category}",
            "company_detected": "Not Specified",
            "is_top_company": False,
            "combined_score": relevance * 0.6 + 0.6 * 0.4
        }
    # def _fallback_categorization(self, title: str, content: str) -> Dict:
    #     """Fallback categorization when LLM fails"""
    #     title_lower = title.lower()
    #     content_lower = content.lower() if content else ""
        
    #     # Software keywords (expanded list)
    #     software_keywords = ['software', 'developer', 'programmer', 'full stack', 'frontend', 
    #                         'backend', 'web', 'mobile', 'app', 'qa', 'tester', 'automation',
    #                         'devops', 'cloud', 'data', 'ai', 'ml', 'it ', 'information technology',
    #                         'plm', 'salesforce', 'observability', 'lifecycle', 'code', 'programming',
    #                         'java', 'python', 'javascript', 'react', 'angular', 'node', 'sql']
        
    #     # Electrical keywords
    #     electrical_keywords = ['electrical', 'power', 'transmission', 'distribution', 'substation',
    #                         'grid', 'switchgear', 'transformer', 'motor', 'drive', 'solar panel',
    #                         'wind turbine', 'hv', 'ehv', 'power system', 'electrical engineer']
        
    #     # Civil keywords
    #     civil_keywords = ['civil', 'structural', 'construction', 'site engineer', 'building',
    #                     'infrastructure', 'road', 'bridge', 'highway', 'rcc', 'concrete']
        
    #     # Check categories in priority order
    #     if any(keyword in title_lower or keyword in content_lower for keyword in software_keywords):
    #         category = "software"
    #         relevance = 0.7
    #     elif any(keyword in title_lower or keyword in content_lower for keyword in electrical_keywords):
    #         category = "electrical"
    #         relevance = 0.7
    #     elif any(keyword in title_lower or keyword in content_lower for keyword in civil_keywords):
    #         category = "civil"
    #         relevance = 0.7
    #     else:
    #         category = "other"
    #         relevance = 0.3
        
    #     return {
    #         "category": category,
    #         "relevance_score": relevance,
    #         "confidence": 0.6,
    #         "reasoning": f"Fallback categorization: {category}",
    #         "company_detected": "Not Specified",
    #         "is_top_company": False,
    #         "combined_score": relevance * 0.6 + 0.6 * 0.4
    #     }





    async def search_with_top_companies(self) -> List[Dict]:
        """Search specifically for top companies"""
        session = await self.get_session()
        all_jobs = []
        
        # TOP 20 COMPANIES SPECIFIC SEARCH
        top_companies_to_search = [
            "Tata Power", "Adani Power", "L&T","Larsen & Toubro", "Torrent Power", "CESC",
            "KEC International", "Afcons","Megha Engineering", "Sterlite Power","shapoorji pallonji",
            "TCS", "Infosys", "Wipro", "HCL", "Accenture",
            "RPSG", "Siemens", "ABB", "Schneider", "GE Power",
            "Jindal", "Adani","Adani Green","Adani Electricity","Adani Transmission","Adani Solar", "JSW Energy", "ReNew", "Microsoft India"
        ]
        
        url = "https://news.google.com/rss/search"
        
        for company in top_companies_to_search:
            try:
                # Search on both LinkedIn and Naukri
                queries = [
                    f'site:linkedin.com "{company}" hiring engineer when:3d',
                    f'site:naukri.com "{company}" jobs when:3d',
                    # WALKIN    
                    f'site:naukri.com "{company}" walkin OR "walk in" OR "walk-in" when:3d',
                    f'site:linkedin.com "{company}" walkin OR "walk in" OR "walk-in" when:3d',
                    
           
                    ]
                
                for query in queries:
                    params = {
                        "q": query,
                        "hl": "en-IN",
                        "gl": "IN", 
                        "ceid": "IN:en"
                    }
                    
                    async with session.get(url, params=params, timeout=20) as resp:
                        if resp.status == 200:
                            feed = feedparser.parse(await resp.text())
                            
                            for entry in feed.entries[:8]:
                                title = entry.title
                                
                                # Skip non-jobs
                                if any(term in title.lower() for term in ['resume', 'cv', 'interview']):
                                    continue
                                
                                # Analyze with enhanced LLM
                                analysis = await self._analyze_job_with_llm(
                                    title=title,
                                    content=entry.get("summary", ""),
                                    source="LinkedIn" if "linkedin" in query else "Naukri.com"
                                )
                                
                                job_data = {
                                    "title": title,
                                    "url": entry.link,
                                    "content": entry.get("summary", "")[:200],
                                    "source": "LinkedIn" if "linkedin" in query else "Naukri.com",
                                    "published_at": entry.get("published", ""),
                                    "company": analysis["company_detected"],
                                    "is_top_company": analysis["is_top_company"],
                                    "category": analysis["category"],
                                    "relevance_score": analysis["relevance_score"],
                                    "forced_company_search": True
                                }
                                
                                if not any(job['title'] == job_data['title'] for job in all_jobs):
                                    all_jobs.append(job_data)
                                    
            except Exception as e:
                logger.warning(f"Company search failed for {company}: {e}")
                continue
        
        logger.info(f"🔍 Company-specific search: {len(all_jobs)} jobs")
        return all_jobs


    async def search_naukri_jobs_simple(self, keywords: List[str] = None) -> List[Dict]:
        """Simple Naukri search with real-time LLM analysis"""
        try:
            session = await self.get_session()
            
            # SIMPLIFIED QUERIES - just broad searches
            queries = [
                'electrical engineer OR power engineer OR transmission engineer when:3d',
                'civil engineer OR construction engineer OR site engineer when:3d',
                'software engineer OR developer OR programmer when:3d',
                'engineer jobs India when:3d'
            ]
            
            all_jobs = []
            url = "https://news.google.com/rss/search"
            
            for query in queries:
                try:
                    complete_query = f'site:naukri.com {query}'
                    
                    params = {
                        "q": complete_query,
                        "hl": "en-IN", 
                        "gl": "IN", 
                        "ceid": "IN:en"
                    }
                    
                    logger.info(f"Naukri search: {query}")
                    
                    async with session.get(url, params=params, timeout=20) as resp:
                        if resp.status == 200:
                            feed = feedparser.parse(await resp.text())
                            
                            for entry in feed.entries[:15]:  # Get more entries
                                title = entry.title
                                
                                # Skip obvious non-jobs
                                if 'resume' in title.lower() or 'interview tips' in title.lower():
                                    continue
                                
                                # REAL-TIME LLM ANALYSIS
                                analysis = await self._analyze_job_with_llm(
                                    title=title,
                                    content=entry.get("summary", ""),
                                    source="Naukri.com"
                                )
                                
                                # Skip if not engineering or very low score
                                if analysis["category"] == "other" and analysis["relevance_score"] < 0.3:
                                    continue
                                
                                job_data = {
                                    "title": title,
                                    "url": entry.link,
                                    "content": entry.get("summary", "")[:200],
                                    "source": "Naukri.com",
                                    "published_at": entry.get("published", ""),
                                    "type": "job_posting",
                                    "company": analysis["company_detected"],
                                    "is_top_company": analysis["is_top_company"],
                                    "category": analysis["category"],
                                    "relevance_score": analysis["relevance_score"],
                                    "analysis_reasoning": analysis["reasoning"]
                                }
                                
                                # Simple deduplication
                                if not any(job['title'] == job_data['title'] for job in all_jobs):
                                    all_jobs.append(job_data)
                                    
                                    # Log if top company found
                                    if analysis["is_top_company"]:
                                        logger.info(f"🏢 TOP COMPANY: {analysis['company_detected']} - {title[:50]}...")
                            
                            logger.info(f"Query found {len(feed.entries)} entries")
                                
                except Exception as e:
                    logger.warning(f"Query failed: {query} - {e}")
                    continue
            
            logger.info(f"✅ Naukri: {len(all_jobs)} jobs analyzed")
            return all_jobs[:100]  # Limit results
                        
        except Exception as e:
            logger.error(f"Naukri search failed: {e}")
            return []
    
    async def search_linkedin_jobs_simple(self, keywords: List[str] = None) -> List[Dict]:
        """Simple LinkedIn search with real-time LLM analysis"""
        try:
            session = await self.get_session()
            
            # SIMPLE QUERIES
            queries = [
                'hiring engineer OR job opening OR career opportunity when:3d',
                'electrical engineer jobs India when:3d',
                'civil engineer jobs India when:3d',
                'software engineer jobs India when:3d'
            ]
            
            all_jobs = []
            url = "https://news.google.com/rss/search"
            
            for query in queries:
                try:
                    complete_query = f'site:linkedin.com {query}'
                    
                    params = {
                        "q": complete_query,
                        "hl": "en-IN",
                        "gl": "IN", 
                        "ceid": "IN:en"
                    }
                    
                    logger.info(f"LinkedIn search: {query}")
                    
                    async with session.get(url, params=params, timeout=25) as resp:
                        if resp.status == 200:
                            feed = feedparser.parse(await resp.text())
                            
                            for entry in feed.entries[:15]:
                                title = entry.title
                                
                                # Skip non-jobs
                                if any(term in title.lower() for term in ['resume', 'cv', 'how to', 'career advice']):
                                    continue
                                
                                # REAL-TIME LLM ANALYSIS
                                analysis = await self._analyze_job_with_llm(
                                    title=title,
                                    content=entry.get("summary", ""),
                                    source="LinkedIn"
                                )
                                
                                # Skip non-engineering/low relevance
                                if analysis["category"] == "other" and analysis["relevance_score"] < 0.3:
                                    continue
                                
                                job_data = {
                                    "title": title,
                                    "url": entry.link,
                                    "content": entry.get("summary", "")[:300],
                                    "source": "LinkedIn",
                                    "published_at": entry.get("published", ""),
                                    "type": "job_posting",
                                    "company": analysis["company_detected"],
                                    "is_top_company": analysis["is_top_company"],
                                    "category": analysis["category"],
                                    "relevance_score": analysis["relevance_score"],
                                    "analysis_reasoning": analysis["reasoning"]
                                }
                                
                                if not any(job['title'] == job_data['title'] for job in all_jobs):
                                    all_jobs.append(job_data)
                                    
                                    if analysis["is_top_company"]:
                                        logger.info(f"🏢 LINKEDIN TOP COMPANY: {analysis['company_detected']} - {title[:50]}...")
                                
                except Exception as e:
                    logger.warning(f"LinkedIn query failed: {query} - {e}")
                    continue
            
            logger.info(f"✅ LinkedIn: {len(all_jobs)} jobs analyzed")
            return all_jobs[:120]
                
        except Exception as e:
            logger.error(f"LinkedIn search failed: {e}")
            return []


############################################  NEW APPROACH FOR JOB SEARCH  ############################################


    def _clean_json_response(self, text: str) -> str:
        """Clean JSON response to fix parsing issues"""
        # Remove markdown code blocks
        text = text.replace('```json', '').replace('```', '').strip()
        
        # Remove any leading/trailing non-JSON text
        lines = text.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('{') or line.startswith('['):
                in_json = True
            if in_json:
                json_lines.append(line)
            if line.endswith('}') or line.endswith(']'):
                break
        
        cleaned = '\n'.join(json_lines)
        
        # Ensure it starts and ends properly
        if not cleaned.startswith('{'):
            cleaned = '{' + cleaned
        if not cleaned.endswith('}'):
            cleaned = cleaned + '}'
        
        return cleaned

    def _quick_fallback(self, title: str) -> Dict[str, Any]:
        """Fast fallback for LLM failures"""
        title_lower = title.lower()
        
        # Quick keyword checks
        if any(term in title_lower for term in ['electrical', 'power', 'transmission', 'solar', 'wind']):
            category = "electrical"
        elif any(term in title_lower for term in ['civil', 'structural', 'construction', 'site engineer']):
            category = "civil"
        elif any(term in title_lower for term in ['software', 'developer', 'programmer', 'qa', 'tester']):
            category = "software"
        else:
            category = "other"
        
        # Simple scoring
        relevance = 0.6 if category != "other" else 0.3
        confidence = 0.7 if category != "other" else 0.5
        quality = 0.5
        
        return {
            "relevance_score": relevance,
            "relevance": relevance,
            "category": category,
            "confidence": confidence,
            "confidence_score": confidence,
            "quality": quality,
            "quality_score": quality,
            "combined_score": relevance * 0.4 + confidence * 0.3 + quality * 0.3
        }



    async def _llm_score_and_categorize_single_call(self, job_data: Dict, keywords: List[str] = None) -> Dict[str, Any]:
        """Single LLM call that does both scoring AND categorization - WITH PROFILE/BULK DETECTION"""
        try:
            model = get_gemini_model("gemini-2.0-flash-lite")
            
            title = job_data['title']
            source = job_data['source']

            logger.debug(f"📏 TITLE LENGTH: {len(title)} chars")
            logger.debug(f"📏 TITLE (first 100 chars): {title[:100]}...")
            
   
            
            prompt = f"""
            Analyze this job/posting and provide JSON with:
            1. relevance_score (0.0-1.0): Engineering job relevance
            2. category (electrical/civil/software/other): Job category
            3. confidence (0.0-1.0): Confidence in categorization
            4. quality (0.0-1.0): Job quality based on title clarity
            5. is_personal_profile (true/false): Is this a personal profile/looking for job?
            6. is_bulk_posting (true/false): Is this a bulk/mass recruitment?
            7. reasoning (brief explanation): Why you chose this categorization
            
            Title: {title}
            Source: {source}

            
            **CRITICAL SCORING RULES:**
            
            **PERSONAL PROFILE INDICATORS (Score DOWN 50% if true):**
            - Name followed by dash/hyphen: "John Doe - Engineer at Company"
            - "Looking for opportunities", "Open to work", "Seeking roles"
            - Personal pronouns: "I", "my", "me" in title/description
            - Contains "| Ex-" or "Former" indicating personal career history
            - Generic titles like "Contracts Manager", "General Manager" without specific engineering role
            - Social media style: "welcome kit from L&T" - not a job posting
            
            **BULK POSTING INDICATORS (Score DOWN 70% if true):**
            - "Multiple openings", "Bulk hiring", "Mass recruitment"
            - "Urgent requirement for 10+ engineers"
            - "Immediate joiners required" with quantity mentioned
            - Recruitment agency posting multiple roles
            
            **CATEGORY DEFINITIONS:**
            - electrical: electrical, power, transmission, substation, solar, wind, electrical engineer, power engineer
            - civil: civil, structural, construction, site engineer, building, structural engineer, civil engineer
            - software: software, developer, programmer, QA, tester, automation, software engineer
            - other: sales, HR, marketing
            
            **SCORING ADJUSTMENTS:**
            - PERSONAL PROFILE: Multiply final relevance_score by 0.5 (50% reduction)
            - BULK POSTING: Multiply final relevance_score by 0.3 (70% reduction)
        
            
            **Response format (JSON only):**
            {{
                "relevance_score": 0.85,
                "category": "software",
                "confidence": 0.90,
                "quality": 0.75,
                "is_personal_profile": false,
                "is_bulk_posting": false,
                "reasoning": "Title contains 'software developer' which clearly indicates software category"
            }}
            """
            
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean JSON response
            text = self._clean_json_response(text)
            
            try:
                result = json.loads(text)
                
                # Ensure all fields exist
                result["relevance_score"] = float(result.get("relevance_score", 0.5))
                result["category"] = result.get("category", "other").lower()
                result["confidence"] = float(result.get("confidence", 0.7))
                result["quality"] = float(result.get("quality", 0.6))
                result["is_personal_profile"] = bool(result.get("is_personal_profile", False))
                result["is_bulk_posting"] = bool(result.get("is_bulk_posting", False))
                result["reasoning"] = result.get("reasoning", "No reasoning provided")
                
                # **CRITICAL: Apply penalty for personal profiles and bulk postings**
                base_relevance = result["relevance_score"]
                
                if result["is_personal_profile"]:
                    logger.info(f"🔴 PERSONAL PROFILE detected: {title[:50]}...")
                    result["relevance_score"] = base_relevance * 0.5  # 50% reduction
                    result["reasoning"] += " | Penalized: Personal profile detected"
                    
                if result["is_bulk_posting"]:
                    logger.info(f"🔴 BULK POSTING detected: {title[:50]}...")
                    result["relevance_score"] = base_relevance * 0.3  # 70% reduction
                    result["reasoning"] += " | Penalized: Bulk posting detected"
                
                # # Ensure category is correct for management roles
                # title_lower = title.lower()
                # management_terms = ["manager", "head", "director", "lead", "agm", "dgm", "gm"]
                # engineering_terms = ["engineer", "design", "technical", "structural", "civil", "electrical", "software"]
                
                # # If it's a management role without clear engineering focus → "other"
                # if any(term in title_lower for term in management_terms):
                #     if not any(term in title_lower for term in engineering_terms):
                #         if result["category"] != "other":
                #             logger.info(f"🟡 MANAGEMENT ROLE → OTHER: {title[:50]}...")
                #             result["category"] = "other"
                #             result["relevance_score"] = result["relevance_score"] * 0.6  # Additional penalty
                #             result["reasoning"] += " | Management role without engineering focus"
                
                # Calculate combined score with penalties applied
                result["combined_score"] = (
                    result["relevance_score"] * 0.4 +
                    result["confidence"] * 0.3 +
                    result["quality"] * 0.3
                )
                
                # Add backward compatibility fields
                result["relevance"] = result["relevance_score"]
                result["confidence_score"] = result["confidence"]
                result["quality_score"] = result["quality"]
                
                # Log penalty details
                if result["is_personal_profile"] or result["is_bulk_posting"]:
                    logger.info(f"📉 Score adjusted: {title[:50]}...")
                    logger.info(f"   Original relevance: {base_relevance:.2f}, Final: {result['relevance_score']:.2f}")
                    logger.info(f"   Combined score: {result['combined_score']:.2f}")
                    logger.info(f"   Reason: {result['reasoning']}")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse failed: {text[:100]}...")
                fallback = self._quick_fallback(title)
                fallback["reasoning"] = f"JSON parse failed: {str(e)[:50]}"
                return fallback
                
        except Exception as e:
            logger.debug(f"LLM call failed: {e}")
            fallback = self._quick_fallback(job_data['title'])
            fallback["reasoning"] = f"LLM call failed: {str(e)[:50]}"
            return fallback
    async def fast_batch_llm_check(self, jobs: List[Dict], keywords: List[str] = None) -> List[Dict]:
        """Faster LLM processing with SINGLE call per job - WITH DETAILED LOGGING"""
        if not jobs:
            logger.info("🚨 No jobs to process in fast_batch_llm_check")
            return []
        
        batch_size = 12  # Increased from 5
        processed_jobs = []
        
        logger.info(f"🚀 STARTING OPTIMIZED LLM PROCESSING")
        logger.info(f"📊 Total jobs to process: {len(jobs)} (batch size: {batch_size})")
        
        # Log statistics about incoming jobs
        source_distribution = {}
        for job in jobs:
            source = job.get('source', 'unknown')
            source_distribution[source] = source_distribution.get(source, 0) + 1
        
        logger.info(f"📥 INPUT DISTRIBUTION: {source_distribution}")
        
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(jobs) + batch_size - 1) // batch_size
            
            logger.info(f"")
            logger.info(f"📦 BATCH {batch_num}/{total_batches}")
            logger.info(f"🔢 Processing {len(batch)} jobs in this batch")
            
            # Log individual jobs in this batch BEFORE processing
            logger.info("📋 JOBS IN THIS BATCH:")
            for idx, job in enumerate(batch):
                logger.info(f"   {idx+1}. [{job.get('source', 'Unknown')}] {job['title'][:70]}...")
            
            # Create tasks for this batch - SINGLE LLM CALL PER JOB
            tasks = []
            for job in batch:
                # Single combined LLM call instead of two separate calls
                task = self._llm_score_and_categorize_single_call(job, keywords)
                tasks.append(task)
            
            try:
                # Process batch in parallel
                logger.info(f"🤖 Sending batch to LLM for scoring...")
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                successful = 0
                failed = 0
                
                logger.info("")
                logger.info("📊 INDIVIDUAL JOB SCORING RESULTS:")
                logger.info("-" * 80)
                
                for job_idx, (job, result) in enumerate(zip(batch, batch_results)):
                    job_title_short = job['title'][:60] + "..." if len(job['title']) > 60 else job['title']
                    
                    if isinstance(result, Exception):
                        # LLM call failed
                        logger.warning(f"   ❌ JOB {job_idx+1} FAILED: {type(result).__name__}: {str(result)[:100]}")
                        logger.warning(f"      Title: {job_title_short}")
                        logger.warning(f"      Source: {job.get('source', 'Unknown')}")
                        
                        # Quick fallback
                        fallback_result = self._quick_fallback(job['title'])
                        job.update(fallback_result)
                        job["combined_score"] = 0.4
                        
                        # Log fallback details
                        logger.warning(f"      Using fallback: Category={fallback_result['category']}, Score={fallback_result['combined_score']:.2f}")
                        
                        failed += 1
                        processed_jobs.append(job)
                    else:
                        # LLM call succeeded
                        successful += 1
                        
                        # Update job with combined results
                        job.update(result)
                        
                        # Filter low-quality jobs
                        if result.get("combined_score", 0) >= 0.4:
                            processed_jobs.append(job)
                            status = "✅ ACCEPTED"
                        else:
                            status = "❌ REJECTED (low score)"
                        
                        # LOG INDIVIDUAL JOB DETAILS
                        logger.info(f"   {status} JOB {job_idx+1}:")
                        logger.info(f"      📝 Title: {job_title_short}")
                        logger.info(f"      🏷️  Source: {job.get('source', 'Unknown')}")
                        logger.info(f"      📊 Category: {result.get('category', 'N/A')}")
                        logger.info(f"      🔢 Scores: Relevance={result.get('relevance_score', 0):.2f}, "
                                f"Confidence={result.get('confidence', 0):.2f}, "
                                f"Quality={result.get('quality', 0):.2f}")
                        logger.info(f"      ⚡ Combined Score: {result.get('combined_score', 0):.2f}")
                        
                        # Log reasoning if available
                        if 'reasoning' in result and result['reasoning']:
                            logger.info(f"      💭 Reasoning: {result['reasoning'][:100]}...")
                        
           
                            
                        logger.info("")
                
                logger.info(f"   ✅ Batch {batch_num}: {successful}/{len(batch)} successful, {failed} failed")
                
                # Log batch statistics
                if successful > 0:
                    batch_scores = [r.get('combined_score', 0) for r in batch_results if not isinstance(r, Exception)]
                    avg_score = sum(batch_scores) / len(batch_scores) if batch_scores else 0
                    logger.info(f"   📈 Batch average score: {avg_score:.2f}")
                    
                    # Count categories in this batch
                    batch_categories = {}
                    for result in batch_results:
                        if not isinstance(result, Exception):
                            cat = result.get('category', 'unknown')
                            batch_categories[cat] = batch_categories.get(cat, 0) + 1
                    logger.info(f"   🏷️  Batch categories: {batch_categories}")
                
            except Exception as e:
                logger.error(f"   ❌ Batch {batch_num} error: {e}")
                for job in batch:
                    fallback = self._quick_fallback(job['title'])
                    job.update(fallback)
                    job["combined_score"] = 0.4
                    processed_jobs.append(job)
        
        # FINAL STATISTICS
        logger.info("")
        logger.info("=" * 80)
        logger.info("🎯 FINAL LLM PROCESSING SUMMARY")
        logger.info(f"📊 Input jobs: {len(jobs)}")
        logger.info(f"📊 Output jobs: {len(processed_jobs)}")
        logger.info(f"📊 Filtered out: {len(jobs) - len(processed_jobs)} (score < 0.4)")
        
        if processed_jobs:
            # Calculate final statistics
            final_scores = [j.get('combined_score', 0) for j in processed_jobs]
            avg_final_score = sum(final_scores) / len(final_scores)
            
            final_categories = {}
            final_sources = {}
            
            for job in processed_jobs:
                cat = job.get('category', 'unknown')
                src = job.get('source', 'unknown')
                final_categories[cat] = final_categories.get(cat, 0) + 1
                final_sources[src] = final_sources.get(src, 0) + 1
            
            logger.info(f"📈 Average combined score: {avg_final_score:.2f}")
            logger.info(f"🏷️  Category distribution: {final_categories}")
            logger.info(f"📱 Source distribution: {final_sources}")
            
            # Log top 5 highest scoring jobs
            sorted_jobs = sorted(processed_jobs, key=lambda x: x.get('combined_score', 0), reverse=True)
            logger.info("")
            logger.info("🏆 TOP 5 HIGHEST SCORING JOBS:")
            for i, job in enumerate(sorted_jobs[:5]):
                logger.info(f"   {i+1}. [{job.get('category', 'N/A')}] {job['title'][:50]}...")
                logger.info(f"      Score: {job.get('combined_score', 0):.2f} | Source: {job.get('source', 'Unknown')}")
        
        logger.info("=" * 80)
        logger.info("✅ LLM processing complete")
        
        return processed_jobs
    


    def _get_query_category(self, query: str, topic_queries: dict) -> str:
        """Helper to categorize which topic query found the job"""
        for category, queries in topic_queries.items():
            for q in queries:
                if q in query:
                    return category
        return "general"


    async def search_topic_specific(self, topic: str, num_results: int = 80, keywords: List[str] = None) -> List[Dict]:
        """MAIN SEARCH - Focus on category distribution without top company emphasis"""
        try:
            if topic.lower() in ["hiring", "hiring_jobs", "jobs"]:
                logger.info(f"🚀 COMBINED SEARCH: Regular + Company-specific")
                
                # Run ALL searches in parallel
                linkedin_jobs, naukri_jobs, company_jobs = await asyncio.gather(
                    self.search_linkedin_jobs_simple(keywords),
                    self.search_naukri_jobs_simple(keywords),
                    self.search_with_top_companies(),
                    return_exceptions=True
                )
                
                all_jobs = []
                
                # Combine results
                for source_name, result in zip(["LinkedIn", "Naukri", "CompanySearch"], 
                                            [linkedin_jobs, naukri_jobs, company_jobs]):
                    if isinstance(result, list):
                        all_jobs.extend(result)
                        logger.info(f"📊 {source_name}: {len(result)} jobs")
                    else:
                        logger.error(f"{source_name} failed: {result}")
                
                logger.info(f"📊 TOTAL COLLECTED: {len(all_jobs)} jobs")
                
                # Remove duplicates
                unique_jobs = []
                seen = set()
                for job in all_jobs:
                    title_key = job['title'].lower()[:100]
                    if title_key not in seen:
                        seen.add(title_key)
                        unique_jobs.append(job)
                
                # ====== FOCUS ON CATEGORY DISTRIBUTION ======
                # Categorize all jobs
                electrical_jobs = [j for j in unique_jobs if j.get('category') == 'electrical']
                civil_jobs = [j for j in unique_jobs if j.get('category') == 'civil']
                software_jobs = [j for j in unique_jobs if j.get('category') == 'software']
                other_jobs = [j for j in unique_jobs if j.get('category') == 'other']
                
                logger.info(f"📊 CATEGORY DISTRIBUTION:")
                logger.info(f"   Electrical: {len(electrical_jobs)}")
                logger.info(f"   Civil: {len(civil_jobs)}")
                logger.info(f"   Software: {len(software_jobs)}")
                logger.info(f"   Other (filtered out): {len(other_jobs)}")
                
                # Sort each category by relevance score
                electrical_jobs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                civil_jobs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                software_jobs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                
                # Target distribution (35 total)
                target_electrical = 16
                target_civil = 9
                target_software = 10
                
                # Take best from each category
                final_jobs = []
                final_jobs.extend(electrical_jobs[:target_electrical])
                final_jobs.extend(civil_jobs[:target_civil])
                final_jobs.extend(software_jobs[:target_software])
                
                # If we don't have enough, fill from remaining high-scoring jobs
                if len(final_jobs) < 35:
                    remaining = 35 - len(final_jobs)
                    all_remaining = (electrical_jobs[target_electrical:] + 
                                    civil_jobs[target_civil:] + 
                                    software_jobs[target_software:])
                    all_remaining.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                    final_jobs.extend(all_remaining[:remaining])
                
                # Final stats
                logger.info(f"✅ FINAL SELECTION: {len(final_jobs)} jobs")
                logger.info(f"📊 FINAL CATEGORY DISTRIBUTION:")
                final_electrical = sum(1 for j in final_jobs if j.get('category') == 'electrical')
                final_civil = sum(1 for j in final_jobs if j.get('category') == 'civil')
                final_software = sum(1 for j in final_jobs if j.get('category') == 'software')
                logger.info(f"   Electrical: {final_electrical}")
                logger.info(f"   Civil: {final_civil}")
                logger.info(f"   Software: {final_software}")
                
                return final_jobs[:num_results]
            
            else:
                # For non-job topics, use the enhanced search
                normalized_topic = topic.lower().replace(" ", "_")
                async with EnhancedNewsSearcher() as searcher:
                    return await searcher.batch_process_topic_news(normalized_topic, keywords)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


    # async def search_topic_specific(self, topic: str, num_results: int = 100, keywords: List[str] = None) -> List[Dict]:
    #     """MAIN SEARCH METHOD - Handles both jobs and other topics"""
    #     try:
    #         if topic.lower() in ["hiring", "hiring_jobs", "jobs"]:
    #             logger.info(f"🚀 STARTING JOB SEARCH with keywords: {keywords}")
                
    #             linkedin_jobs, naukri_jobs = await asyncio.gather(
    #                 self.search_linkedin_jobs(keywords),
    #                 self.search_naukri_jobs_fast(keywords),
               
    #                 return_exceptions=True
    #             )
                
    #             results = [linkedin_jobs, naukri_jobs]
    #             source_names = ["LinkedIn", "Naukri"]
                
    #             linkedin_jobs, naukri_jobs = results
                
    #             all_jobs = []
    #             source_counts = {}
                
    #             for source_name, result in zip(["LinkedIn", "Naukri"], [linkedin_jobs, naukri_jobs]):
    #                 if isinstance(result, list):
    #                     all_jobs.extend(result)
    #                     source_counts[source_name] = len(result)
    #                 else:
    #                     source_counts[source_name] = 0
    #                     logger.error(f"{source_name} search failed: {result}")
                
    #             logger.info(f"📊 COLLECTED: {len(all_jobs)} total jobs - {source_counts}")
    #             all_jobs = self.deduplicate_jobs(all_jobs)
    #             logger.info(f"📊 AFTER DEDUPLICATION: {len(all_jobs)} unique jobs - {source_counts}")


    #               # ============== CRITICAL FILTER: REMOVE "OTHER" CATEGORY JOBS ==============
    #             logger.info("🔍 FILTERING OUT NON-ENGINEERING ('other') JOBS...")
    #             engineering_jobs = []
    #             other_jobs = []
                
    #             for job in all_jobs:
    #                 category = job.get('category', 'unknown')
    #                 if category in ['electrical', 'civil', 'software']:
    #                     engineering_jobs.append(job)
    #                 else:
    #                     other_jobs.append(job)
                
    #             logger.info(f"📊 Engineering jobs: {len(engineering_jobs)} | Non-engineering (other): {len(other_jobs)}")
                
    #             # Log some examples of filtered out jobs
    #             if other_jobs:
    #                 logger.info("❌ FILTERED OUT NON-ENGINEERING JOBS (sample):")
    #                 for i, job in enumerate(other_jobs):  # Show first 5
    #                     logger.info(f"   {i+1}. [{job.get('category', 'N/A')}] {job['title']}...")
                
    #             # Use only engineering jobs for LLM filtering
    #             filtered_jobs = engineering_jobs
                
    #             # ============== END FILTER ==============
                    
    #             # ==== ADD DEBUG CODE HERE ==================================================================
    #             # logger.info("🔍 DEBUG - BEFORE LLM FILTERING:")
    #             # logger.info(f"   LinkedIn jobs: {len([j for j in all_jobs if j['source'] == 'LinkedIn'])}")
    #             # logger.info(f"   Naukri jobs: {len([j for j in all_jobs if j['source'] == 'Naukri.com'])}")
                
    #             # # Check LLM categories and scores
    #             # if all_jobs:
    #             #     categories = {}
    #             #     scores_by_source = {'LinkedIn': [], 'Naukri.com': []}
                    
    #             #     for job in all_jobs:
    #             #         cat = job.get('category', 'unknown')
    #             #         categories[cat] = categories.get(cat, 0) + 1
                        
    #             #         source = job.get('source', 'unknown')
    #             #         score = job.get('combined_score', 0)
    #             #         if source in scores_by_source:
    #             #             scores_by_source[source].append(score)
                    
    #             #     logger.info(f"📊 LLM Categories: {categories}")
                    
    #             #     # Log score statistics
    #             #     for source, scores in scores_by_source.items():
    #             #         if scores:
    #             #             avg_score = sum(scores) / len(scores)
    #             #             high_scores = sum(1 for s in scores if s >= 0.5)
    #             #             logger.info(f"📊 {source} Scores: Avg={avg_score:.2f}, ≥0.5={high_scores}/{len(scores)}")
                
    #             # ==== END DEBUG CODE ====++++++++++++++++++++++++++++++++++++++++++++++++
                
    #             if len(filtered_jobs) > 35:
    #                     logger.info(f"🤖 Sending {len(filtered_jobs)} ENGINEERING jobs to LLM for filtering...")
    #                     final_jobs = await self.llm_batch_filter_jobs(filtered_jobs, keywords=keywords, max_output_jobs=35)
    #             else:
    #                     final_jobs = filtered_jobs
    #                     logger.warning(f"⚠️  Only {len(filtered_jobs)} engineering jobs found, using all")

                
    #             final_source_count = {}
    #             for job in final_jobs:
    #                 final_source_count[job['source']] = final_source_count.get(job['source'], 0) + 1
                
    #             logger.info(f"✅ FINAL: {len(final_jobs)} curated jobs - {final_source_count}")
    #             return final_jobs[:num_results]
    #         else:
    #             # For non-job topics, use the enhanced search
    #             normalized_topic = topic.lower().replace(" ", "_")
    #             async with EnhancedNewsSearcher() as searcher:
    #                 return await searcher.batch_process_topic_news(normalized_topic, keywords)
            
    #     except Exception as e:
    #         logger.error(f"Search failed: {e}")
    #         return []



  
    def _basic_categorization_fallback(self, title: str) -> Dict[str, Any]:
        """Unified fallback for LLM failures"""
        return self._quick_fallback(title)

    def _quick_fallback(self, title: str) -> Dict[str, Any]:
        """Simplified fallback - just basic categorization, NO aggressive filtering"""
        title_lower = title.lower()
        
        # SIMPLE CATEGORIZATION ONLY - no complex filtering
        if any(term in title_lower for term in ['electrical', 'power', 'transmission', 'solar', 'wind']):
            category = "electrical"
            relevance = 0.5
        elif any(term in title_lower for term in ['civil', 'structural', 'construction', 'site']):
            category = "civil" 
            relevance = 0.5
        elif any(term in title_lower for term in ['software', 'developer', 'programmer', 'devops']):
            category = "software"
            relevance = 0.5
        else:
            category = "other"
            relevance = 0.3
        
        # SIMPLE scores
        confidence = 0.6  # Lower confidence because it's fallback
        quality = 0.5
        combined = (relevance * 0.4 + confidence * 0.3 + quality * 0.3)
        
        return {
            "relevance_score": relevance,
            "relevance": relevance,
            "category": category,
            "confidence": confidence,
            "confidence_score": confidence,
            "quality": quality,
            "quality_score": quality,
            "combined_score": combined,
            "is_personal_profile": False,  # Don't try to detect in fallback
            "is_bulk_posting": False,      # Don't try to detect in fallback
            "reasoning": "Simple keyword fallback categorization"
        }

class EnhancedNewsSearcher(EnergyNewsSearcher):
    def __init__(self):
        super().__init__()
        self.topic_sources = {
            "india_power_projects": [
                "projectstoday.com", 
                "economictimes.indiatimes.com/industry/energy/power",
                "moneycontrol.com/news/business/power", 
                "business-standard.com/industry/energy",
                "financialexpress.com/industry/energy", 
                "livemint.com/industry/energy",
                "thehindu.com/business/energy", 
                "reuters.com/business/energy",
                "economic times energy", "power grid india", "ntpc", "nhpc",
                "solar power india", "wind energy india", "renewable energy india"
            ]
        }

    def validate_and_clean_url(self, url: str) -> str:
        """Class method version of URL validation"""
        if not url:
            return "#"
        
        # Convert Google News RSS URLs to proper article URLs
        if 'news.google.com/rss/articles/' in url:
            try:
                article_match = re.search(r'/articles/([^?]+)', url)
                if article_match:
                    article_id = article_match.group(1)
                    cleaned_url = f"https://news.google.com/articles/{article_id}?hl=en-IN&gl=IN&ceid=IN:en"
                    logger.info(f"✅ Converted RSS URL: {cleaned_url}")
                    return cleaned_url
            except Exception as e:
                logger.warning(f"URL conversion failed for {url}: {e}")
        
        # Return original URL if no conversion needed
        return url

    def extract_real_article_url(self, entry) -> str:
        """PROPERLY convert Google News RSS URLs to real article URLs"""
        link = getattr(entry, 'link', '')
        return self.validate_and_clean_url(link)

    async def search_general_topic(self, topic: str, keywords: List[str] = None) -> List[Dict]:
        """Generic search for technology, sports, business, etc. – now with proper diversity"""
        all_news = []
        
        # Expanded & balanced queries - ENHANCED FOR SPORTS DIVERSITY
        topic_queries = {
            "sports": [
                # Cricket + Football + Other sports
                "cricket OR IPL OR BCCI OR ICC OR WPL when:2d",
                "badminton OR kabaddi OR chess India when:2d",
                "football OR ISL OR AIFF OR FIFA when:2d", 
                "hockey OR badminton OR tennis OR chess OR kabaddi India when:2d",
                "sports business OR sports policy OR sports ministry India when:2d",
                "Olympics OR Commonwealth Games OR Asian Games when:2d"
            ],
            "technology": [
            '(technology OR AI OR "artificial intelligence" OR "machine learning" OR gadget OR smartphone OR "startup India") OR (from:Reuters technology) OR (from:Reuters tech) OR from:TechCrunch OR from:TheVerge when:2d',

            '(blockchain OR cryptocurrency OR fintech OR edtech OR healthtech OR "startup India" OR NASDAQ OR "Bloomberg Tech") OR (from:Reuters technology) OR (from:Reuters tech) OR from:TechCrunch OR from:TheVerge when:2d',

            '(cybersecurity OR "data privacy" OR "cloud computing" OR 5G OR "quantum computing" OR IoT OR "TimesOfIndia Tech") OR (from:Reuters technology) OR (from:Reuters tech) OR from:TechCrunch OR from:TheVerge when:2d',

            '(web3 OR metaverse OR NFT OR "virtual reality" OR "augmented reality" OR robotics OR "MoneyControl Tech") OR (from:Reuters technology) OR (from:Reuters tech) OR from:TechCrunch OR from:TheVerge when:2d',

            '(semiconductors OR chips OR Apple OR Google OR Microsoft OR (Amazon AND tech) OR (Amazon AND innovation)) OR (from:Reuters technology) OR (from:Reuters tech) OR from:TechCrunch OR from:TheVerge when:2d'
            ],
            "business": [
                "startup OR funding OR IPO OR stock market OR sensex OR nifty OR unicorn India when:2d",
                "moneycontrol OR economic times OR business standard OR livemint when:2d"
            ]
        }
        
        queries = topic_queries.get(topic.lower(), [
            f"{topic} India when:2d",
            f"{topic} news when:2d"
        ])
        
        # Add user keywords if any
        if keywords:
            extra = " OR ".join(keywords[:5])
            queries.append(f"({extra}) when:2d")
        
        session = await self.get_session()

        for query in queries:
            try:
                url = "https://news.google.com/rss/search"
                params = {
                    "q": query,
                    "hl": "en-IN",
                    "gl": "IN", 
                    "ceid": "IN:en"
                }
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        feed = feedparser.parse(text)
                        for entry in feed.entries[:11]:
                            real_url = self.extract_real_article_url(entry)
                            
                            news_item = {
                                "title": entry.title,
                                "url": real_url,
                                "content": entry.get("summary", "")[:400],
                                "source": self.get_source_name(real_url),
                                "published_at": entry.get("published", ""),
                                "type": "news_article",
                                "relevance_score": 0.8,
                                "topic": topic
                            }
                            all_news.append(news_item)
            except Exception as e:
                logger.warning(f"Query failed for {topic}: {query} - {e}")
                continue
        
        # Apply sports-specific filtering
        if topic.lower() == "sports":
            logger.info(f"🏀 Applying sports filtering to {len(all_news)} raw sports news")
            all_news = self.deduplicate_jobs(all_news)
            logger.info(f"🏀 After basic deduplication: {len(all_news)} news")
        else:
            all_news = self.deduplicate_jobs(all_news)
    
        logger.info(f"Collected {len(all_news)} articles for topic: {topic}")
        return all_news[:60]
        
    async def search_projects_today_direct(self) -> List[Dict]:
        """Fixed Projects Today scraping with proper session management"""
        urls_to_try = [
            "https://www.projectstoday.com/News/NewsList.aspx",
            "http://www.projectstoday.com/News/NewsList.aspx",
            "https://projectstoday.com/News/NewsList.aspx"
        ]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        session = await self.get_session()
        
        for url in urls_to_try:
            try:
                logger.info(f"🔍 Trying Projects Today: {url}")
                
                async with session.get(url, headers=headers, timeout=35) as resp:
                    logger.info(f"📡 Projects Today response status: {resp.status}")
                    
                    if resp.status == 200:
                        html = await resp.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        projects = []
                        
                        news_items = soup.find_all('div', class_='discription')
                        logger.info(f"📰 Found {len(news_items)} raw news items")
                        
                        for i, item in enumerate(news_items[:50]):
                            try:
                                title_link = item.find('h3').find('a') if item.find('h3') else None
                                if not title_link:
                                    continue
                                    
                                title = title_link.get_text(strip=True)
                                relative_link = title_link.get('href', '')
                                
                                if not relative_link or relative_link == "#":
                                    continue
                                
                                date_span = item.find('span', class_='date')
                                date_text = date_span.get_text(strip=True) if date_span else ""
                                
                                # URL construction
                                if relative_link.startswith('../'):
                                    full_link = f"https://www.projectstoday.com{relative_link[2:]}"
                                elif relative_link.startswith('/'):
                                    full_link = f"https://www.projectstoday.com{relative_link}"
                                else:
                                    full_link = f"https://www.projectstoday.com/{relative_link}"
                                
                                # Content extraction
                                content = ""
                                for p in item.find_all('p'):
                                    if p.find('span', class_='date'):
                                        continue
                                    p_text = p.get_text(strip=True)
                                    if p_text and p_text != title and len(p_text) > 20:
                                        content = p_text
                                        break
                                
                                if not content:
                                    content = f"Project update: {title}"
                                
                                project_data = {
                                    "title": title,
                                    "url": full_link,
                                    "content": content[:400],
                                    "source": "Projects Today",
                                    "published_at": date_text,
                                    "type": "project_news", 
                                    "relevance_score": 0.95,
                                    "topic": "india_power_projects"
                                }
                                projects.append(project_data)
                                logger.info(f"✅ Added: {title[:50]}...")
                                    
                            except Exception as e:
                                logger.warning(f"⚠️ Error processing item {i}: {e}")
                                continue
                        
                        logger.info(f"🎯 Projects Today SUCCESS: {len(projects)} news scraped")
                        return projects
                        
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout for {url}, trying next...")
                continue
            except Exception as e:
                logger.warning(f"⚠️ Failed for {url}: {e}, trying next...")
                continue
        
        logger.error("🚨 All Projects Today URLs failed")
        return []

    async def search_topic_news_enhanced(self, topic: str, keywords: List[str] = None) -> List[Dict]:
        """Enhanced news search with PROPER session handling - NO LEAKS"""
        all_news = []
        search_queries = [
            # 1. New project announcements, updates, commissioning
            "(India power project OR new power project OR project commissioned OR project inauguration OR capacity addition) India when:2d",

            # 2. Major tenders and contract awards
            "(power tender awarded OR EPC contract power OR transmission tender OR solar tender India OR wind project tender) when:2d",

            # 3. Transmission & distribution projects
            "(power transmission project OR transmission line project OR substation project OR smart grid India OR distribution company project) India when:2d",

            # 4. DISCOM updates & reforms
            "(DISCOM reforms OR distribution company India OR power tariff order OR electricity amendment India) when:2d",

            # 5. Renewable energy developments
            "(renewable energy India OR solar energy project India OR wind energy project India OR hydroelectric project India OR green hydrogen India) when:2d",

            # 6. PSU and private sector power companies
            "(Tata OR Adani  OR JSW Energy OR RPSG OR NHPC OR L&T OR Torrent Power OR ReNew OR KEC OR STERLITE ) project when:2d",

            # 7. Government policy & investment announcements
            "(energy policy India OR government power scheme OR energy investment India OR budget power sector OR PLI scheme renewable) when:2d",

            # 8. Major partnerships and financing
            "(MoU power project OR joint venture power OR funding power project OR investment renewable energy India) when:2d"
        ]


        session = await self.get_session()

        for query in search_queries:
            try:
                url = "https://news.google.com/rss/search"
                params = {
                    "q": query,
                    "hl": "en-IN",
                    "gl": "IN",
                    "ceid": "IN:en"
                }
               
                logger.info(f"Searching with query: {query}")
               
                async with session.get(url, params=params, timeout=35) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        feed = feedparser.parse(text)
                       
                        logger.info(f"Found {len(feed.entries)} entries for query: {query}")
                       
                        for entry in feed.entries[:13]:
                            if 'projectstoday.com' in entry.link:
                                continue
                               
                            news_item = {
                                "title": entry.title,
                                "url": entry.link,
                                "content": entry.get("summary", "")[:350],
                                "source": self.get_source_name(entry.link),
                                "published_at": entry.get("published", ""),
                                "type": "news_article",
                                "relevance_score": 0.7,
                                "topic": topic
                            }
                            all_news.append(news_item)
                    else:
                        logger.warning(f"Google News returned status {resp.status} for query: {query}")
                       
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for query: {query}")
                continue
            except Exception as e:
                logger.warning(f"Query failed: {query} - {e}")
                continue
       
        logger.info(f"Collected {len(all_news)} news from enhanced search")
        return all_news[:100]
    
    async def llm_batch_filter_news(self, all_news: List[Dict], topic: str, max_output: int = 35):
        """LLM filtering with intelligent duplicate removal"""
        try:
            if len(all_news) <= max_output:
                return all_news[:max_output]
                
            model = genai.GenerativeModel('gemini-2.0-flash-lite')
            
            # Separate Projects Today news
            projects_today_news = [news for news in all_news if news.get('source') == 'Projects Today']
            other_news = [news for news in all_news if news.get('source') != 'Projects Today']
            
            # LOG BEFORE LLM INPUT
            logger.info(f"📊 INPUT TO LLM: {len(projects_today_news)} Projects Today + {len(other_news)} other news = {len(all_news)} total")
            
            # Prepare news for LLM - KEEP ALL
            news_text = "\n\n".join([
                f"NEWS {i+1} (Source: {news['source']}):\nTitle: {news['title']}\nContent: {news['content'][:300]}"
                for i, news in enumerate(all_news)
            ])
            
            prompt = f"""
            CRITICAL: Select exactly {max_output} most relevant and UNIQUE news about Indian power projects.

            PROJECTS TODAY HANDLING:
            - There are {len(projects_today_news)} Projects Today articles available
            - You MUST include ALL {len(projects_today_news)} Projects Today articles if they are ≤10
            - If there are more than 10 Projects Today articles, select the BEST 10 most important ones
            - Projects Today articles are direct project updates and have highest priority

            INTELLIGENT DUPLICATE REMOVAL - CRITICAL RULES:
            - SAME STORY, DIFFERENT SOURCES: If multiple sources cover the same news event (like "ADB $800 million loan"), pick ONLY THE BEST/COMPLETE VERSION
            - SAME PROJECT UPDATES: If same project is reported by multiple sources, keep only the most authoritative source
            - SIMILAR ANNOUNCEMENTS: Different stories about same type of projects (multiple solar tenders) are OK if they are about DIFFERENT projects
            - SAME COMPANY NEWS: Multiple stories about same company are OK if they cover DIFFERENT projects/announcements

            DUPLICATE PATTERNS TO WATCH FOR:
            - "ADB $800 million loan" variations → KEEP ONLY ONE
            - "Tata Power investment" same announcement → KEEP ONLY ONE  
            - "Solar project with GUVNL" same project → KEEP ONLY ONE
            - Different projects by same company → KEEP ALL (they are unique)

            HOW TO IDENTIFY DUPLICATES:
            - Check if core event/announcement is the same
            - Check if same project name/company is mentioned
            - Check if same financial amount/numbers are repeated
            - Different angles on same event = DUPLICATE
            - Different events involving same company = UNIQUE

            SELECTION PRIORITY:
            1. Projects Today articles (direct project updates)
            2. New project announcements and commissions
            3. Major tenders and contracts awarded
            4. Distribution and transmission projects
            5. State Distribution Companies (DISCOMs) updates.
            6. Renewable energy developments (solar, wind, hydro)
            7. Government energy policies and initiatives
            8. Major investments and partnerships

            SOURCE DISTRIBUTION:
            - Projects Today: {len(projects_today_news) if len(projects_today_news) <= 10 else 10} articles
            - Other sources: {max_output - (len(projects_today_news) if len(projects_today_news) <= 10 else 10)} articles
            TOTAL: {max_output} UNIQUE news items

            AVAILABLE NEWS:
            {news_text}

            RESPONSE FORMAT: Return ONLY comma-separated numbers like "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35"

            IMPORTANT: 
            - You MUST include {"ALL" if len(projects_today_news) <= 10 else "EXACTLY 10"} Projects Today articles.
            - Be RUTHLESS with duplicates - each story should be UNIQUE and DISTINCT
            - Prefer authoritative sources for the same story
            - Ensure good diversity across different power sector areas
            """

            response = model.generate_content(prompt)
            selected_numbers_text = response.text.strip()
            
            selected_indices = []
            numbers_found = re.findall(r'\d+', selected_numbers_text)
            
            for num_str in numbers_found:
                idx = int(num_str) - 1
                if 0 <= idx < len(all_news) and idx not in selected_indices:
                    selected_indices.append(idx)
            
            selected_news = [all_news[i] for i in selected_indices[:max_output]]
            
            # LOG AFTER LLM OUTPUT - FINAL RESULT
            final_pt_count = sum(1 for n in selected_news if n.get('source') == 'Projects Today')
            final_other_count = len(selected_news) - final_pt_count
            
            logger.info(f"📊 OUTPUT FROM LLM: {final_pt_count} Projects Today + {final_other_count} other news = {len(selected_news)} total")
            
            # Validate Projects Today count
            expected_pt_count = len(projects_today_news) if len(projects_today_news) <= 10 else 10
            if final_pt_count != expected_pt_count:
                logger.warning(f"⚠️  LLM didn't follow Projects Today instructions: Expected {expected_pt_count}, got {final_pt_count}")

            # FINAL SUCCESS LOG
            logger.info(f"🎯 FINAL SELECTION: {len(selected_news)} curated news - {final_pt_count} Projects Today + {final_other_count} other sources")
            
            return selected_news
                
        except Exception as e:
            logger.error(f"🚨 LLM news filtering failed: {e}")
            # Fallback with Projects Today protection
            projects_today_news = [news for news in all_news if news.get('source') == 'Projects Today']
            other_news = [news for news in all_news if news.get('source') != 'Projects Today']
            
            # Include all Projects Today if ≤10, otherwise take first 10
            pt_to_include = projects_today_news if len(projects_today_news) <= 10 else projects_today_news[:10]
            other_to_include = other_news[:max_output - len(pt_to_include)]
            
            result = pt_to_include + other_to_include
            logger.info(f"🔄 FALLBACK: {len(pt_to_include)} Projects Today + {len(other_to_include)} other news = {len(result)} total")
            
            return result[:max_output]

    def remove_duplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """Remove duplicate news based on title similarity"""
        unique_news = []
        seen_titles = set()
        
        for news in news_list:
            title = news['title'].lower()
            title_key = re.sub(r'[^\w\s]', '', title)
            title_key = ' '.join(title_key.split()[:8])
            
            duplicate_checks = [
                title_key,
                title_key.replace('power grid', 'powergrid'),
                title_key.replace('bc jindal', 'b c jindal'),
                title_key.replace('mnre', 'ministry of renewable energy'),
            ]
            
            is_duplicate = any(check in seen_titles for check in duplicate_checks)
            
            if not is_duplicate:
                seen_titles.add(title_key)
                unique_news.append(news)
            else:
                logger.info(f"🔍 Removed duplicate: {news['title'][:60]}...")
        
        return unique_news

    async def batch_process_topic_news(self, topic: str, keywords: List[str] = None) -> List[Dict]:
        """Main method - ensures we get and format ALL 35 news properly"""
        try:
            logger.info(f"🚀 STARTING {topic.upper()} NEWS SEARCH - TARGET: 35 NEWS")

            if topic.lower().replace(" ", "_") == "india_power_projects":
                # Run both searches in parallel
                projects_today_news, enhanced_news = await asyncio.gather(
                    self.search_projects_today_direct(),
                    self.search_topic_news_enhanced(topic, keywords),
                    return_exceptions=True
                )

                all_news = []
                
                if isinstance(projects_today_news, list):
                    all_news.extend(projects_today_news)
                    logger.info(f"Projects Today: {len(projects_today_news)} news")
                
                if isinstance(enhanced_news, list):
                    all_news.extend(enhanced_news)
                    logger.info(f"Enhanced Search: {len(enhanced_news)} news")
                
                # Remove duplicates based on title
                all_news = self.deduplicate_jobs(all_news)
                logger.info(f"📊 COLLECTED: {len(all_news)} total news after deduplication")
                
                # Apply LLM filtering to get exactly 35
                if len(all_news) > 35:
                    final_news = await self.llm_batch_filter_news(all_news, topic, 35)
                else:
                    final_news = all_news
                    logger.warning(f"Only {len(all_news)} unique news found")
                
                logger.info(f"✅ FINAL: {len(final_news)} curated power news")
                return final_news
            else:
                # For other topics like sports/technology
                return await self.search_general_topic(topic, keywords)
            
        except Exception as e:
            logger.error(f"Topic news processing failed: {e}")
            return []

    def get_source_name(self, url: str) -> str:
        """Extract source name from URL"""
        try:
            domain = urlparse(url).netloc
            
            if not domain:
                return "News Source"
                
            domain = domain.replace('www.', '').replace('news.', '')
            
            # Handle Google News URLs specifically
            if 'news.google.com' in domain:
                return "Google News"
                
            # Extract main domain
            main_domain_parts = domain.split('.')
            if len(main_domain_parts) >= 2:
                # Get the actual domain name (not TLD)
                main_domain = main_domain_parts[-2] if main_domain_parts[-2] not in ['co', 'com'] else main_domain_parts[-3]
                return main_domain.title()
            else:
                return domain.title()
                
        except Exception as e:
            logger.error(f"Error extracting source name from {url}: {e}")
            return "News Source"

# ============================
# SEPARATE TEMPLATE FUNCTIONS
# ============================
def format_sources_to_content(sources: List[Dict]) -> str:
    """Convert already-curated sources to newsletter content format - NO LLM"""
    content = "## ⚡ POWER PROJECTS UPDATE\n\n"
    
    # Use all curated sources - they're already high quality
    for source in sources:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    return content
async def process_with_killer_formatting(topic: str, sources: List[Dict], publish_date: str, keywords: List[str], style: str):
    try:
        if topic.lower() in ["hiring", "hiring_jobs", "jobs"]:
            content = generate_manual_job_newsletter(sources)
            html, words, read_time = create_jobs_newsletter(topic, content, sources, publish_date)
        elif topic.lower() == "sports":
            content = await generate_sports_newsletter_content(sources, topic)
            html, words, read_time = create_sports_newsletter(topic, content, sources, publish_date)
        elif topic.lower() == "technology":
            content = await generate_tech_newsletter_content(sources, topic)
            html, words, read_time = create_tech_newsletter(topic, content, sources, publish_date)
        elif topic.lower() in ["india_power_projects", "india power projects"]:
            # content = await generate_power_newsletter_content(sources, topic)
            content = format_sources_to_content(sources)
            html, words, read_time = create_power_newsletter(topic, content, sources, publish_date)
        else:
            content = await generate_simple_newsletter_content(sources, topic)
            html, words, read_time = create_consistent_newsletter(topic, content, sources, publish_date)
        
        return {
            "html": html,
            "metadata": {
                "topic": topic,
                "publish_date": publish_date,
                "sources_used": len(sources),
                "real_sources": True,
                "word_count": words,
                "estimated_read_time": read_time
            },
            "raw_sources": sources
        }
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return create_basic_newsletter(topic, sources, publish_date)

# ============================
# SPORTS NEWSLETTER FUNCTIONS
# ============================

async def generate_sports_newsletter_content(sources: List[Dict], topic: str) -> str:
    """Generate sports newsletter with comprehensive debugging"""
    try:
        logger.info("🔧 STARTING COMPREHENSIVE DEBUGGING")
        
        model = get_gemini_model("gemini-2.0-flash-lite")
        
        # Step 1: Check initial sources
        logger.info(f"📥 INITIAL SOURCES: {len(sources)}")
        logger.info("📋 SAMPLE INITIAL SOURCES (first 5):")
        for i, source in enumerate(sources[:5]):
            logger.info(f"  {i+1}. {source['title'][:80]}...")
        
        unique_sources = remove_duplicate_news_sports(sources)
        logger.info(f"🔄 AFTER BASIC DEDUPLICATION: {len(sources)} → {len(unique_sources)}")
        
        target_count = 35
        
        # Step 2: Validate URLs and check specific stories
        validated_sources = []
        specific_stories_to_check = ["Faf du Plessis", "WPL", "Commonwealth Games", "Srihari Nataraj", "Syed Modi"]
        
        logger.info("🔍 CHECKING SPECIFIC STORIES IN VALIDATED SOURCES:")
        for source in unique_sources:
            validated_source = source.copy()
            validated_source['url'] = validate_and_clean_url_standalone(source['url'])
            validated_sources.append(validated_source)
            
            # Check for specific stories
            title_lower = source['title'].lower()
            for story in specific_stories_to_check:
                if story.lower() in title_lower:
                    logger.info(f"   ✅ FOUND '{story}': {source['title'][:60]}...")
        
        sources_for_llm = validated_sources
        
        # Step 3: Detailed analysis of sources being sent to LLM
        logger.info("📊 DETAILED SOURCE ANALYSIS FOR LLM:")
        logger.info(f"📤 SENDING {len(sources_for_llm)} SOURCES TO LLM")
        
        # Count by category
        cricket_count = sum(1 for s in sources_for_llm if any(word in s['title'].lower() for word in ['cricket', 'ipl', 'psl', 'odi', 't20']))
        football_count = sum(1 for s in sources_for_llm if any(word in s['title'].lower() for word in ['football', 'soccer', 'fifa', 'premier']))
        other_count = len(sources_for_llm) - cricket_count - football_count
        
        logger.info(f"🏏 Cricket sources: {cricket_count}")
        logger.info(f"⚽ Football sources: {football_count}")
        logger.info(f"🏸 Other sports sources: {other_count}")
        
        # Step 4: Create URL mapping with verification
        url_map = {}
        logger.info("🔗 CREATING URL MAPPING:")
        for i, source in enumerate(sources_for_llm):
            short_id = f"URL_{i+1}"
            url_map[short_id] = source['url']
            if i < 5:  # Log first 5 mappings
                logger.info(f"   {short_id} → {source['title'][:50]}...")
        
        # Step 5: Prepare news text for LLM
        news_text = "\n".join([
            f"{i+1}. {news['title']} | URL: URL_{i+1}"
            for i, news in enumerate(sources_for_llm)
        ])
        
        # Step 6: Enhanced prompt with clear instructions
        prompt = f"""
        CRITICAL: You are a sports editor selecting the TOP 35 most important and DIVERSE sports stories from {len(validated_sources)} options.

        MUST SELECT EXACTLY 35 UNIQUE STORIES - NO DUPLICATES, NO SIMILAR STORIES.

        DUPLICATE REMOVAL RULES:
        - SAME EVENT: If multiple sources cover the same match/event (e.g., "India vs South Africa ODI"), pick ONLY THE BEST ONE
        - SAME ANNOUNCEMENT: If multiple sources report the same news (e.g., "Faf du Plessis IPL"), pick ONLY THE MOST COMPREHENSIVE
        - SAME TOURNAMENT: Don't include multiple routine updates from the same tournament

        PRIORITY ORDER FOR SELECTION:
        1. BREAKING NEWS: Major announcements, tournament results, record-breaking performances
        2. INDIAN SPORTS: Stories about Indian athletes/teams get priority
        3. DIVERSITY: Ensure good mix across different sports
        4. AUTHORITATIVE SOURCES: Prefer established sports media over generic news

        SPECIFIC DUPLICATES TO WATCH FOR:
        - Multiple "India vs South Africa" cricket matches → PICK ONLY 1-2 BEST
        - Multiple "Syed Modi Badminton" updates → PICK ONLY THE KEY STORIES
        - Multiple "Commonwealth Games" stories → PICK THE MAIN ANNOUNCEMENT
        - Multiple "Srihari Nataraj" stories → PICK THE MOST SIGNIFICANT

        SECTION DISTRIBUTION (35 TOTAL):
        ## 🏏 CRICKET UPDATES (10-12 items)
        - Focus on major matches, key player news, important tournaments
        - Avoid multiple similar match reports

        ## ⚽ FOOTBALL NEWS (8-10 items)  
        - Include international, Indian football, major leagues
        - Mix of match results and important news

        ## 🏸 OTHER SPORTS (10-12 items)
        - Badminton, Hockey, Chess, Swimming, etc.
        - Ensure diversity across different sports

        ## 🏆 SPORTS BUSINESS (3-5 items)
        - Major announcements, infrastructure, policy changes

        FORMATTING:
        [Complete Title](URL_ID)

        AVAILABLE SOURCES:
        {news_text}

        IMPORTANT: You MUST output EXACTLY 35 items total across all 4 sections. 
        Be ruthless in removing duplicates and similar stories.
        If you find multiple stories about the same specific event, pick THE BEST ONE and discard the rest.
        """
        
        logger.info(f"📝 PROMPT LENGTH: {len(prompt)} characters")
        logger.info("🔍 SENDING REQUEST TO LLM...")
        
        # Step 7: LLM call with timing
        import time
        start_time = time.time()
        response = model.generate_content(prompt)
        llm_time = time.time() - start_time
        logger.info(f"⏱️ LLM RESPONSE TIME: {llm_time:.2f}s")
        
        content = response.text.strip()
        
        # Step 8: Analyze LLM response
        logger.info("📄 LLM RAW RESPONSE ANALYSIS:")
        logger.info(f"📏 Response length: {len(content)} characters")
        logger.info(f"🔍 First 300 chars: {content[:300]}...")
        
        # Count sections in response
        sections_found = content.count('## 🏏') + content.count('## ⚽') + content.count('## 🏸') + content.count('## 🏆')
        logger.info(f"📑 Sections found in response: {sections_found}/4")
        
        # Step 9: URL restoration with detailed tracking
        logger.info("🔄 STARTING URL RESTORATION...")
        original_content = content
        restoration_log = []
        
        for short_id, full_url in url_map.items():
            if f"]({short_id})" in content:
                content = content.replace(f"]({short_id})", f"]({full_url})")
                # Find which title this URL corresponds to
                source_idx = int(short_id.split('_')[1]) - 1
                if source_idx < len(sources_for_llm):
                    title = sources_for_llm[source_idx]['title'][:40]
                    restoration_log.append(f"{short_id} → {title}...")
        
        logger.info(f"✅ URL RESTORATION COMPLETE: {len(restoration_log)} URLs replaced")
        if restoration_log:
            logger.info("📋 RESTORED URLS (first 10):")
            for log_entry in restoration_log[:10]:
                logger.info(f"   {log_entry}")
        
        remaining_short_ids = content.count('URL_')
        logger.info(f"🔍 Remaining short IDs after restoration: {remaining_short_ids}")
        
        # Step 10: Final content analysis
        logger.info("📊 FINAL CONTENT ANALYSIS:")
        import re
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        link_count = len(markdown_links)
        
        logger.info(f"📎 Total markdown links found: {link_count}")
        logger.info(f"🎯 Target count: {target_count}")
        
        # Analyze link distribution
        cricket_links = sum(1 for title, _ in markdown_links if any(word in title.lower() for word in ['cricket', 'ipl', 'psl', 'odi']))
        football_links = sum(1 for title, _ in markdown_links if any(word in title.lower() for word in ['football', 'soccer']))
        business_links = sum(1 for title, _ in markdown_links if any(word in title.lower() for word in ['business', 'commonwealth', 'olympic', 'infrastructure']))
        other_links = link_count - cricket_links - football_links - business_links
        
        logger.info(f"🏏 Cricket links: {cricket_links}")
        logger.info(f"⚽ Football links: {football_links}") 
        logger.info(f"🏆 Business links: {business_links}")
        logger.info(f"🏸 Other sports links: {other_links}")
        
        # Check for specific stories in final output
        logger.info("🔍 CHECKING SPECIFIC STORIES IN FINAL OUTPUT:")
        found_stories = {}
        for story in specific_stories_to_check:
            count = sum(1 for title, _ in markdown_links if story.lower() in title.lower())
            found_stories[story] = count
            logger.info(f"   '{story}': {count} occurrences")
        
        # Check URL integrity
        proper_urls_count = sum(1 for _, url in markdown_links if not url.startswith('URL_'))
        logger.info(f"🔗 Proper URLs: {proper_urls_count}/{link_count}")
        
        # Step 11: Log sample of final output
        logger.info("📋 FINAL OUTPUT SAMPLE (first 10 items):")
        for i, (title, url) in enumerate(markdown_links[:10]):
            logger.info(f"   {i+1}. {title[:50]}...")
            logger.info(f"      URL: {url[:80]}...")
        
        content = content.strip()
        
        # Step 12: Final validation
 # Step 12: Final validation - FIXED VERSION
        variance_allowed = 4  # Allow ±4 variance

        if abs(link_count - target_count) <= variance_allowed and proper_urls_count == link_count:
            logger.info(f"🎯 SUCCESS: Newsletter generated with {link_count} items (target: {target_count}±{variance_allowed})")
            return content
        else:
            logger.warning(f"⚠️  Using fallback: Got {link_count}/{target_count} links, {proper_urls_count} proper URLs")
            return force_exact_sports_count(validated_sources, topic, target_count)
        
    except Exception as e:
        logger.error(f"❌ SPORTS NEWSLETTER FAILED: {e}")
        import traceback
        logger.error(f"🔍 FULL ERROR TRACEBACK:\n{traceback.format_exc()}")
        return force_exact_sports_count(sources[:target_count], topic, target_count)


def force_exact_sports_count(sources: List[Dict], topic: str, target_count: int) -> str:
    """Fallback method with debugging"""
    logger.info("🔄 USING FALLBACK SELECTION METHOD")
    logger.info(f"📥 Fallback input: {len(sources)} sources")
    
    # Categorize sources for fallback
    cricket = []
    football = []
    other_sports = []
    business = []
    
    for source in sources:
        title_lower = source['title'].lower()
        if any(word in title_lower for word in ['cricket', 'ipl', 'psl', 'odi', 't20']):
            cricket.append(source)
        elif any(word in title_lower for word in ['football', 'soccer']):
            football.append(source)
        elif any(word in title_lower for word in ['business', 'commonwealth', 'olympic', 'infrastructure']):
            business.append(source)
        else:
            other_sports.append(source)
    
    logger.info(f"🏏 Fallback cricket: {len(cricket)}")
    logger.info(f"⚽ Fallback football: {len(football)}")
    logger.info(f"🏸 Fallback other: {len(other_sports)}")
    logger.info(f"🏆 Fallback business: {len(business)}")
    
    # Build content (your existing fallback logic)
    content = "## 🏏 CRICKET UPDATES\n\n"
    for source in cricket[:8]:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    content += "## ⚽ FOOTBALL NEWS\n\n"
    for source in football[:8]:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    content += "## 🏸 OTHER SPORTS\n\n"
    for source in other_sports[:14]:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    content += "## 🏆 SPORTS BUSINESS\n\n"
    for source in business[:5]:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    # Validate fallback output
    import re
    markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    logger.info(f"🔄 Fallback generated {len(markdown_links)} links")
    
    return content


    
def remove_duplicate_news_sports(news_list: List[Dict]) -> List[Dict]:
    """Enhanced duplicate removal specifically for sports news"""
    unique_news = []
    seen_titles = set()
    
    for news in news_list:
        title = news['title'].lower()
        
        # Create a normalized title key for comparison
        title_key = re.sub(r'[^\w\s]', '', title)
        title_key = ' '.join(title_key.split()[:8])  # First 8 words
        
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_news.append(news)
        else:
            logger.info(f"🔍 Removed duplicate: {news['title'][:60]}...")
    
    logger.info(f"🔄 Sports Deduplication: {len(news_list)} → {len(unique_news)} unique news")
    return unique_news



def create_sports_newsletter(topic: str, content: str, sources: List[Dict], publish_date: str):
    """Create sports-specific newsletter with custom styling"""
    def validate_and_clean_url(url: str) -> str:
        if not url:
            return "#"
        if 'news.google.com/rss/articles/' in url:
            try:
                article_match = re.search(r'/articles/([^?]+)', url)
                if article_match:
                    article_id = article_match.group(1)
                    return f"https://news.google.com/articles/{article_id}?hl=en-IN&gl=IN&ceid=IN:en"
            except Exception as e:
                logger.warning(f"URL cleanup failed for {url}: {e}")
        return url

    # Clean content
    content = re.sub(r'\*{1,}', '', content)
    content = re.sub(r'\d+\.\s*\n?', '', content)
    content = re.sub(r'^\s*[\-\*]\s*', '', content, flags=re.MULTILINE)
   
    def create_news_link(match):
        text = match.group(1)
        url = match.group(2)
        clean_url = validate_and_clean_url(url)
        clean_text = re.sub(r'[\[\]\(\)]', '', text).strip()
        
        # Sports-specific icons
        title_lower = clean_text.lower()
        if any(word in title_lower for word in ['cricket', 'ipl', 'bcci']):
            icon = "🏏"
        elif any(word in title_lower for word in ['football', 'soccer', 'isl']):
            icon = "⚽"
        elif any(word in title_lower for word in ['hockey', 'badminton', 'tennis']):
            icon = "🏸"
        else:
            icon = "🏆"
        
        return f'''
        <div class="sports-card">
            <div class="sports-content-inner">
                <a href="{clean_url}" class="sports-headline" target="_blank" rel="noopener noreferrer">
                    {icon} {clean_text}
                </a>
                <div class="sports-source">
                    <a href="{clean_url}" target="_blank" rel="noopener noreferrer">
                        📍 Read Full Story →
                    </a>
                </div>
            </div>
        </div>'''

    # Convert markdown links
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', create_news_link, content)
    
    # Handle section headers for sports
    content = re.sub(r'🏏 CRICKET UPDATES', '<div class="section-header cricket-header">🏏 CRICKET UPDATES</div>', content)
    content = re.sub(r'⚽ FOOTBALL NEWS', '<div class="section-header football-header">⚽ FOOTBALL NEWS</div>', content)
    content = re.sub(r'🏸 OTHER SPORTS', '<div class="section-header othersports-header">🏸 OTHER SPORTS</div>', content)
    content = re.sub(r'🏆 SPORTS BUSINESS', '<div class="section-header sportsbiz-header">🏆 SPORTS BUSINESS</div>', content)
    
    # Remove any remaining raw markdown links
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', '', content)
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = content.strip()

    words = len(re.sub('<[^>]*>', '', content).split())
    read_time = max(1, round(words / 220))
   
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sports Daily - Latest Updates</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
           
            body {{
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                line-height: 1.6;
                color: #1a1a1a;
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                min-height: 100vh;
                padding: 20px;
            }}
           
            .sports-container {{
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                overflow: hidden;
                box-shadow: 0 25px 50px rgba(0,0,0,0.15);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
           
            .sports-header {{
                background: linear-gradient(135deg, #ff6b6b, #ee5a24);
                color: white;
                padding: 40px 30px;
                text-align: center;
                position: relative;
                overflow: hidden;
            }}
           
            .sports-header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            }}
           
            .sports-header h1 {{
                font-size: 36px;
                font-weight: 800;
                margin-bottom: 15px;
                position: relative;
                z-index: 2;
            }}
           
            .sports-header p {{
                font-size: 16px;
                opacity: 0.9;
                font-weight: 600;
                position: relative;
                z-index: 2;
            }}
           
            .sports-content {{
                padding-left: 5px;
            }}
           
            .section-header {{
                font-size: 22px;
                font-weight: 800;
                margin: 35px 0 25px;
                padding: 8px 30px;
                border-radius: 16px;
                display: flex;
                align-items: center;
                gap: 15px;
                position: relative;
                overflow: hidden;
                color: white;
            }}
           
            .cricket-header {{
                background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            }}
           
            .football-header {{
                background: linear-gradient(135deg, #00d2d3, #54a0ff);
            }}
           
            .othersports-header {{
                background: linear-gradient(135deg, #feca57, #ff9ff3);
            }}
           
            .sportsbiz-header {{
                background: linear-gradient(135deg, #5f27cd, #341f97);
            }}
           
            .sports-card {{
                background: rgba(248, 249, 250, 0.8);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                margin: 20px 30px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                transition: all 0.4s ease;
                overflow: hidden;
                position: relative;
            }}
           
            .sports-card:hover {{
                transform: translateY(-8px);
                box-shadow: 0 20px 40px rgba(255, 107, 107, 0.2);
                border-color: #ff6b6b;
            }}
           
            .sports-content-inner {{
                padding: 0px;
            }}
           
            .sports-headline {{
                font-size: 18px;
                font-weight: 700;
                color: #2d3436;
                text-decoration: none;
                display: block;
                margin-bottom: 15px;
                line-height: 1.5;
                transition: color 0.3s ease;
            }}
           
            .sports-headline:hover {{
                color: #ff6b6b;
            }}
           
            .sports-source {{
                background: rgba(255, 107, 107, 0.08);
                padding: 12px 20px;
                border-radius: 12px;
                margin-top: 15px;
                border-left: 4px solid #ff6b6b;
            }}
           
            .sports-source a {{
                color: #636e72;
                font-size: 13px;
                text-decoration: none;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
                transition: color 0.2s ease;
            }}
           
            .sports-source a:hover {{
                color: #ff6b6b;
            }}
           
            @media (max-width: 768px) {{
                body {{
                    padding: 0px;
                }}
               
                .sports-header {{
                    padding: 30px 20px;
                }}
               
                .sports-header h1 {{
                    font-size: 28px;
                }}
               
                .section-header {{
                    font-size: 20px;
                    margin: 30px 0 20px;
                    padding: 8px 20px;
                }}
               
                .sports-card {{
                    margin: 15px 0px;
                }}
               
                .sports-headline {{
                    font-size: 16px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="sports-container">
            <div class="sports-header">
                <h1>🏆 SPORTS DAILY</h1>
                <p>{publish_date} • {len(sources)} Sources • Latest Sports Updates</p>
            </div>
           
            <div class="sports-content">
                {content}
            </div>
        </div>
       
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const links = document.querySelectorAll('a[href^="http"]');
                links.forEach(link => {{
                    link.setAttribute('target', '_blank');
                    link.setAttribute('rel', 'noopener noreferrer');
                }});
            }});
        </script>
    </body>
    </html>
    """
   
    return html, words, read_time


# ============================
# TECHNOLOGY NEWSLETTER FUNCTIONS
# ============================

# ============================
# TECHNOLOGY NEWSLETTER FUNCTIONS - ENHANCED
# ============================

async def generate_tech_newsletter_content(sources: List[Dict], topic: str) -> str:
    """Generate technology newsletter with comprehensive debugging - SAME AS SPORTS APPROACH"""
    try:
        logger.info("🔧 STARTING TECHNOLOGY DEBUGGING")
        
        model = get_gemini_model("gemini-2.0-flash-lite")
        
        # Step 1: Check initial sources
        logger.info(f"📥 INITIAL TECH SOURCES: {len(sources)}")
        logger.info("📋 SAMPLE INITIAL TECH SOURCES (first 5):")
        for i, source in enumerate(sources[:5]):
            logger.info(f"  {i+1}. {source['title'][:80]}...")
        
        unique_sources = remove_duplicate_news_tech(sources)
        logger.info(f"🔄 AFTER BASIC DEDUPLICATION: {len(sources)} → {len(unique_sources)}")
        
        target_count = 35
        
        # Step 2: Validate URLs and check specific tech stories
        validated_sources = []
        specific_stories_to_check = ["AI", "machine learning", "startup", "funding", "iPhone", "Google", "Microsoft"]
        
        logger.info("🔍 CHECKING SPECIFIC TECH STORIES IN VALIDATED SOURCES:")
        for source in unique_sources:
            validated_source = source.copy()
            validated_source['url'] = validate_and_clean_url_standalone(source['url'])
            validated_sources.append(validated_source)
            
            # Check for specific tech stories
            title_lower = source['title'].lower()
            for story in specific_stories_to_check:
                if story.lower() in title_lower:
                    logger.info(f"   ✅ FOUND '{story}': {source['title'][:60]}...")
        
        sources_for_llm = validated_sources
        
        # Step 3: Detailed analysis of sources being sent to LLM
        logger.info("📊 DETAILED TECH SOURCE ANALYSIS FOR LLM:")
        logger.info(f"📤 SENDING {len(sources_for_llm)} TECH SOURCES TO LLM")
        
        # Count by category
        ai_ml_count = sum(1 for s in sources_for_llm if any(word in s['title'].lower() for word in ['ai', 'artificial intelligence', 'machine learning', 'neural', 'deep learning']))
        software_count = sum(1 for s in sources_for_llm if any(word in s['title'].lower() for word in ['software', 'developer', 'programming', 'code', 'app', 'web']))
        gadgets_count = sum(1 for s in sources_for_llm if any(word in s['title'].lower() for word in ['gadget', 'hardware', 'phone', 'smartphone', 'device', 'laptop']))
        startups_count = sum(1 for s in sources_for_llm if any(word in s['title'].lower() for word in ['startup', 'funding', 'venture', 'unicorn', 'investment']))
        other_count = len(sources_for_llm) - ai_ml_count - software_count - gadgets_count - startups_count
        
        logger.info(f"🤖 AI/ML sources: {ai_ml_count}")
        logger.info(f"💻 Software sources: {software_count}")
        logger.info(f"📱 Gadgets sources: {gadgets_count}")
        logger.info(f"🚀 Startups sources: {startups_count}")
        logger.info(f"🔧 Other tech sources: {other_count}")
        
        # Step 4: Create URL mapping with verification
        url_map = {}
        logger.info("🔗 CREATING TECH URL MAPPING:")
        for i, source in enumerate(sources_for_llm):
            short_id = f"URL_{i+1}"
            url_map[short_id] = source['url']
            if i < 5:  # Log first 5 mappings
                logger.info(f"   {short_id} → {source['title'][:50]}...")
        
        # Step 5: Prepare news text for LLM
        news_text = "\n".join([
            f"{i+1}. {news['title']} | URL: URL_{i+1}"
            for i, news in enumerate(sources_for_llm)
        ])
        
        # Step 6: Enhanced prompt with clear instructions for technology
        prompt = f"""
        CRITICAL: You are a technology editor selecting the TOP 35 most important and DIVERSE technology stories from {len(validated_sources)} options.

        MUST SELECT EXACTLY 35 UNIQUE STORIES - NO DUPLICATES, NO SIMILAR STORIES.

        DUPLICATE REMOVAL RULES:
        - SAME PRODUCT LAUNCH: If multiple sources cover the same product (e.g., "iPhone 16 launch"), pick ONLY THE BEST ONE
        - SAME FUNDING NEWS: If multiple sources report the same startup funding, pick ONLY THE MOST COMPREHENSIVE
        - SAME TECH ANNOUNCEMENT: Don't include multiple routine updates about the same tech release

        PRIORITY ORDER FOR SELECTION:
        1. BREAKING NEWS: Major tech announcements, product launches, big funding rounds
        2. INDIAN TECH: Stories about Indian tech companies/startups get priority
        3. DIVERSITY: Ensure good mix across different tech domains
        4. AUTHORITATIVE SOURCES: Prefer established tech media over generic news

        SPECIFIC DUPLICATES TO WATCH FOR:
        - Multiple "iPhone 16" launch stories → PICK ONLY 1-2 BEST
        - Multiple "AI startup funding" updates → PICK ONLY THE KEY STORIES
        - Multiple "Google AI" announcements → PICK THE MAIN RELEASE
        - Multiple "Microsoft Copilot" stories → PICK THE MOST SIGNIFICANT

        SECTION DISTRIBUTION (35 TOTAL):
        ## 🤖 AI & MACHINE LEARNING (10-12 items)
        - Focus on major AI breakthroughs, research, important releases
        - Avoid multiple similar AI model announcements

        ## 💻 SOFTWARE DEVELOPMENT (8-10 items)  
        - Include programming languages, frameworks, tools, platforms
        - Mix of technical updates and industry news

        ## 📱 GADGETS & HARDWARE (8-10 items)
        - Smartphones, laptops, wearables, consumer electronics
        - Both launches and reviews/analysis

        ## 🚀 STARTUPS & INNOVATION (5-7 items)
        - Funding rounds, startup launches, tech innovations
        - Major tech business developments

        FORMATTING:
        [Complete Title](URL_ID)

        AVAILABLE SOURCES:
        {news_text}

        IMPORTANT: You MUST output EXACTLY 35 items total across all 4 sections. 
        Be ruthless in removing duplicates and similar stories.
        If you find multiple stories about the same specific tech product/announcement, pick THE BEST ONE and discard the rest.
        """

        logger.info(f"📝 TECH PROMPT LENGTH: {len(prompt)} characters")
        logger.info("🔍 SENDING TECH REQUEST TO LLM...")
        
        # Step 7: LLM call with timing
        import time
        start_time = time.time()
        response = model.generate_content(prompt)
        llm_time = time.time() - start_time
        logger.info(f"⏱️ TECH LLM RESPONSE TIME: {llm_time:.2f}s")
        
        content = response.text.strip()
        
        # Step 8: Analyze LLM response
        logger.info("📄 TECH LLM RAW RESPONSE ANALYSIS:")
        logger.info(f"📏 Response length: {len(content)} characters")
        logger.info(f"🔍 First 300 chars: {content[:300]}...")
        
        # Count sections in response
        sections_found = content.count('## 🤖') + content.count('## 💻') + content.count('## 📱') + content.count('## 🚀')
        logger.info(f"📑 Tech sections found in response: {sections_found}/4")
        
        # Step 9: URL restoration with detailed tracking
        logger.info("🔄 STARTING TECH URL RESTORATION...")
        original_content = content
        restoration_log = []
        
        for short_id, full_url in url_map.items():
            if f"]({short_id})" in content:
                content = content.replace(f"]({short_id})", f"]({full_url})")
                # Find which title this URL corresponds to
                source_idx = int(short_id.split('_')[1]) - 1
                if source_idx < len(sources_for_llm):
                    title = sources_for_llm[source_idx]['title'][:40]
                    restoration_log.append(f"{short_id} → {title}...")
        
        logger.info(f"✅ TECH URL RESTORATION COMPLETE: {len(restoration_log)} URLs replaced")
        if restoration_log:
            logger.info("📋 TECH RESTORED URLS (first 10):")
            for log_entry in restoration_log[:10]:
                logger.info(f"   {log_entry}")
        
        remaining_short_ids = content.count('URL_')
        logger.info(f"🔍 Remaining short IDs after restoration: {remaining_short_ids}")
        
        # Step 10: Final content analysis
        logger.info("📊 TECH FINAL CONTENT ANALYSIS:")
        import re
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        link_count = len(markdown_links)
        
        logger.info(f"📎 Total tech markdown links found: {link_count}")
        logger.info(f"🎯 Target count: {target_count}")
        
        # Analyze link distribution
        ai_links = sum(1 for title, _ in markdown_links if any(word in title.lower() for word in ['ai', 'artificial intelligence', 'machine learning']))
        software_links = sum(1 for title, _ in markdown_links if any(word in title.lower() for word in ['software', 'developer', 'programming']))
        gadgets_links = sum(1 for title, _ in markdown_links if any(word in title.lower() for word in ['gadget', 'phone', 'hardware', 'device']))
        startups_links = sum(1 for title, _ in markdown_links if any(word in title.lower() for word in ['startup', 'funding', 'venture']))
        
        logger.info(f"🤖 AI links: {ai_links}")
        logger.info(f"💻 Software links: {software_links}") 
        logger.info(f"📱 Gadgets links: {gadgets_links}")
        logger.info(f"🚀 Startups links: {startups_links}")
        
        # Check for specific stories in final output
        logger.info("🔍 CHECKING SPECIFIC TECH STORIES IN FINAL OUTPUT:")
        found_stories = {}
        for story in specific_stories_to_check:
            count = sum(1 for title, _ in markdown_links if story.lower() in title.lower())
            found_stories[story] = count
            logger.info(f"   '{story}': {count} occurrences")
        
        # Check URL integrity
        proper_urls_count = sum(1 for _, url in markdown_links if not url.startswith('URL_'))
        logger.info(f"🔗 Proper tech URLs: {proper_urls_count}/{link_count}")
        
        # Step 11: Log sample of final output
        logger.info("📋 TECH FINAL OUTPUT SAMPLE (first 10 items):")
        for i, (title, url) in enumerate(markdown_links[:10]):
            logger.info(f"   {i+1}. {title[:50]}...")
            logger.info(f"      URL: {url[:80]}...")
        
        content = content.strip()
        
        # Step 12: Final validation - SAME AS SPORTS
        variance_allowed = 4  # Allow ±4 variance

        if abs(link_count - target_count) <= variance_allowed and proper_urls_count == link_count:
            logger.info(f"🎯 TECH SUCCESS: Newsletter generated with {link_count} items (target: {target_count}±{variance_allowed})")
            return content
        else:
            logger.warning(f"⚠️  Using tech fallback: Got {link_count}/{target_count} links, {proper_urls_count} proper URLs")
            return force_exact_tech_count(validated_sources, topic, target_count)
        
    except Exception as e:
        logger.error(f"❌ TECH NEWSLETTER FAILED: {e}")
        import traceback
        logger.error(f"🔍 TECH FULL ERROR TRACEBACK:\n{traceback.format_exc()}")
        return force_exact_tech_count(sources[:target_count], topic, target_count)

def remove_duplicate_news_tech(news_list: List[Dict]) -> List[Dict]:
    """Enhanced duplicate removal specifically for technology news"""
    unique_news = []
    seen_titles = set()
    
    for news in news_list:
        title = news['title'].lower()
        
        # Create a normalized title key for comparison
        title_key = re.sub(r'[^\w\s]', '', title)
        title_key = ' '.join(title_key.split()[:8])  # First 8 words
        
        # Tech-specific duplicate patterns
        duplicate_patterns = [
            title_key,
            title_key.replace('iphone', 'apple iphone'),
            title_key.replace('google', 'alphabet'),
            title_key.replace('ai', 'artificial intelligence'),
            title_key.replace('funding', 'raise'),
            title_key.replace('launch', 'release'),
        ]
        
        is_duplicate = any(pattern in seen_titles for pattern in duplicate_patterns)
        
        if not is_duplicate:
            seen_titles.add(title_key)
            unique_news.append(news)
        else:
            logger.info(f"🔍 Removed tech duplicate: {news['title'][:60]}...")
    
    logger.info(f"🔄 Tech Deduplication: {len(news_list)} → {len(unique_news)} unique news")
    return unique_news

def force_exact_tech_count(sources: List[Dict], topic: str, target_count: int) -> str:
    """Fallback method for technology with debugging"""
    logger.info("🔄 USING TECH FALLBACK SELECTION METHOD")
    logger.info(f"📥 Tech fallback input: {len(sources)} sources")
    
    # Categorize sources for fallback - TECH SPECIFIC
    ai_ml = []
    software = []
    gadgets = []
    startups = []
    
    for source in sources:
        title_lower = source['title'].lower()
        if any(word in title_lower for word in ['ai', 'artificial intelligence', 'machine learning', 'neural', 'deep learning']):
            ai_ml.append(source)
        elif any(word in title_lower for word in ['software', 'developer', 'programming', 'code', 'app', 'web']):
            software.append(source)
        elif any(word in title_lower for word in ['gadget', 'hardware', 'phone', 'smartphone', 'device', 'laptop']):
            gadgets.append(source)
        elif any(word in title_lower for word in ['startup', 'funding', 'venture', 'unicorn', 'investment']):
            startups.append(source)
        else:
            # Default to software if no clear category
            software.append(source)
    
    logger.info(f"🤖 Fallback AI/ML: {len(ai_ml)}")
    logger.info(f"💻 Fallback software: {len(software)}")
    logger.info(f"📱 Fallback gadgets: {len(gadgets)}")
    logger.info(f"🚀 Fallback startups: {len(startups)}")
    
    # Build content with tech-specific distribution
    content = "## 🤖 AI & MACHINE LEARNING\n\n"
    for source in ai_ml[:10]:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    content += "## 💻 SOFTWARE DEVELOPMENT\n\n"
    for source in software[:10]:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    content += "## 📱 GADGETS & HARDWARE\n\n"
    for source in gadgets[:8]:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    content += "## 🚀 STARTUPS & INNOVATION\n\n"
    for source in startups[:7]:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    # Validate fallback output
    import re
    markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    logger.info(f"🔄 Tech fallback generated {len(markdown_links)} links")
    
    return content

def create_tech_newsletter(topic: str, content: str, sources: List[Dict], publish_date: str):
    """Create technology-specific newsletter with custom styling"""
    def validate_and_clean_url(url: str) -> str:
        if not url:
            return "#"
        if 'news.google.com/rss/articles/' in url:
            try:
                article_match = re.search(r'/articles/([^?]+)', url)
                if article_match:
                    article_id = article_match.group(1)
                    return f"https://news.google.com/articles/{article_id}?hl=en-IN&gl=IN&ceid=IN:en"
            except Exception as e:
                logger.warning(f"URL cleanup failed for {url}: {e}")
        return url

    # Clean content
    content = re.sub(r'\*{1,}', '', content)
    content = re.sub(r'\d+\.\s*\n?', '', content)
    content = re.sub(r'^\s*[\-\*]\s*', '', content, flags=re.MULTILINE)
   
    def create_news_link(match):
        text = match.group(1)
        url = match.group(2)
        clean_url = validate_and_clean_url(url)
        clean_text = re.sub(r'[\[\]\(\)]', '', text).strip()
        
        # Tech-specific icons
        title_lower = clean_text.lower()
        if any(word in title_lower for word in ['ai', 'artificial intelligence', 'machine learning']):
            icon = "🤖"
        elif any(word in title_lower for word in ['software', 'developer', 'programming']):
            icon = "💻"
        elif any(word in title_lower for word in ['gadget', 'phone', 'hardware']):
            icon = "📱"
        else:
            icon = "🚀"
        
        return f'''
        <div class="tech-card">
            <div class="tech-content-inner">
                <a href="{clean_url}" class="tech-headline" target="_blank" rel="noopener noreferrer">
                    {icon} {clean_text}
                </a>
                <div class="tech-source">
                    <a href="{clean_url}" target="_blank" rel="noopener noreferrer">
                        📍 Read Full Story →
                    </a>
                </div>
            </div>
        </div>'''

    # Convert markdown links
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', create_news_link, content)
    
    # Handle section headers for tech
    content = re.sub(r'🤖 AI & MACHINE LEARNING', '<div class="section-header ai-header">🤖 AI & MACHINE LEARNING</div>', content)
    content = re.sub(r'💻 SOFTWARE DEVELOPMENT', '<div class="section-header software-header">💻 SOFTWARE DEVELOPMENT</div>', content)
    content = re.sub(r'📱 GADGETS & HARDWARE', '<div class="section-header gadgets-header">📱 GADGETS & HARDWARE</div>', content)
    content = re.sub(r'🚀 STARTUPS & INNOVATION', '<div class="section-header startups-header">🚀 STARTUPS & INNOVATION</div>', content)
    
    # Remove any remaining raw markdown links
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', '', content)
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = content.strip()

    words = len(re.sub('<[^>]*>', '', content).split())
    read_time = max(1, round(words / 220))
   
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tech Daily - Latest Updates</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
           
            body {{
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                line-height: 1.6;
                color: #1a1a1a;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
           
            .tech-container {{
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                overflow: hidden;
                box-shadow: 0 25px 50px rgba(0,0,0,0.15);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
           
            .tech-header {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 40px 30px;
                text-align: center;
                position: relative;
                overflow: hidden;
            }}
           
            .tech-header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            }}
           
            .tech-header h1 {{
                font-size: 36px;
                font-weight: 800;
                margin-bottom: 15px;
                position: relative;
                z-index: 2;
            }}
           
            .tech-header p {{
                font-size: 16px;
                opacity: 0.9;
                font-weight: 600;
                position: relative;
                z-index: 2;
            }}
           
            .tech-content {{
                padding-left: 5px;
            }}
           
            .section-header {{
                font-size: 22px;
                font-weight: 800;
                margin: 35px 0 25px;
                padding: 8px 30px;
                border-radius: 16px;
                display: flex;
                align-items: center;
                gap: 15px;
                position: relative;
                overflow: hidden;
                color: white;
            }}
           
            .ai-header {{
                background: linear-gradient(135deg, #667eea, #764ba2);
            }}
           
            .software-header {{
                background: linear-gradient(135deg, #00d2d3, #54a0ff);
            }}
           
            .gadgets-header {{
                background: linear-gradient(135deg, #feca57, #ff9ff3);
            }}
           
            .startups-header {{
                background: linear-gradient(135deg, #5f27cd, #341f97);
            }}
           
            .tech-card {{
                background: rgba(248, 249, 250, 0.8);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                margin: 20px 30px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                transition: all 0.4s ease;
                overflow: hidden;
                position: relative;
            }}
           
            .tech-card:hover {{
                transform: translateY(-8px);
                box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
                border-color: #667eea;
            }}
           
            .tech-content-inner {{
                padding: 0px;
            }}
           
            .tech-headline {{
                font-size: 18px;
                font-weight: 700;
                color: #2d3436;
                text-decoration: none;
                display: block;
                margin-bottom: 15px;
                line-height: 1.5;
                transition: color 0.3s ease;
            }}
           
            .tech-headline:hover {{
                color: #667eea;
            }}
           
            .tech-source {{
                background: rgba(102, 126, 234, 0.08);
                padding: 12px 20px;
                border-radius: 12px;
                margin-top: 15px;
                border-left: 4px solid #667eea;
            }}
           
            .tech-source a {{
                color: #636e72;
                font-size: 13px;
                text-decoration: none;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
                transition: color 0.2s ease;
            }}
           
            .tech-source a:hover {{
                color: #667eea;
            }}
           
            @media (max-width: 768px) {{
                body {{
                    padding: 0px;
                }}
               
                .tech-header {{
                    padding: 30px 20px;
                }}
               
                .tech-header h1 {{
                    font-size: 28px;
                }}
               
                .section-header {{
                    font-size: 20px;
                    margin: 30px 0 20px;
                    padding: 8px 20px;
                }}
               
                .tech-card {{
                    margin: 15px 0px;
                }}
               
                .tech-headline {{
                    font-size: 16px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="tech-container">
            <div class="tech-header">
                <h1>🤖 TECH DAILY</h1>
                <p>{publish_date} • {len(sources)} Sources • Latest Technology News</p>
            </div>
           
            <div class="tech-content">
                {content}
            </div>
        </div>
       
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const links = document.querySelectorAll('a[href^="http"]');
                links.forEach(link => {{
                    link.setAttribute('target', '_blank');
                    link.setAttribute('rel', 'noopener noreferrer');
                }});
            }});
        </script>
    </body>
    </html>
    """
   
    return html, words, read_time

# ============================
# POWER PROJECTS NEWSLETTER FUNCTIONS
# ============================

async def generate_power_newsletter_content(sources: List[Dict], topic: str) -> str:
    """Generate power projects-specific newsletter content"""
    try:
        model = get_gemini_model("gemini-2.0-flash-lite")
        
        unique_sources = remove_duplicate_news(sources)
        logger.info(f"🔄 Unique power sources available: {len(unique_sources)}")
        
        target_count = 35
        
        validated_sources = []
        for source in unique_sources:
            validated_source = source.copy()
            validated_source['url'] = validate_and_clean_url_standalone(source['url'])
            validated_sources.append(validated_source)
        
        sources_for_llm = validated_sources[:target_count]
        
        url_map = create_url_mapping(sources_for_llm)
        
        news_text = "\n".join([
            f"{i+1}. {news['title'][:80]} | URL: URL_{i+1}"
            for i, news in enumerate(sources_for_llm)
        ])
        
        prompt = f"""
        Create power projects newsletter with EXACTLY 35 items. Use ALL sources below.

        CRITICAL FORMATTING RULES:
        - DO NOT TRUNCATE TITLES - use full titles as provided
        - Every link must have proper format: [Complete Title](URL_ID)

        FORMAT EACH ITEM AS: [Title](URL_ID)

        SECTIONS (fill sequentially):
        ## ⚡ POWER PROJECTS UPDATE
        ## 🌞 RENEWABLE ENERGY
        ## 🏗️ INFRASTRUCTURE & GRID
        ## 💼 BUSINESS & POLICY

        RULES:
        - Use ALL 35 items below
        - Fill sections in order: Projects, Renewable, Infrastructure, Business
        - MUST reach 35 total items
        - Use EXACT URL IDs as provided
        - DO NOT TRUNCATE OR SHORTEN TITLES

        SOURCES:
        {news_text}

        OUTPUT: Just the 4 section headers and 35 markdown links. No explanations.
        """

        response = model.generate_content(prompt, stream=True)
        
        content_chunks = []
        for chunk in response:
            content_chunks.append(chunk.text)
        
        content = "".join(content_chunks)
        content = restore_full_urls(content, url_map)
        
        content = content.strip()
        
        import re
        markdown_patterns = re.findall(r'(\[.*?\]\(.*?\))', content)
        link_count = len(markdown_patterns)
        
        if link_count != target_count:
            logger.warning(f"🔄 Power LLM got {link_count}, using template")
            return force_exact_power_count(sources_for_llm, topic, target_count)
        
        logger.info(f"🎯 POWER SUCCESS: LLM included {link_count} sources")
        return content
        
    except Exception as e:
        logger.error(f"Power LLM newsletter failed: {e}")
        return force_exact_power_count(unique_sources[:target_count], topic, target_count)

def force_exact_power_count(sources: List[Dict], topic: str, target_count: int) -> str:
    """Bulletproof method that guarantees exact count for power projects"""
    projects = []
    renewable = []
    infrastructure = []
    business = []
    
    for source in sources:
        title_lower = source['title'].lower()
        if any(word in title_lower for word in ['project', 'commission', 'launch', 'install']):
            projects.append(source)
        elif any(word in title_lower for word in ['solar', 'wind', 'renewable', 'green energy']):
            renewable.append(source)
        elif any(word in title_lower for word in ['grid', 'transmission', 'infrastructure', 'substation']):
            infrastructure.append(source)
        else:
            business.append(source)
    
    # Ensure we have exactly target_count
    total = len(projects) + len(renewable) + len(infrastructure) + len(business)
    if total > target_count:
        projects_target = max(1, int(target_count * 0.4))
        renewable_target = max(1, int(target_count * 0.3))
        infrastructure_target = max(1, int(target_count * 0.2))
        business_target = max(1, target_count - projects_target - renewable_target - infrastructure_target)
        
        projects = projects[:projects_target]
        renewable = renewable[:renewable_target]
        infrastructure = infrastructure[:infrastructure_target]
        business = business[:business_target]
    
    content = "## ⚡ POWER PROJECTS UPDATE\n\n"
    for source in projects:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    content += "## 🌞 RENEWABLE ENERGY\n\n"
    for source in renewable:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    content += "## 🏗️ INFRASTRUCTURE & GRID\n\n"
    for source in infrastructure:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    content += "## 💼 BUSINESS & POLICY\n\n"
    for source in business:
        content += f"[{source['title']}]({source['url']})\n\n"
    
    return content

def create_power_newsletter(topic: str, content: str, sources: List[Dict], publish_date: str):
    """Create power projects-specific newsletter with custom styling"""
    def validate_and_clean_url(url: str) -> str:
        if not url:
            return "#"
        if 'news.google.com/rss/articles/' in url:
            try:
                article_match = re.search(r'/articles/([^?]+)', url)
                if article_match:
                    article_id = article_match.group(1)
                    return f"https://news.google.com/articles/{article_id}?hl=en-IN&gl=IN&ceid=IN:en"
            except Exception as e:
                logger.warning(f"URL cleanup failed for {url}: {e}")
        return url

    # Clean content
    content = re.sub(r'\*{1,}', '', content)
    content = re.sub(r'\d+\.\s*\n?', '', content)
    content = re.sub(r'^\s*[\-\*]\s*', '', content, flags=re.MULTILINE)
   
    def create_news_link(match):
        text = match.group(1)
        url = match.group(2)
        clean_url = validate_and_clean_url(url)
        clean_text = re.sub(r'[\[\]\(\)]', '', text).strip()
        
        # Power-specific icons
        title_lower = clean_text.lower()
        if any(word in title_lower for word in ['solar', 'wind', 'renewable']):
            icon = "🌞"
        elif any(word in title_lower for word in ['grid', 'transmission', 'infrastructure']):
            icon = "🏗️"
        elif any(word in title_lower for word in ['business', 'policy', 'deal']):
            icon = "💼"
        else:
            icon = "⚡"
        
        return f'''
        <div class="power-card">
            <div class="power-content-inner">
                <a href="{clean_url}" class="power-headline" target="_blank" rel="noopener noreferrer">
                    {icon} {clean_text}
                </a>
                <div class="power-source">
                    <a href="{clean_url}" target="_blank" rel="noopener noreferrer">
                        📍 Read Full Story →
                    </a>
                </div>
            </div>
        </div>'''

    # Convert markdown links
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', create_news_link, content)
    
    # Handle section headers for power
    content = re.sub(r'⚡ POWER PROJECTS UPDATE', '<div class="section-header projects-header">⚡ POWER PROJECTS UPDATE</div>', content)
    content = re.sub(r'🌞 RENEWABLE ENERGY', '<div class="section-header renewable-header">🌞 RENEWABLE ENERGY</div>', content)
    content = re.sub(r'🏗️ INFRASTRUCTURE & GRID', '<div class="section-header infrastructure-header">🏗️ INFRASTRUCTURE & GRID</div>', content)
    content = re.sub(r'💼 BUSINESS & POLICY', '<div class="section-header business-header">💼 BUSINESS & POLICY</div>', content)
    
    # Remove any remaining raw markdown links
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', '', content)
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = content.strip()

    words = len(re.sub('<[^>]*>', '', content).split())
    read_time = max(1, round(words / 220))
   
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Power Projects Daily - Latest Updates</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
           
            body {{
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                line-height: 1.6;
                color: #1a1a1a;
                background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
                min-height: 100vh;
                padding: 20px;
            }}
           
            .power-container {{
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                overflow: hidden;
                box-shadow: 0 25px 50px rgba(0,0,0,0.15);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
           
            .power-header {{
                background: linear-gradient(135deg, #0f4c75, #3282b8);
                color: white;
                padding: 40px 30px;
                text-align: center;
                position: relative;
                overflow: hidden;
            }}
           
            .power-header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            }}
           
            .power-header h1 {{
                font-size: 36px;
                font-weight: 800;
                margin-bottom: 15px;
                position: relative;
                z-index: 2;
            }}
           
            .power-header p {{
                font-size: 16px;
                opacity: 0.9;
                font-weight: 600;
                position: relative;
                z-index: 2;
            }}
           
            .power-content {{
                padding-left: 5px;
            }}
           
            .section-header {{
                font-size: 22px;
                font-weight: 800;
                margin: 35px 0 25px;
                padding: 8px 30px;
                border-radius: 16px;
                display: flex;
                align-items: center;
                gap: 15px;
                position: relative;
                overflow: hidden;
                color: white;
            }}
           
            .projects-header {{
                background: linear-gradient(135deg, #0f4c75, #3282b8);
            }}
           
            .renewable-header {{
                background: linear-gradient(135deg, #00d2d3, #54a0ff);
            }}
           
            .infrastructure-header {{
                background: linear-gradient(135deg, #feca57, #ff9ff3);
            }}
           
            .business-header {{
                background: linear-gradient(135deg, #5f27cd, #341f97);
            }}
           
            .power-card {{
                background: rgba(248, 249, 250, 0.8);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                margin: 20px 30px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                transition: all 0.4s ease;
                overflow: hidden;
                position: relative;
            }}
           
            .power-card:hover {{
                transform: translateY(-8px);
                box-shadow: 0 20px 40px rgba(15, 76, 117, 0.2);
                border-color: #0f4c75;
            }}
           
            .power-content-inner {{
                padding: 0px;
            }}
           
            .power-headline {{
                font-size: 18px;
                font-weight: 700;
                color: #2d3436;
                text-decoration: none;
                display: block;
                margin-bottom: 15px;
                line-height: 1.5;
                transition: color 0.3s ease;
            }}
           
            .power-headline:hover {{
                color: #0f4c75;
            }}
           
            .power-source {{
                background: rgba(15, 76, 117, 0.08);
                padding: 12px 20px;
                border-radius: 12px;
                margin-top: 15px;
                border-left: 4px solid #0f4c75;
            }}
           
            .power-source a {{
                color: #636e72;
                font-size: 13px;
                text-decoration: none;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
                transition: color 0.2s ease;
            }}
           
            .power-source a:hover {{
                color: #0f4c75;
            }}
           
            @media (max-width: 768px) {{
                body {{
                    padding: 0px;
                }}
               
                .power-header {{
                    padding: 30px 20px;
                }}
               
                .power-header h1 {{
                    font-size: 28px;
                }}
               
                .section-header {{
                    font-size: 20px;
                    margin: 30px 0 20px;
                    padding: 8px 20px;
                }}
               
                .power-card {{
                    margin: 15px 0px;
                }}
               
                .power-headline {{
                    font-size: 16px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="power-container">
            <div class="power-header">
                <h1>⚡ POWER PROJECTS DAILY</h1>
                <p>{publish_date} • {len(sources)} Sources • Latest Energy Updates</p>
            </div>
           
            <div class="power-content">
                {content}
            </div>
        </div>
       
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const links = document.querySelectorAll('a[href^="http"]');
                links.forEach(link => {{
                    link.setAttribute('target', '_blank');
                    link.setAttribute('rel', 'noopener noreferrer');
                }});
            }});
        </script>
    </body>
    </html>
    """
   
    return html, words, read_time

# ============================
# JOBS NEWSLETTER FUNCTIONS (KEEP EXISTING)
# ============================
def validate_and_clean_url_standalone_job(url: str) -> str:
    """Standalone version of URL validation for use outside classes"""
    if not url:
        return "#"
    
    # Convert Google News RSS URLs to proper article URLs
    if 'news.google.com/rss/articles/' in url or ('news.google.com' in url and '/articles/' in url):
        try:
            # Extract article ID from RSS URL
            article_match = re.search(r'/articles/([^?]+)', url)
            if article_match:
                article_id = article_match.group(1)
                # Clean the article ID (remove any extra parameters)
                clean_id = article_id.split('?')[0] if '?' in article_id else article_id
                cleaned_url = f"https://news.google.com/articles/{clean_id}?hl=en-IN&gl=IN&ceid=IN:en"
                # logger.info(f"✅ Converted RSS URL to: {cleaned_url[:80]}...")
                return cleaned_url
        except Exception as e:
            logger.warning(f"URL conversion failed for {url}: {e}")
    
    # Return original URL if no conversion needed
    return url

def generate_manual_job_newsletter(sources: List[Dict]) -> str:
    """FIXED: Generate job newsletter with proper score handling"""
    
    # First, clean and validate all URLs
    for job in sources:
        job['url'] = validate_and_clean_url_standalone_job(job['url'])
        # Ensure all required fields exist
        if 'company_detected' not in job:
            job['company_detected'] = job.get('company', 'Unknown')
        if 'is_top_company' not in job:
            job['is_top_company'] = False
        if 'combined_score' not in job:
            job['combined_score'] = job.get('relevance_score', 0.5)
    
    # DEBUG: Log what we received with detailed scores
    logger.info("📊 GENERATING NEWSLETTER WITH SOURCES - DETAILED ANALYSIS:")
    categories_count = {}
    top_companies_count = {}
    score_distribution = {'0.0': 0, '0.1-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7+': 0}
    
    for i, job in enumerate(sources):
        cat = job.get('category', 'unknown')
        categories_count[cat] = categories_count.get(cat, 0) + 1
        
        company = job.get('company_detected', 'Unknown')
        if job.get('is_top_company', False):
            top_companies_count[company] = top_companies_count.get(company, 0) + 1
        
        # Analyze score
        score = job.get('combined_score', 0)
        if score == 0:
            score_distribution['0.0'] += 1
        elif score < 0.3:
            score_distribution['0.1-0.3'] += 1
        elif score < 0.5:
            score_distribution['0.3-0.5'] += 1
        elif score < 0.7:
            score_distribution['0.5-0.7'] += 1
        else:
            score_distribution['0.7+'] += 1
        
        # Log first 10 jobs for detailed analysis
        if i < 10:
            logger.info(f"  {i+1}. Cat: {cat}, Company: {company}, "
                       f"Top: {job.get('is_top_company', False)}, "
                       f"Score: {score:.2f}, "
                       f"Title: {job['title'][:40]}...")
    
    logger.info(f"📊 Input categories: {categories_count}")
    logger.info(f"🏆 Top companies in input: {top_companies_count}")
    logger.info(f"📈 Score distribution: {score_distribution}")
    
    # FIXED FILTERING: Handle 0.00 scores properly
    quality_jobs = []
    filtered_out_reasons = {'score': 0, 'resume': 0, 'no_category': 0}
    
    for job in sources:
        title = job['title'].lower()
        
        # 1. Skip obvious non-jobs
        if 'resume' in title or 'cv' in title:
            filtered_out_reasons['resume'] += 1
            continue
            
        # 2. FIXED: Handle 0.00 scores - they might be valid jobs!
        score = job.get('combined_score', 0)
        
        # If score is 0.00 but it's a top company or has proper category, keep it
        if score == 0:
            # Check if it's worth keeping
            is_worth_keeping = (
                job.get('is_top_company', False) or
                job.get('category') in ['electrical', 'civil', 'software'] or
                any(word in title for word in ['engineer', 'developer', 'manager'])
            )
            
            if not is_worth_keeping:
                filtered_out_reasons['score'] += 1
                continue
            else:
                # Assign a default score for display
                job['combined_score'] = 0.5 if job.get('is_top_company', False) else 0.4
                logger.debug(f"Fixed 0.00 score for: {job['title'][:50]}...")
        
        # 3. Skip if no proper category
        if job.get('category') not in ['electrical', 'civil', 'software']:
            # Try to infer category from title
            if any(word in title for word in ['electrical', 'power', 'transmission', 'solar', 'wind']):
                job['category'] = 'electrical'
            elif any(word in title for word in ['civil', 'construction', 'site', 'structural']):
                job['category'] = 'civil'
            elif any(word in title for word in ['software', 'developer', 'programmer', 'it engineer']):
                job['category'] = 'software'
            else:
                filtered_out_reasons['no_category'] += 1
                continue
        
        quality_jobs.append(job)
    
    logger.info(f"📊 After filtering: {len(sources)} → {len(quality_jobs)} quality jobs")
    logger.info(f"📉 Filtered out reasons: {filtered_out_reasons}")
    
    if len(quality_jobs) == 0:
        logger.warning("⚠️ No quality jobs found after filtering! Checking if we can use any jobs...")
        # Use all jobs but fix their scores
        for job in sources:
            if job.get('combined_score', 0) == 0:
                job['combined_score'] = 0.5 if job.get('is_top_company', False) else 0.4
            # Ensure category exists
            if job.get('category') not in ['electrical', 'civil', 'software']:
                title = job['title'].lower()
                if any(word in title for word in ['electrical', 'power', 'transmission']):
                    job['category'] = 'electrical'
                elif any(word in title for word in ['civil', 'construction', 'site']):
                    job['category'] = 'civil'
                elif any(word in title for word in ['software', 'developer', 'programmer']):
                    job['category'] = 'software'
                else:
                    job['category'] = 'other'
        quality_jobs = sources
        logger.info(f"⚠️ Using all {len(quality_jobs)} jobs with adjusted scores")
    
    # Group jobs by ACTUAL LLM categories
    electrical_jobs = []
    civil_jobs = []
    software_jobs = []
    
    for job in quality_jobs:
        category = job.get('category', 'other')
        if category == 'electrical':
            electrical_jobs.append(job)
        elif category == 'civil':
            civil_jobs.append(job)
        elif category == 'software':
            software_jobs.append(job)
    
    # Sort each category: TOP COMPANIES FIRST, then by score
    def sort_with_top_companies_first(jobs):
        return sorted(jobs, key=lambda x: (
            not x.get('is_top_company', False),  # True (top company) comes before False
            -x.get('combined_score', 0)  # Higher score first
        ))
    
    electrical_jobs = sort_with_top_companies_first(electrical_jobs)
    civil_jobs = sort_with_top_companies_first(civil_jobs)
    software_jobs = sort_with_top_companies_first(software_jobs)
    
    # Count top companies in each category
    top_electrical = sum(1 for j in electrical_jobs if j.get('is_top_company', False))
    top_civil = sum(1 for j in civil_jobs if j.get('is_top_company', False))
    top_software = sum(1 for j in software_jobs if j.get('is_top_company', False))
    
    logger.info(f"📊 Final counts: Electrical={len(electrical_jobs)} (Top: {top_electrical}), "
                f"Civil={len(civil_jobs)} (Top: {top_civil}), "
                f"Software={len(software_jobs)} (Top: {top_software})")
    
    # Log sample of each category
    logger.info("📋 Sample jobs by category:")
    logger.info("  Electrical (first 3):")
    for i, job in enumerate(electrical_jobs[:3]):
        logger.info(f"    {i+1}. {job['title'][:50]}... (Score: {job.get('combined_score', 0):.2f})")
    logger.info("  Civil (first 3):")
    for i, job in enumerate(civil_jobs[:3]):
        logger.info(f"    {i+1}. {job['title'][:50]}... (Score: {job.get('combined_score', 0):.2f})")
    logger.info("  Software (first 3):")
    for i, job in enumerate(software_jobs[:3]):
        logger.info(f"    {i+1}. {job['title'][:50]}... (Score: {job.get('combined_score', 0):.2f})")
    
    # Build newsletter content (same as before, but with fixed scores)
    content = ""
    
    # ====== TOP COMPANIES HEADER ======
    all_companies = {}
    for job in quality_jobs:
        company = job.get('company_detected', 'Unknown')
        if company not in ['Unknown', 'Not Found', 'Not Specified']:
            all_companies[company] = all_companies.get(company, 0) + 1
    
    if all_companies:
        sorted_companies = sorted(all_companies.items(), key=lambda x: x[1], reverse=True)
        display_companies = [company for company, count in sorted_companies[:8]]
        total_jobs_display = sum(count for company, count in sorted_companies[:8])
        
        # content += f'''
        # <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 15px 30px; border-radius: 12px; margin: 20px 30px; text-align: center;">
        #     <h3 style="color: white; margin: 0; font-size: 18px; font-weight: bold;">
        #         🏢 FEATURED COMPANIES ({total_jobs_display} Jobs)
        #     </h3>
        #     <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0; font-size: 14px;">
        #         {', '.join(display_companies)}{'...' if len(all_companies) > 8 else ''}
        #     </p>
        # </div>
        # '''
    

    if electrical_jobs:
        content += '<div class="section-header electrical-header">⚡ ELECTRICAL & POWER JOBS</div>\n'
        
        if top_electrical > 0:
            content += f'''
            <div style="color: #ff9800; font-size: 14px; margin: 10px 30px 5px 30px; font-weight: 600;">
                🏆 {top_electrical} top company jobs in this section
            </div>
            '''
        
        for job in electrical_jobs[:16]:
            # top_badge = " 🏆" if job.get('is_top_company', False) else ""
            # company_text = ""
            top_badge = ""
            company_text = ""
            
            company = job.get('company_detected', '')
            # if company and company not in ['Unknown', 'Not Found', 'Not Specified']:
            #     company_text = f" • {company}"
            
            content += f'''
            <div class="tech-card" style="{'border-left: 4px solid #ff9800; background: rgba(255, 152, 0, 0.05);' if job.get('is_top_company', False) else ''}">
                <div class="tech-content-inner">
                    <a href="{job['url']}" class="tech-headline" target="_blank">
                        ⚡ {safe_job_title(job["title"])}{top_badge}{company_text}
                    </a>
                    <div class="tech-source">
                        <a href="{job['url']}" target="_blank">
                            📍 {job['source']} → Apply Now
                        </a>
                        {'<span style="background: #ff9800; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 10px; font-weight: bold;">TOP COMPANY</span>' if job.get('is_top_company', False) else ''}
                    </div>
                </div>
            </div>\n'''
    
    # ====== CIVIL JOBS SECTION ======
    if civil_jobs:
        content += '<div class="section-header civil-header">🏗️ CIVIL & CONSTRUCTION JOBS</div>\n'
        
        if top_civil > 0:
            content += f'''
            <div style="color: #ff9800; font-size: 14px; margin: 10px 30px 5px 30px; font-weight: 600;">
                🏆 {top_civil} top company jobs in this section
            </div>
            '''
        
        for job in civil_jobs[:10]:
            # top_badge = " 🏆" if job.get('is_top_company', False) else ""
            # company_text = ""
            top_badge = ""
            company_text=""
            company = job.get('company_detected', '')
            # if company and company not in ['Unknown', 'Not Found', 'Not Specified']:
            #     company_text = f" • {company}"
            
            content += f'''
            <div class="tech-card" style="{'border-left: 4px solid #ff9800; background: rgba(255, 152, 0, 0.05);' if job.get('is_top_company', False) else ''}">
                <div class="tech-content-inner">
                    <a href="{job['url']}" class="tech-headline" target="_blank">
                        🏗️ {safe_job_title(job["title"])}{top_badge}{company_text}
                    </a>
                    <div class="tech-source">
                        <a href="{job['url']}" target="_blank">
                            📍 {job['source']} → Apply Now
                        </a>
                        {'<span style="background: #ff9800; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 10px; font-weight: bold;">TOP COMPANY</span>' if job.get('is_top_company', False) else ''}
                    </div>
                </div>
            </div>\n'''
    
    # ====== SOFTWARE JOBS SECTION ======
    if software_jobs:
        content += '<div class="section-header software-header">💻 SOFTWARE & IT JOBS</div>\n'
        
        if top_software > 0:
            content += f'''
            <div style="color: #ff9800; font-size: 14px; margin: 10px 30px 5px 30px; font-weight: 600;">
                🏆 {top_software} top company jobs in this section
            </div>
            '''
        
        for job in software_jobs[:9]:
            # top_badge = " 🏆" if job.get('is_top_company', False) else ""
            # company_text = ""
            top_badge = ""
            company_text=""
            
            company = job.get('company_detected', '')
            # if company and company not in ['Unknown', 'Not Found', 'Not Specified']:
            #     company_text = f" • {company}"
            
            content += f'''
            <div class="tech-card" style="{'border-left: 4px solid #ff9800; background: rgba(255, 152, 0, 0.05);' if job.get('is_top_company', False) else ''}">
                <div class="tech-content-inner">
                    <a href="{job['url']}" class="tech-headline" target="_blank">
                        💻 {safe_job_title(job["title"])}{top_badge}{company_text}
                    </a>
                    <div class="tech-source">
                        <a href="{job['url']}" target="_blank">
                            📍 {job['source']} → Apply Now
                        </a>
                        {'<span style="background: #ff9800; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 10px; font-weight: bold;">TOP COMPANY</span>' if job.get('is_top_company', False) else ''}
                    </div>
                </div>
            </div>\n'''
    
    # ====== FOOTER WITH STATISTICS ======
    total_jobs = len(electrical_jobs[:16]) + len(civil_jobs[:10]) + len(software_jobs[:9])
    total_top = top_electrical + top_civil + top_software
    
    content += f'''
    <div style="background: rgba(102, 126, 234, 0.1); padding: 15px 30px; border-radius: 12px; margin: 30px; text-align: center;">
        <h4 style="color: #667eea; margin: 0 0 10px; font-size: 16px;">📊 Newsletter Summary</h4>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <div style="background: white; padding: 8px 15px; border-radius: 8px; border: 1px solid #ddd;">
                <strong>Total Jobs:</strong> {total_jobs}
            </div>
            <div style="background: white; padding: 8px 15px; border-radius: 8px; border: 1px solid #ddd;">
                <strong>Top Companies:</strong> {total_top}
            </div>
            <div style="background: white; padding: 8px 15px; border-radius: 8px; border: 1px solid #ddd;">
                <strong>Electrical:</strong> {len(electrical_jobs[:16])}
            </div>
            <div style="background: white; padding: 8px 15px; border-radius: 8px; border: 1px solid #ddd;">
                <strong>Civil:</strong> {len(civil_jobs[:10])}
            </div>
            <div style="background: white; padding: 8px 15px; border-radius: 8px; border: 1px solid #ddd;">
                <strong>Software:</strong> {len(software_jobs[:9])}
            </div>
        </div>
    </div>
    '''
    
    return content

# def generate_manual_job_newsletter(sources: List[Dict]) -> str:
#     """FIXED: Generate job newsletter with proper company handling"""
    
#     # First, clean and validate all URLs
#     for job in sources:
#         job['url'] = validate_and_clean_url_standalone_job(job['url'])
#         # Ensure company field exists
#         if 'company_detected' not in job:
#             job['company_detected'] = job.get('company', 'Unknown')
#         if 'is_top_company' not in job:
#             job['is_top_company'] = False
    
#     # DEBUG: Log what we received
#     logger.info("📊 GENERATING NEWSLETTER WITH SOURCES:")
#     categories_count = {}
#     top_companies_count = {}
    
#     for job in sources:
#         cat = job.get('category', 'unknown')
#         categories_count[cat] = categories_count.get(cat, 0) + 1
        
#         company = job.get('company_detected', 'Unknown')
#         if job.get('is_top_company', False):
#             top_companies_count[company] = top_companies_count.get(company, 0) + 1
#         else:
#             # Log if company is not "Unknown" but not marked as top
#             if company != 'Unknown' and company != 'Not Found' and company != 'Not Specified':
#                 logger.debug(f"Company '{company}' not marked as top company")
    
#     logger.info(f"📊 Input categories: {categories_count}")
#     logger.info(f"🏆 Top companies in input: {top_companies_count}")
#     logger.info(f"📋 Sample jobs (first 5):")
#     for i, job in enumerate(sources[:5]):
#         logger.info(f"  {i+1}. Cat: {job.get('category')}, Company: {job.get('company_detected')}, "
#                    f"Top: {job.get('is_top_company')}, Score: {job.get('combined_score', 0):.2f}")
    
#     # FIXED FILTERING: Less aggressive filtering
#     quality_jobs = []
#     for job in sources:
#         # Basic quality checks
#         title = job['title']
        
#         # Skip obvious non-jobs
#         if 'resume' in title.lower() or 'cv' in title.lower():
#             continue
            
#         # Skip very low scores
#         if job.get('combined_score', 0) < 0.3:
#             continue
            
#         # Skip if no category (shouldn't happen but just in case)
#         if job.get('category') not in ['electrical', 'civil', 'software']:
#             continue
            
#         quality_jobs.append(job)
    
#     logger.info(f"📊 After filtering: {len(sources)} → {len(quality_jobs)} quality jobs")
    
#     if len(quality_jobs) == 0:
#         logger.warning("⚠️ No quality jobs found! Using all jobs with adjustments...")
#         # Use all jobs but with categorization
#         quality_jobs = sources
    
#     # Group jobs by ACTUAL LLM categories
#     electrical_jobs = []
#     civil_jobs = []
#     software_jobs = []
    
#     for job in quality_jobs:
#         category = job.get('category', 'other')
#         if category == 'electrical':
#             electrical_jobs.append(job)
#         elif category == 'civil':
#             civil_jobs.append(job)
#         elif category == 'software':
#             software_jobs.append(job)
#         else:
#             # Try to categorize based on title if LLM failed
#             title_lower = job['title'].lower()
#             if any(word in title_lower for word in ['electrical', 'power', 'transmission', 'solar', 'wind']):
#                 job['category'] = 'electrical'
#                 electrical_jobs.append(job)
#             elif any(word in title_lower for word in ['civil', 'construction', 'site', 'structural']):
#                 job['category'] = 'civil'
#                 civil_jobs.append(job)
#             elif any(word in title_lower for word in ['software', 'developer', 'programmer', 'it engineer']):
#                 job['category'] = 'software'
#                 software_jobs.append(job)
    
#     # Sort each category: TOP COMPANIES FIRST, then by score
#     def sort_with_top_companies_first(jobs):
#         return sorted(jobs, key=lambda x: (
#             not x.get('is_top_company', False),  # True (top company) comes before False
#             -x.get('combined_score', 0)  # Higher score first
#         ))
    
#     electrical_jobs = sort_with_top_companies_first(electrical_jobs)
#     civil_jobs = sort_with_top_companies_first(civil_jobs)
#     software_jobs = sort_with_top_companies_first(software_jobs)
    
#     # Count top companies in each category
#     top_electrical = sum(1 for j in electrical_jobs if j.get('is_top_company', False))
#     top_civil = sum(1 for j in civil_jobs if j.get('is_top_company', False))
#     top_software = sum(1 for j in software_jobs if j.get('is_top_company', False))
    
#     logger.info(f"📊 Final counts: Electrical={len(electrical_jobs)} (Top: {top_electrical}), "
#                 f"Civil={len(civil_jobs)} (Top: {top_civil}), "
#                 f"Software={len(software_jobs)} (Top: {top_software})")
    
#     # Build newsletter content
#     content = ""
    
#     # ====== TOP COMPANIES HEADER ======
#     # Collect all companies (not just top ones) for display
#     all_companies = {}
#     for job in quality_jobs:
#         company = job.get('company_detected', 'Unknown')
#         if company not in ['Unknown', 'Not Found', 'Not Specified']:
#             all_companies[company] = all_companies.get(company, 0) + 1
    
#     if all_companies:
#         # Sort by count
#         sorted_companies = sorted(all_companies.items(), key=lambda x: x[1], reverse=True)
#         display_companies = [company for company, count in sorted_companies[:8]]
#         total_jobs_display = sum(count for company, count in sorted_companies[:8])
        
#         content += f'''
#         <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 15px 30px; border-radius: 12px; margin: 20px 30px; text-align: center;">
#             <h3 style="color: white; margin: 0; font-size: 18px; font-weight: bold;">
#                 🏢 FEATURED COMPANIES ({total_jobs_display} Jobs)
#             </h3>
#             <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0; font-size: 14px;">
#                 {', '.join(display_companies)}{'...' if len(all_companies) > 8 else ''}
#             </p>
#         </div>
#         '''
    
#     # ====== ELECTRICAL JOBS SECTION ======
#     if electrical_jobs:
#         content += '<div class="section-header electrical-header">⚡ ELECTRICAL & POWER JOBS</div>\n'
        
#         if top_electrical > 0:
#             content += f'''
#             <div style="color: #ff9800; font-size: 14px; margin: 10px 30px 5px 30px; font-weight: 600;">
#                 🏆 {top_electrical} top company jobs in this section
#             </div>
#             '''
        
#         for job in electrical_jobs[:16]:  # Show up to 16 electrical jobs
#             # Prepare display elements
#             top_badge = " 🏆" if job.get('is_top_company', False) else ""
#             company_text = ""
            
#             company = job.get('company_detected', '')
#             if company and company not in ['Unknown', 'Not Found', 'Not Specified']:
#                 company_text = f" • {company}"
            
#             content += f'''
#             <div class="tech-card" style="{'border-left: 4px solid #ff9800; background: rgba(255, 152, 0, 0.05);' if job.get('is_top_company', False) else ''}">
#                 <div class="tech-content-inner">
#                     <a href="{job['url']}" class="tech-headline" target="_blank">
#                         ⚡ {safe_job_title(job["title"])}{top_badge}{company_text}
#                     </a>
#                     <div class="tech-source">
#                         <a href="{job['url']}" target="_blank">
#                             📍 {job['source']} → Apply Now
#                         </a>
#                         {'<span style="background: #ff9800; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 10px; font-weight: bold;">TOP COMPANY</span>' if job.get('is_top_company', False) else ''}
#                     </div>
#                 </div>
#             </div>\n'''
#     else:
#         # Show message if no electrical jobs found
#         content += '''
#         <div class="section-header electrical-header">⚡ ELECTRICAL & POWER JOBS</div>
#         <div style="padding: 20px 30px; text-align: center; color: #666;">
#             <p>No electrical/power engineering jobs found in this batch.</p>
#         </div>
#         '''
    
#     # ====== CIVIL JOBS SECTION ======
#     if civil_jobs:
#         content += '<div class="section-header civil-header">🏗️ CIVIL & CONSTRUCTION JOBS</div>\n'
        
#         if top_civil > 0:
#             content += f'''
#             <div style="color: #ff9800; font-size: 14px; margin: 10px 30px 5px 30px; font-weight: 600;">
#                 🏆 {top_civil} top company jobs in this section
#             </div>
#             '''
        
#         for job in civil_jobs[:10]:  # Show up to 10 civil jobs
#             top_badge = " 🏆" if job.get('is_top_company', False) else ""
#             company_text = ""
            
#             company = job.get('company_detected', '')
#             if company and company not in ['Unknown', 'Not Found', 'Not Specified']:
#                 company_text = f" • {company}"
            
#             content += f'''
#             <div class="tech-card" style="{'border-left: 4px solid #ff9800; background: rgba(255, 152, 0, 0.05);' if job.get('is_top_company', False) else ''}">
#                 <div class="tech-content-inner">
#                     <a href="{job['url']}" class="tech-headline" target="_blank">
#                         🏗️ {safe_job_title(job["title"])}{top_badge}{company_text}
#                     </a>
#                     <div class="tech-source">
#                         <a href="{job['url']}" target="_blank">
#                             📍 {job['source']} → Apply Now
#                         </a>
#                         {'<span style="background: #ff9800; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 10px; font-weight: bold;">TOP COMPANY</span>' if job.get('is_top_company', False) else ''}
#                     </div>
#                 </div>
#             </div>\n'''
#     else:
#         # Show message if no civil jobs found
#         content += '''
#         <div class="section-header civil-header">🏗️ CIVIL & CONSTRUCTION JOBS</div>
#         <div style="padding: 20px 30px; text-align: center; color: #666;">
#             <p>No civil/construction engineering jobs found in this batch.</p>
#         </div>
#         '''
    
#     # ====== SOFTWARE JOBS SECTION ======
#     if software_jobs:
#         content += '<div class="section-header software-header">💻 SOFTWARE & IT JOBS</div>\n'
        
#         if top_software > 0:
#             content += f'''
#             <div style="color: #ff9800; font-size: 14px; margin: 10px 30px 5px 30px; font-weight: 600;">
#                 🏆 {top_software} top company jobs in this section
#             </div>
#             '''
        
#         for job in software_jobs[:9]:  # Show up to 9 software jobs
#             top_badge = " 🏆" if job.get('is_top_company', False) else ""
#             company_text = ""
            
#             company = job.get('company_detected', '')
#             if company and company not in ['Unknown', 'Not Found', 'Not Specified']:
#                 company_text = f" • {company}"
            
#             content += f'''
#             <div class="tech-card" style="{'border-left: 4px solid #ff9800; background: rgba(255, 152, 0, 0.05);' if job.get('is_top_company', False) else ''}">
#                 <div class="tech-content-inner">
#                     <a href="{job['url']}" class="tech-headline" target="_blank">
#                         💻 {safe_job_title(job["title"])}{top_badge}{company_text}
#                     </a>
#                     <div class="tech-source">
#                         <a href="{job['url']}" target="_blank">
#                             📍 {job['source']} → Apply Now
#                         </a>
#                         {'<span style="background: #ff9800; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 10px; font-weight: bold;">TOP COMPANY</span>' if job.get('is_top_company', False) else ''}
#                     </div>
#                 </div>
#             </div>\n'''
#     else:
#         # Show message if no software jobs found
#         content += '''
#         <div class="section-header software-header">💻 SOFTWARE & IT JOBS</div>
#         <div style="padding: 20px 30px; text-align: center; color: #666;">
#             <p>No software/IT engineering jobs found in this batch.</p>
#         </div>
#         '''
    
#     # ====== FOOTER WITH STATISTICS ======
#     total_jobs = len(electrical_jobs[:16]) + len(civil_jobs[:10]) + len(software_jobs[:9])
#     total_top = top_electrical + top_civil + top_software
    
#     content += f'''
#     <div style="background: rgba(102, 126, 234, 0.1); padding: 15px 30px; border-radius: 12px; margin: 30px; text-align: center;">
#         <h4 style="color: #667eea; margin: 0 0 10px; font-size: 16px;">📊 Newsletter Summary</h4>
#         <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
#             <div style="background: white; padding: 8px 15px; border-radius: 8px; border: 1px solid #ddd;">
#                 <strong>Total Jobs:</strong> {total_jobs}
#             </div>
#             <div style="background: white; padding: 8px 15px; border-radius: 8px; border: 1px solid #ddd;">
#                 <strong>Top Companies:</strong> {total_top}
#             </div>
#             <div style="background: white; padding: 8px 15px; border-radius: 8px; border: 1px solid #ddd;">
#                 <strong>Electrical:</strong> {len(electrical_jobs[:16])}
#             </div>
#             <div style="background: white; padding: 8px 15px; border-radius: 8px; border: 1px solid #ddd;">
#                 <strong>Civil:</strong> {len(civil_jobs[:10])}
#             </div>
#             <div style="background: white; padding: 8px 15px; border-radius: 8px; border: 1px solid #ddd;">
#                 <strong>Software:</strong> {len(software_jobs[:9])}
#             </div>
#         </div>
#     </div>
#     '''
    
#     return content


def create_jobs_newsletter(topic: str, content: str, sources: List[Dict], publish_date: str):
    """Create jobs-specific newsletter - KEEP EXISTING STYLING"""
    words = len(re.sub('<[^>]*>', '', content).split())
    read_time = max(1, round(words / 220))
   
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Jobs Daily - Career Opportunities</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
           
            body {{
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                line-height: 1.6;
                color: #1a1a1a;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
           
            .tech-container {{
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                overflow: hidden;
                box-shadow: 0 25px 50px rgba(0,0,0,0.15);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
           
            .tech-header {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 40px 30px;
                text-align: center;
                position: relative;
                overflow: hidden;
            }}
           
            .tech-header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            }}
           
            .tech-header h1 {{
                font-size: 36px;
                font-weight: 800;
                margin-bottom: 15px;
                position: relative;
                z-index: 2;
            }}
           
            .tech-header p {{
                font-size: 16px;
                opacity: 0.9;
                font-weight: 600;
                position: relative;
                z-index: 2;
            }}
           
            .tech-content {{
                padding-left: 5px;
            }}
           
            .section-header {{
                font-size: 22px;
                font-weight: 800;
                margin: 35px 0 25px;
                padding: 8px 30px;
                border-radius: 16px;
                display: flex;
                align-items: center;
                gap: 15px;
                position: relative;
                overflow: hidden;
                color: white;
            }}
           
            .section-header {{
                background: linear-gradient(135deg, #667eea, #764ba2);
            }}
           
            .tech-card {{
                background: rgba(248, 249, 250, 0.8);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                margin: 20px 30px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                transition: all 0.4s ease;
                overflow: hidden;
                position: relative;
            }}
           
            .tech-card:hover {{
                transform: translateY(-8px);
                box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
                border-color: #667eea;
            }}
           
            .tech-content-inner {{
                padding: 0px;
            }}
           
            .tech-headline {{
                font-size: 18px;
                font-weight: 700;
                color: #2d3436;
                text-decoration: none;
                display: block;
                margin-bottom: 15px;
                line-height: 1.5;
                transition: color 0.3s ease;
            }}
           
            .tech-headline:hover {{
                color: #667eea;
            }}
           
            .tech-source {{
                background: rgba(102, 126, 234, 0.08);
                padding: 12px 20px;
                border-radius: 12px;
                margin-top: 15px;
                border-left: 4px solid #667eea;
            }}
           
            .tech-source a {{
                color: #636e72;
                font-size: 13px;
                text-decoration: none;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
                transition: color 0.2s ease;
            }}
           
            .tech-source a:hover {{
                color: #667eea;
            }}
           
            @media (max-width: 768px) {{
                body {{
                    padding: 0px;
                }}
               
                .tech-header {{
                    padding: 30px 20px;
                }}
               
                .tech-header h1 {{
                    font-size: 28px;
                }}
               
                .section-header {{
                    font-size: 20px;
                    margin: 30px 0 20px;
                    padding: 8px 20px;
                }}
               
                .tech-card {{
                    margin: 15px 0px;
                }}
               
                .tech-headline {{
                    font-size: 16px;
                }}
            }}

            .electrical-header {{
                background: linear-gradient(135deg, #0f4c75, #3282b8) !important;
            }}

            .civil-header {{
                background: linear-gradient(135deg, #feca57, #ff9ff3) !important;
            }}

            .software-header {{
                background: linear-gradient(135deg, #5f27cd, #341f97) !important;
            }}

            


            /* Add these styles to the existing CSS */
            .top-companies-header {{
                margin: 20px 30px;
            }}

            .tech-card.top-company {{
                border-left: 4px solid #ff9800 !important;
                background: rgba(255, 152, 0, 0.05) !important;
            }}

            .tech-card.top-company:hover {{
                box-shadow: 0 20px 40px rgba(255, 152, 0, 0.2) !important;
            }}

            .top-company-badge {{
                background: #ff9800;
                color: white;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 11px;
                font-weight: bold;
                margin-left: 10px;
                display: inline-block;
            }}

            .company-name {{
                color: #ff9800;
                font-weight: 600;
                font-size: 14px;
                margin-left: 5px;
            }}


        </style>
    </head>
    <body>
        <div class="tech-container">
            <div class="tech-header">
                <h1>💼 JOBS DAILY</h1>
                <p>{publish_date} • {len(sources)} Sources • Career Opportunities</p>
            </div>
           
            <div class="tech-content">
                {content}
            </div>
        </div>
       
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const links = document.querySelectorAll('a[href^="http"]');
                links.forEach(link => {{
                    link.setAttribute('target', '_blank');
                    link.setAttribute('rel', 'noopener noreferrer');
                }});
            }});
        </script>
    </body>
    </html>
    """
   
    return html, words, read_time

# ============================
# UTILITY FUNCTIONS
# ============================

def remove_duplicate_news(news_list: List[Dict]) -> List[Dict]:
    """Remove duplicate news based on title similarity - STANDALONE FUNCTION"""
    unique_news = []
    seen_titles = set()
    
    for news in news_list:
        title = news['title'].lower()
        title_key = re.sub(r'[^\w\s]', '', title)
        title_key = ' '.join(title_key.split()[:8])
        
        duplicate_checks = [
            title_key,
            title_key.replace('power grid', 'powergrid'),
            title_key.replace('bc jindal', 'b c jindal'),
            title_key.replace('mnre', 'ministry of renewable energy'),
            title_key.replace('secures', 'wins'),
            title_key.replace('acquires', 'buys'),
        ]
        
        is_duplicate = any(check in seen_titles for check in duplicate_checks)
        
        if not is_duplicate:
            seen_titles.add(title_key)
            unique_news.append(news)
        else:
            logger.info(f"🔍 Removed duplicate: {news['title'][:60]}...")
    
    logger.info(f"🔄 Deduplication: {len(news_list)} → {len(unique_news)} unique news")
    return unique_news

async def generate_simple_newsletter_content(sources: List[Dict], topic: str) -> str:
    """Fallback for other topics"""
    unique_sources = remove_duplicate_news(sources)
    content = f"## Latest {topic.title()} Updates\n\n"
    for source in unique_sources[:35]:
        content += f"[{source['title']}]({source['url']})\n\n"
    return content

def create_consistent_newsletter(topic: str, content: str, sources: List[Dict], publish_date: str):
    """Fallback template for other topics"""
    words = len(re.sub('<[^>]*>', '', content).split())
    read_time = max(1, round(words / 220))
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{topic} Daily</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .header {{ background: #f0f0f0; padding: 20px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{topic.upper()} DAILY</h1>
                <p>{publish_date} • {len(sources)} Sources</p>
            </div>
            <div class="content">
                {content}
            </div>
        </div>
    </body>
    </html>
    """
    return html, words, read_time

def create_basic_newsletter(topic: str, sources: List[Dict], publish_date: str):
    items = ""
    for s in sources[:6]:
        items += f'<div class="card"><a href="{s["url"]}" class="title-link">{s["title"]}</a><p>{s["content"][:200]}...</p><small>{s["source"]}</small></div>'
    html = f'<div class="newsletter"><div class="header"><h1>{topic.upper()} Brief</h1><p>{publish_date}</p></div>{items}</div>'
    return {"html": html, "metadata": {"topic": topic, "real_sources": False}, "raw_sources": sources[:6]}

def validate_sources_urls(sources: List[Dict]) -> List[Dict]:
    """Ensure all sources have valid, non-RSS URLs"""
    validated_sources = []
    
    for source in sources:
        url = source.get('url', '')
        
        # Fix RSS URLs
        if 'news.google.com/rss/articles/' in url:
            article_match = re.search(r'/articles/([^?]+)', url)
            if article_match:
                article_id = article_match.group(1)
                source['url'] = f"https://news.google.com/articles/{article_id}?hl=en-IN&gl=IN&ceid=IN:en"
                logger.info(f"Fixed RSS URL in source: {source['title'][:50]}...")
        
        validated_sources.append(source)
    
    return validated_sources

async def generate_dynamic_newsletter(
    topic: str,
    publish_date: str = None,
    keywords: List[str] = None,
    style: str = "professional",
    news_searcher_instance: EnergyNewsSearcher = None
) -> str:
    try:
        if publish_date is None:
            publish_date = datetime.now(timezone.utc).date().isoformat()

        logger.info(f"Starting newsletter generation for: {topic} with keywords: {keywords}")
        
        if news_searcher_instance is None:
            raise ValueError("news_searcher_instance must be provided via context manager")
        searcher = news_searcher_instance
        
        # PASS KEYWORDS to search function
        sources = await searcher.search_topic_specific(topic, num_results=80, keywords=keywords)
        
        if not sources:
            logger.warning(f"No sources found for {topic}, using demo data")
            sources = await get_demo_sources(topic)

        logger.info(f"Processing {len(sources)} keyword-matched sources for newsletter")
        
        # 🚨 TEMPORARY DEBUG - ADD THIS RIGHT HERE
        # logger.info("🔗 FINAL URL CHECK BEFORE NEWSLETTER:")
        # for i, source in enumerate(sources[:5]):
        #     logger.info(f"  {i+1}. Title: {source['title'][:50]}...")
        #     logger.info(f"     URL: {source['url']}")
        #     logger.info(f"     URL Type: {'RSS' if 'rss/articles' in source['url'] else 'ARTICLE'}")
        
        # Continue with newsletter processing
        newsletter_data = await process_with_killer_formatting(
            topic, sources, publish_date, keywords or [], style
        )

        return json.dumps(newsletter_data, indent=2)

    except Exception as e:
        logger.error(f"Newsletter generation error: {e}")
        return json.dumps({"error": f"Generation failed: {str(e)}"})


async def get_demo_sources(topic: str) -> List[Dict]:
    """Provide demo sources when real sources aren't available"""
    demo_sources = {
        "india_power_projects": [
            {
                "title": "NTPC commissions 250 MW solar project in Rajasthan",
                "url": "https://economictimes.indiatimes.com/industry/energy/power/ntpc-commissions-250-mw-solar-project-in-rajasthan/articleshow/12345678.cms",
                "content": "NTPC has successfully commissioned a 250 MW solar power project in Bikaner, Rajasthan.",
                "source": "Economic Times",
                "published_at": "2024-12-19T10:30:00Z",
                "relevance_score": 0.8
            }
        ],
        "technology": [
            {
                "title": "AI startup raises $50M for Indian language models",
                "url": "https://techcrunch.com/ai-startup-funding",
                "content": "Indian AI startup secures major funding for developing regional language AI models.",
                "source": "TechCrunch",
                "published_at": "2024-12-19T09:15:00Z",
                "relevance_score": 0.7
            }
        ],
        "sports": [
            {
                "title": "India wins thrilling T20 match against Australia",
                "url": "https://espn.com/cricket/match123",
                "content": "Team India clinches victory in last over thriller against Australia.",
                "source": "ESPN",
                "published_at": "2024-12-19T14:20:00Z",
                "relevance_score": 0.9
            }
        ],
        "hiring_jobs": [
            {
                "title": "Tech Company Hiring Senior Software Engineers",
                "url": "https://linkedin.com/jobs/senior-software-engineer",
                "content": "Leading tech company is hiring senior software engineers with 5+ years experience.",
                "source": "LinkedIn",
                "published_at": "2024-12-19T08:00:00Z",
                "relevance_score": 0.9
            },
            {
                "title": "Multiple Openings at Startup - Full Stack Developers",
                "url": "https://naukri.com/job123",
                "content": "Fast-growing startup hiring full stack developers with React and Node.js experience.",
                "source": "Naukri.com",
                "published_at": "2024-12-19T10:30:00Z",
                "relevance_score": 0.8
            }
        ]
    }
    
    return demo_sources.get(topic, [
        {
            "title": f"Latest updates in {topic}",
            "url": "https://example.com",
            "content": f"Recent developments and news in the {topic} sector.",
            "source": "News",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "relevance_score": 0.5
        }
    ])
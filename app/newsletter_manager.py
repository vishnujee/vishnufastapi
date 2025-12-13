# newsletter_manager.py
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional

from app.newsletter_database import NewsletterDB

logger = logging.getLogger(__name__)

class NewsletterManager:
    def __init__(self):
        self.db = NewsletterDB()
        self.storage_path = Path("newsletter_storage")
        self.storage_path.mkdir(exist_ok=True)
        
        # Create topic directories
        self.topics = ["technology", "sports", "india_power_projects", "hiring_jobs"]
        for topic in self.topics:
            (self.storage_path / topic).mkdir(parents=True, exist_ok=True)
        
        logger.info("NewsletterManager initialized")
    
    def generate_daily_newsletter(self, topic: str, force_new: bool = False) -> Dict:
        """Generate or retrieve daily newsletter"""
        try:
            today = datetime.now().date().isoformat()
            
            # Check if today's newsletter already exists
            if not force_new:
                existing = self.db.get_active_newsletter(topic)
                if existing and existing['publish_date'] == today:
                    logger.info(f"Using existing newsletter for {topic} on {today}")
                    return existing
            
            logger.info(f"Generating new newsletter for {topic}")
            
            # Generate new newsletter (using your existing generator)
            # This assumes you have the generate_dynamic_newsletter function
            from app.newsletter import EnhancedNewsSearcher, generate_dynamic_newsletter
            import asyncio
            
            async def generate():
                async with EnhancedNewsSearcher() as searcher:
                    result_json = await generate_dynamic_newsletter(
                        topic=topic,
                        publish_date=today,
                        keywords=[],
                        style="professional",
                        news_searcher_instance=searcher
                    )
                    return json.loads(result_json)
            
            result = asyncio.run(generate())
            
            # Prepare newsletter data
            newsletter_data = {
                'topic': topic,
                'title': f"{topic.replace('_', ' ').title()} Daily - {today}",
                'html_content': result['html'],
                'publish_date': today,
                'metadata': result.get('metadata', {})
            }
            
            # Save to database
            newsletter_id = self.db.save_newsletter(newsletter_data)
            
            # Save HTML file
            self.save_html_file(topic, today, result['html'])
            
            logger.info(f"Newsletter generated and saved: ID={newsletter_id}")
            
            return {
                **newsletter_data,
                'id': newsletter_id,
                'file_path': str(self.get_file_path(topic, today))
            }
            
        except Exception as e:
            logger.error(f"Error generating newsletter: {e}")
            raise
    
    def get_newsletter_view(self, topic: str, date: str = None) -> Optional[Dict]:
        """Get newsletter for viewing"""
        try:
            if not date:
                date = datetime.now().date().isoformat()
            
            # Try to get from database first
            newsletter = self.db.get_newsletter_by_date(topic, date)
            if newsletter:
                return newsletter
            
            # Try to load from file
            file_path = self.get_file_path(topic, date)
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    html = f.read()
                
                return {
                    'topic': topic,
                    'title': f"{topic.replace('_', ' ').title()} - {date}",
                    'html_content': html,
                    'publish_date': date,
                    'metadata': {
                        'topic': topic,
                        'publish_date': date,
                        'sources_used': 35,
                        'word_count': len(html.split()),
                        'estimated_read_time': len(html.split()) // 200
                    }
                }
            
            # Return None if not found
            return None
            
        except Exception as e:
            logger.error(f"Error getting newsletter view: {e}")
            return None
    
    def save_html_file(self, topic: str, date: str, html: str):
        """Save HTML to file"""
        try:
            file_path = self.get_file_path(topic, date)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"HTML file saved: {file_path}")
        except Exception as e:
            logger.error(f"Error saving HTML file: {e}")
    
    def get_file_path(self, topic: str, date: str) -> Path:
        """Get file path for newsletter"""
        return self.storage_path / topic / f"{date}.html"
    
    def get_all_newsletters(self, limit: int = 20) -> List[Dict]:
        """Get all newsletters across all topics"""
        try:
            return self.db.get_all_newsletters(limit)
        except Exception as e:
            logger.error(f"Error getting all newsletters: {e}")
            return []
    
    def get_topic_newsletters(self, topic: str, limit: int = 7) -> List[Dict]:
        """Get newsletters for a specific topic"""
        try:
            return self.db.get_newsletter_history(topic, limit)
        except Exception as e:
            logger.error(f"Error getting topic newsletters: {e}")
            return []
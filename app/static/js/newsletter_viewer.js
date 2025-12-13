// static/js/newsletter_viewer.js
class NewsletterViewer {
    constructor() {
        this.currentTopic = 'technology';
        this.currentDate = new Date().toISOString().split('T')[0];
        this.newsletterCache = new Map();
    }
    
    async loadNewsletter(topic = null, date = null) {
        if (topic) this.currentTopic = topic;
        if (date) this.currentDate = date;
        
        const loadingEl = document.getElementById('newsletter-loading');
        const contentEl = document.getElementById('newsletter-content');
        const errorEl = document.getElementById('newsletter-error');
        
        // Show loading
        loadingEl.classList.remove('hidden');
        contentEl.classList.add('hidden');
        errorEl.classList.add('hidden');
        
        try {
            // Check cache first
            const cacheKey = `${this.currentTopic}-${this.currentDate}`;
            if (this.newsletterCache.has(cacheKey)) {
                this.displayNewsletter(this.newsletterCache.get(cacheKey));
                return;
            }
            
            // Fetch from API
            const response = await fetch(`/api/newsletter/${this.currentTopic}?date=${this.currentDate}`);
            const data = await response.json();
            
            if (data.success) {
                // Cache the result
                this.newsletterCache.set(cacheKey, data.newsletter);
                
                // Display newsletter
                this.displayNewsletter(data.newsletter);
                
                // Update UI
                this.updateUI(data.newsletter);
            } else {
                throw new Error('Failed to load newsletter');
            }
        } catch (error) {
            console.error('Error loading newsletter:', error);
            errorEl.classList.remove('hidden');
            errorEl.innerHTML = `
                <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                    <div class="flex items-center">
                        <i class="fas fa-exclamation-triangle text-red-500 mr-3"></i>
                        <div>
                            <h3 class="font-semibold text-red-800">Failed to load newsletter</h3>
                            <p class="text-red-700 text-sm mt-1">${error.message}</p>
                        </div>
                    </div>
                    <button onclick="window.location.reload()" 
                            class="mt-3 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg text-sm">
                        Try Again
                    </button>
                </div>
            `;
        } finally {
            loadingEl.classList.add('hidden');
        }
    }
    
    displayNewsletter(newsletter) {
        const contentEl = document.getElementById('newsletter-content');
        const metadataEl = document.getElementById('newsletter-metadata');
        
        // Display HTML content
        contentEl.innerHTML = newsletter.html_content;
        
        // Display metadata
        const metadata = newsletter.metadata || {};
        metadataEl.innerHTML = `
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div class="bg-blue-50 p-3 rounded-lg text-center">
                    <div class="text-2xl font-bold text-blue-600">${metadata.sources_used || 35}</div>
                    <div class="text-sm text-blue-800">Sources</div>
                </div>
                <div class="bg-green-50 p-3 rounded-lg text-center">
                    <div class="text-2xl font-bold text-green-600">${metadata.word_count || 0}</div>
                    <div class="text-sm text-green-800">Words</div>
                </div>
                <div class="bg-purple-50 p-3 rounded-lg text-center">
                    <div class="text-2xl font-bold text-purple-600">${metadata.estimated_read_time || 5}</div>
                    <div class="text-sm text-purple-800">Min Read</div>
                </div>
                <div class="bg-orange-50 p-3 rounded-lg text-center">
                    <div class="text-2xl font-bold text-orange-600">${newsletter.publish_date}</div>
                    <div class="text-sm text-orange-800">Published</div>
                </div>
            </div>
        `;
        
        contentEl.classList.remove('hidden');
        
        // Add newsletter-specific styles
        this.addNewsletterStyles();
    }
    
    updateUI(newsletter) {
        // Update topic selector
        document.querySelectorAll('.topic-tab').forEach(tab => {
            tab.classList.remove('active');
            if (tab.dataset.topic === this.currentTopic) {
                tab.classList.add('active');
            }
        });
        
        // Update date selector
        const dateInput = document.getElementById('newsletter-date');
        if (dateInput) {
            dateInput.value = this.currentDate;
        }
        
        // Update title
        const titleEl = document.getElementById('newsletter-title');
        if (titleEl) {
            titleEl.textContent = newsletter.title || `${this.currentTopic.replace('_', ' ').title()} Daily`;
        }
        
        // Update share buttons
        this.updateShareButtons(newsletter);
    }
    
    addNewsletterStyles() {
        // Ensure newsletter-specific styles are applied
        const styleId = 'newsletter-dynamic-styles';
        let styleEl = document.getElementById(styleId);
        
        if (!styleEl) {
            styleEl = document.createElement('style');
            styleEl.id = styleId;
            document.head.appendChild(styleEl);
        }
        
        // Add responsive styles for newsletter content
        styleEl.textContent = `
            .newsletter-content img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
            }
            
            .newsletter-content a {
                color: #2563eb;
                text-decoration: none;
                transition: color 0.2s;
            }
            
            .newsletter-content a:hover {
                color: #1d4ed8;
                text-decoration: underline;
            }
            
            @media (max-width: 768px) {
                .newsletter-content .section-header {
                    font-size: 1.25rem !important;
                    padding: 0.5rem 1rem !important;
                }
                
                .newsletter-content .tech-card,
                .newsletter-content .sports-card,
                .newsletter-content .power-card {
                    margin: 0.75rem 0 !important;
                    padding: 0.75rem !important;
                }
            }
        `;
    }
    
    updateShareButtons(newsletter) {
        const shareUrl = `${window.location.origin}/newsletter/${this.currentTopic}/${this.currentDate}`;
        const shareText = `Check out today's ${this.currentTopic.replace('_', ' ')} newsletter`;
        
        document.getElementById('share-twitter').href = 
            `https://twitter.com/intent/tweet?url=${encodeURIComponent(shareUrl)}&text=${encodeURIComponent(shareText)}`;
        
        document.getElementById('share-linkedin').href = 
            `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`;
        
        document.getElementById('copy-link').onclick = () => {
            navigator.clipboard.writeText(shareUrl).then(() => {
                alert('Link copied to clipboard!');
            });
        };
    }
    
    async generateNewNewsletter() {
        if (!confirm('Generate a new newsletter? This may take 1-2 minutes.')) {
            return;
        }
        
        const generateBtn = document.getElementById('generate-new-btn');
        const originalText = generateBtn.innerHTML;
        
        generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Generating...';
        generateBtn.disabled = true;
        
        try {
            const response = await fetch(`/api/newsletter/${this.currentTopic}/generate`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Start polling for completion
                this.pollForNewNewsletter();
            } else {
                throw new Error(data.message || 'Generation failed');
            }
        } catch (error) {
            alert('Failed to start generation: ' + error.message);
            generateBtn.innerHTML = originalText;
            generateBtn.disabled = false;
        }
    }
    
    async pollForNewNewsletter() {
        const maxAttempts = 30; // 30 attempts with 5-second intervals = 2.5 minutes
        let attempts = 0;
        
        const pollInterval = setInterval(async () => {
            attempts++;
            
            try {
                const response = await fetch(`/api/newsletter/status/${this.currentTopic}`);
                const data = await response.json();
                
                if (data.has_todays_newsletter) {
                    clearInterval(pollInterval);
                    alert('New newsletter generated successfully!');
                    this.loadNewsletter(); // Reload with new newsletter
                    
                    // Reset generate button
                    const generateBtn = document.getElementById('generate-new-btn');
                    generateBtn.innerHTML = '<i class="fas fa-sync mr-2"></i> Generate New';
                    generateBtn.disabled = false;
                } else if (attempts >= maxAttempts) {
                    clearInterval(pollInterval);
                    alert('Newsletter generation timed out. Please try again.');
                    
                    const generateBtn = document.getElementById('generate-new-btn');
                    generateBtn.innerHTML = '<i class="fas fa-sync mr-2"></i> Generate New';
                    generateBtn.disabled = false;
                }
            } catch (error) {
                console.error('Polling error:', error);
            }
        }, 5000); // Poll every 5 seconds
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.newsletterViewer = new NewsletterViewer();
    
    // Load default newsletter
    newsletterViewer.loadNewsletter();
    
    // Setup event listeners
    document.querySelectorAll('.topic-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            newsletterViewer.loadNewsletter(tab.dataset.topic);
        });
    });
    
    const dateInput = document.getElementById('newsletter-date');
    if (dateInput) {
        dateInput.addEventListener('change', (e) => {
            newsletterViewer.loadNewsletter(null, e.target.value);
        });
        
        // Set max date to today
        dateInput.max = new Date().toISOString().split('T')[0];
    }
});
# newsletter_database.py
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import os
import logging

logger = logging.getLogger(__name__)

class NewsletterDB:
    def __init__(self, db_path: str = "newsletters.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Newsletters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS newsletters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                title TEXT NOT NULL,
                html_content TEXT NOT NULL,
                publish_date DATE NOT NULL,
                expiry_date DATE,
                status TEXT DEFAULT 'active',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(topic, publish_date)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_newsletters_topic_date ON newsletters(topic, publish_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_newsletters_status ON newsletters(status)')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_newsletter(self, newsletter_data: Dict) -> int:
        """Save newsletter to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate expiry date (next newsletter date)
            publish_date = newsletter_data['publish_date']
            try:
                expiry_date = (datetime.strptime(publish_date, '%Y-%m-%d') + timedelta(days=1)).date().isoformat()
            except:
                expiry_date = (datetime.now() + timedelta(days=1)).date().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO newsletters 
                (topic, title, html_content, publish_date, expiry_date, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                newsletter_data['topic'],
                newsletter_data.get('title', f"{newsletter_data['topic'].title()} Daily"),
                newsletter_data['html_content'],
                newsletter_data['publish_date'],
                expiry_date,
                json.dumps(newsletter_data.get('metadata', {}))
            ))
            
            newsletter_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            logger.info(f"Newsletter saved: ID={newsletter_id}, Topic={newsletter_data['topic']}, Date={newsletter_data['publish_date']}")
            return newsletter_id
            
        except Exception as e:
            logger.error(f"Error saving newsletter: {e}")
            raise
    
    def get_active_newsletter(self, topic: str) -> Optional[Dict]:
        """Get today's active newsletter for a topic"""
        try:
            today = datetime.now().date().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # First try to get today's newsletter
            cursor.execute('''
                SELECT * FROM newsletters 
                WHERE topic = ? AND publish_date = ? 
                ORDER BY created_at DESC LIMIT 1
            ''', (topic, today))
            
            row = cursor.fetchone()
            if not row:
                # Get the latest newsletter for this topic
                cursor.execute('''
                    SELECT * FROM newsletters 
                    WHERE topic = ? 
                    ORDER BY publish_date DESC LIMIT 1
                ''', (topic,))
                row = cursor.fetchone()
            
            conn.close()
            
            if row:
                newsletter = dict(row)
                try:
                    newsletter['metadata'] = json.loads(newsletter['metadata'])
                except:
                    newsletter['metadata'] = {}
                return newsletter
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting active newsletter: {e}")
            return None
    
    def get_newsletter_history(self, topic: str, limit: int = 7) -> List[Dict]:
        """Get newsletter history for a topic"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM newsletters 
                WHERE topic = ? 
                ORDER BY publish_date DESC 
                LIMIT ?
            ''', (topic, limit))
            
            newsletters = []
            for row in cursor.fetchall():
                newsletter = dict(row)
                try:
                    newsletter['metadata'] = json.loads(newsletter['metadata'])
                except:
                    newsletter['metadata'] = {}
                newsletters.append(newsletter)
            
            conn.close()
            return newsletters
            
        except Exception as e:
            logger.error(f"Error getting newsletter history: {e}")
            return []
    

    def get_all_newsletters(self, limit: int = 20) -> List[Dict]:
        """Get all newsletters across all topics"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM newsletters 
                ORDER BY publish_date DESC, created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            newsletters = []
            for row in cursor.fetchall():
                newsletter = dict(row)
                try:
                    newsletter['metadata'] = json.loads(newsletter['metadata'])
                except:
                    newsletter['metadata'] = {}
                newsletters.append(newsletter)
            
            conn.close()
            return newsletters
            
        except Exception as e:
            logger.error(f"Error getting all newsletters: {e}")
            return []
    
    def expire_old_newsletters(self):
        """Mark old newsletters as expired"""
        try:
            today = datetime.now().date().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE newsletters 
                SET status = 'expired' 
                WHERE expiry_date < ? AND status = 'active'
            ''', (today,))
            
            affected = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Expired {affected} old newsletters")
            
        except Exception as e:
            logger.error(f"Error expiring old newsletters: {e}")
    
    def get_newsletter_by_date(self, topic: str, date: str) -> Optional[Dict]:
        """Get newsletter for specific date"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM newsletters 
                WHERE topic = ? AND publish_date = ?
                LIMIT 1
            ''', (topic, date))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                newsletter = dict(row)
                try:
                    newsletter['metadata'] = json.loads(newsletter['metadata'])
                except:
                    newsletter['metadata'] = {}
                return newsletter
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting newsletter by date: {e}")
            return None
        

    def get_latest_newsletters_by_topic(self, date_limit: str = None):
        """Get the latest newsletter for EACH topic. If date_limit is provided, 
        get the latest one ON or BEFORE that date."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # This query groups by topic and gets the most recent publish_date for each.
            if date_limit:
                # Get the latest newsletter for each topic published ON or BEFORE the limit date
                cursor.execute('''
                    SELECT n1.* FROM newsletters n1
                    INNER JOIN (
                        SELECT topic, MAX(publish_date) as max_date 
                        FROM newsletters 
                        WHERE publish_date <= ?
                        GROUP BY topic
                    ) n2 ON n1.topic = n2.topic AND n1.publish_date = n2.max_date
                    ORDER BY n1.created_at DESC
                ''', (date_limit,))
            else:
                # Simply get the absolute latest newsletter for each topic
                cursor.execute('''
                    SELECT n1.* FROM newsletters n1
                    INNER JOIN (
                        SELECT topic, MAX(publish_date) as max_date 
                        FROM newsletters 
                        GROUP BY topic
                    ) n2 ON n1.topic = n2.topic AND n1.publish_date = n2.max_date
                    ORDER BY n1.created_at DESC
                ''')
            
            newsletters = []
            for row in cursor.fetchall():
                newsletter = dict(row)
                try:
                    newsletter['metadata'] = json.loads(newsletter['metadata'])
                except:
                    newsletter['metadata'] = {}
                newsletters.append(newsletter)
            
            conn.close()
            return newsletters
            
        except Exception as e:
            logger.error(f"Error getting latest newsletters by topic: {e}")
            return []
        

    def get_latest_newsletter_for_topic(self, topic: str, date_limit: str = None) -> Optional[Dict]:
        """Get the latest newsletter for a SPECIFIC topic on or before date_limit"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if date_limit:
                # Get the latest newsletter for this topic published ON or BEFORE the limit date
                cursor.execute('''
                    SELECT * FROM newsletters 
                    WHERE topic = ? AND publish_date <= ?
                    ORDER BY publish_date DESC, created_at DESC 
                    LIMIT 1
                ''', (topic, date_limit))
            else:
                # Get the absolute latest newsletter for this topic
                cursor.execute('''
                    SELECT * FROM newsletters 
                    WHERE topic = ? 
                    ORDER BY publish_date DESC, created_at DESC 
                    LIMIT 1
                ''', (topic,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                newsletter = dict(row)
                try:
                    newsletter['metadata'] = json.loads(newsletter['metadata'])
                except:
                    newsletter['metadata'] = {}
                return newsletter
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest newsletter for topic {topic}: {e}")
            return None
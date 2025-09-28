#pip install requests beautifulsoup4 tenacity

import os
import requests
from bs4 import BeautifulSoup
import time
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock
from tenacity import retry, stop_after_attempt, wait_exponential
import random

# Configuration parameters
ARTICLES_PATH = 'articles'
DATABASE_PATH = 'articles.db'
NUM_WORKERS = 5
RATE_LIMIT = 5

# Initialize the directory for storing articles
if not os.path.exists(ARTICLES_PATH):
    os.makedirs(ARTICLES_PATH)

# Set up a global queue for managing post IDs
post_queue = Queue()

# Helper class to manage rate limiting
class RateLimiter:
    def __init__(self, rate):
        self.rate = rate  # Requests per second
        self.timestamp = time.time()
        self.lock = Lock()

    def wait(self):
        with self.lock:
            elapsed = time.time() - self.timestamp
            if elapsed < 1 / self.rate:
                time.sleep((1 / self.rate) - elapsed)
            self.timestamp = time.time()

# Ensure the required database table exists
def init_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            headline TEXT,
            published TEXT,
            category TEXT,
            source TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Retry mechanism for HTTP requests
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def scrape_article(post_id, session):
    try:
        article_url = f'https://alresalah.ps/post/{post_id}'
        response = session.get(article_url)

        if response.status_code == 404:
            print(f"No content found for post ID: {post_id}")
            return

        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        headline_tag = soup.find('h1', class_='page-post-title font-weight-bold')
        if not headline_tag:
            print(f"No headline or content found for post ID: {post_id}")
            return

        headline = headline_tag.get_text(strip=True)

        time_tag = soup.find('time', class_='d-flex align-items-center')
        time_text = time_tag.get_text(strip=True) if time_tag else 'No Date'

        # Extract category information
        category = 'No Category'
        breadcrumb = soup.find('ol', class_='breadcrumb p-0')
        if breadcrumb:
            category_link = breadcrumb.find_all('li')[-1].find('a')
            if category_link:
                category = category_link.get_text(strip=True)

        source_tag = soup.find('h4', class_='page-post-source font-size-22 text-danger')
        source_text = source_tag.get_text(strip=True) if source_tag else 'No Source'

        article_tags = soup.find_all('div', class_='p-4 bg-white')
        article_texts = []
        for article in article_tags:
            for p3_div in article.find_all('div', class_='p-3'):
                p3_div.decompose()
            article_texts.append(article.get_text(separator='\n', strip=True))

        article_content = "\n".join(article_texts)

        # Save to TXT file organized into subdirectories
        if article_content.strip():
            directory = os.path.join(ARTICLES_PATH, f'{post_id // 10000 * 10000}-{(post_id // 10000 + 1) * 10000 - 1}')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = f"{post_id}_{headline[:50].replace('/', '-')}.txt"
            filepath = os.path.join(directory, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {headline}\n")
                f.write(f"Published: {time_text}\n")
                f.write(f"Category: {category}\n")
                f.write(f"Source: {source_text}\n\n")
                f.write(article_content)
            print(f"Saved article from post ID: {post_id}")

            # Database insertion
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO articles (id, headline, published, category, source, content)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (post_id, headline, time_text, category, source_text, article_content))
            conn.commit()
            conn.close()

    except requests.exceptions.RequestException as e:
        print(f"Error in scraping article {post_id}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.content}")
            print(f"Response headers: {e.response.headers}")
        raise

# Worker function for thread execution
def worker(session, rate_limiter):
    while not post_queue.empty():
        post_id = post_queue.get()
        rate_limiter.wait()
        scrape_article(post_id, session)
        post_queue.task_done()
        time.sleep(random.uniform(1, 3))

# Main function to initiate article scraping
def scrape_all_articles(start_id, end_id):
    rate_limiter = RateLimiter(RATE_LIMIT)

    for post_id in range(start_id, end_id + 1):
        post_queue.put(post_id)

    with requests.Session() as session:
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(worker, session, rate_limiter) for _ in range(NUM_WORKERS)]
            for future in as_completed(futures):
                future.result()

# Initialize the database before starting to scrape
init_database()

# Start scraping articles with the specified range and settings
scrape_all_articles(273293, 301000)

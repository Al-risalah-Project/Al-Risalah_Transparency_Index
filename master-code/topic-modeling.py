import sqlite3
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

def load_data(database_path='articles.db'):
    """Connects to SQLite database and loads article data."""
    if not os.path.isfile(database_path):
        raise FileNotFoundError(f"Database file '{database_path}' not found.")
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT headline, published, category, content FROM articles", conn)
    conn.close()
    return df

# Arabic month to English mapping
arabic_to_english_months = {
    'يناير': 'January', 'فبراير': 'February', 'مارس': 'March',
    'أبريل': 'April', 'مايو': 'May', 'يونيو': 'June',
    'يوليو': 'July', 'أغسطس': 'August', 'سبتمبر': 'September',
    'أكتوبر': 'October', 'نوفمبر': 'November', 'ديسمبر': 'December'
}

def parse_arabic_date(date_str):
    """Converts Arabic date format to Python datetime."""
    date_pattern = re.compile(r'(\w+)\s+(\d{4})\s\.\sالساعة:\s(\d{2}:\d{2}\s[صم])')
    match = date_pattern.search(date_str)
    if not match:
        return None
    
    arabic_month = match.group(1)
    year = match.group(2)
    time = match.group(3).replace('م', 'PM').replace('ص', 'AM')
    english_month = arabic_to_english_months.get(arabic_month)
    
    if not english_month:
        return None
    
    english_date_str = f'{english_month} {year} {time}'
    try:
        return pd.to_datetime(english_date_str, format='%B %Y %I:%M %p', errors='coerce')
    except Exception as e:
        print(f"Error parsing date: {e}")
        return None

def clean_arabic_text(text):
    """Removes diacritics, punctuation, digits, and excess whitespace."""
    text = re.sub(r'[\u064B-\u0652]', '', text)  # Remove diacritics
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def remove_stopwords(text):
    """Removes Arabic stop words."""
    stop_words = set(stopwords.words('arabic'))
    words = text.split()
    return ' '.join(word for word in words if word not in stop_words)

def perform_topic_modeling(texts):
    """Performs topic modeling using LDA."""
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    for idx, topic in lda_model.print_topics():
        print(f"Topic {idx}: {topic}")

def main():
    # Load data from database
    df = load_data()

    # Apply date parsing
    df['published'] = df['published'].apply(parse_arabic_date)
    df.dropna(subset=['published'], inplace=True)

    # Clean and preprocess text
    df['content'] = df['content'].apply(clean_arabic_text)
    df['content'] = df['content'].apply(remove_stopwords)

    # Tokenize text for topic modeling
    texts = [content.split() for content in df['content']]
    perform_topic_modeling(texts)

    # Dummy sentiment analysis plot
    df['sentiment'] = df['content'].apply(lambda x: 0) # Placeholder, replace with a real sentiment analysis
    sns.histplot(df['sentiment'], kde=True)
    plt.title('Sentiment Distribution of Articles')
    plt.show()

if __name__ == "__main__":
    main()

import sqlite3
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Ensure the necessary stopwords are downloaded
nltk.download('stopwords')

# Define a mapping for Arabic to English month conversion
arabic_to_english_months = {
    'يناير': 'January', 'فبراير': 'February', 'مارس': 'March',
    'أبريل': 'April', 'مايو': 'May', 'يونيو': 'June', 
    'يوليو': 'July', 'أغسطس': 'August', 'سبتمبر': 'September',
    'أكتوبر': 'October', 'نوفمبر': 'November', 'ديسمبر': 'December'
}

def load_data(database_path='articles.db'):
    """Loads article data from SQLite database."""
    if not os.path.isfile(database_path):
        raise FileNotFoundError(f"Database file '{database_path}' not found.")
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT headline, published, category, content FROM articles", conn)
    conn.close()
    return df

def parse_arabic_date(date_str):
    """Converts Arabic-formatted date to English datetime object."""
    date_pattern = re.compile(r'(\w+)\s+(\d{4})\s\.\sالساعة:\s(\d{2}:\d{2}\s[صم])')
    match = date_pattern.search(date_str)
    if match:
        arabic_month = match.group(1)
        year = match.group(2)
        time = match.group(3).replace('م', 'PM').replace('ص', 'AM')
        english_month = arabic_to_english_months.get(arabic_month)
        english_date_str = f'{english_month} {year} {time}'
        try:
            return pd.to_datetime(english_date_str, format='%B %Y %I:%M %p', errors='coerce')
        except Exception as e:
            print(f"Error parsing date: {e}")
            return None
    else:
        return None

def clean_arabic_text(text):
    """Cleans and normalizes Arabic text."""
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

def setup_ner():
    """Sets up the NER pipeline."""
    tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
    model = AutoModelForTokenClassification.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def ner_preserve_entities(text, tokenizer, ner_pipeline, max_length=500):
    """Runs NER and preserves named entities."""
    tokens = tokenizer.tokenize(text)
    chunks = [' '.join(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length)]

    preserved_entities = []
    
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk.split())
        ner_results = ner_pipeline(chunk_text)
        last_pos = 0

        for entity in ner_results:
            entity_start, entity_end, entity_word = entity['start'], entity['end'], entity['word']
            preserved_entities.append(chunk_text[last_pos:entity_start].strip())  # Add non-entity text
            preserved_entities.append(entity_word.replace(' ', '_'))  # Preserve entity with underscore
            last_pos = entity_end

        preserved_entities.append(chunk_text[last_pos:].strip())  # Add remaining text

    return ' '.join(filter(None, preserved_entities)).strip()

def process_text(text, tokenizer, ner_pipeline):
    """Processes the text by cleaning, removing stopwords, and applying NER."""
    text = clean_arabic_text(text)
    text = remove_stopwords(text)
    text = ner_preserve_entities(text, tokenizer, ner_pipeline)
    return text

def main():
    # Load data from database
    df = load_data()

    # Apply date parsing to 'published' column
    df['published'] = df['published'].apply(parse_arabic_date)
    df.dropna(subset=['published'], inplace=True)

    # Setup the NER pipeline
    tokenizer, ner_pipeline = setup_ner()

    # Apply text processing
    df['content'] = df['content'].apply(lambda x: process_text(x, tokenizer, ner_pipeline))

    # Tokenize and prepare for topic modeling
    texts = [content.split() for content in df['content']]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # The resulting 'corpus' and 'dictionary' are ready for LDA modeling

if __name__ == "__main__":
    main()

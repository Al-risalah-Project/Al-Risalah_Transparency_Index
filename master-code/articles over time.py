import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from camel_tools.tokenizers.word import simple_word_tokenize
import nltk
import os

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

def load_data_from_db(query, database_path='processed_articles.db'):
    """Loads article data from SQLite database."""
    if not os.path.isfile(database_path):
        raise FileNotFoundError(f"Database file '{database_path}' not found.")
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def load_data_from_csv(file_path):
    """Loads article data from CSV file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV file '{file_path}' not found.")
    df = pd.read_csv(file_path)
    return df

def prepare_articles_data(df):
    """Processes the articles DataFrame."""
    df['published_dt'] = pd.to_datetime(df['published_dt'], errors='coerce')
    df.dropna(subset=['published_dt'], inplace=True)
    df = df[(df['published_dt'] >= '2000-01-01') & (df['published_dt'] <= '2023-12-31')]
    return df

def plot_articles_count(df, freq='M', title='Number of Articles per Month/Year'):
    """Plots the article counts."""
    df[freq] = df['published_dt'].dt.to_period(freq)
    article_counts = df.groupby(freq).size()

    plt.figure(figsize=(15, 7))
    article_counts.plot(kind='bar', width=0.8)
    plt.title(title)
    plt.xlabel(f'{freq}-Year' if freq == 'Y' else 'Month-Year')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=90 if freq == 'M' else 45, fontsize=8)
    plt.tight_layout()
    plt.show()

def analyze_article_production(df):
    """Analyzes and plots article production over time."""
    df['month_year'] = df['published_dt'].dt.to_period('M')
    monthly_articles = df.groupby('month_year').size()

    mean_articles = monthly_articles.mean()
    std_articles = monthly_articles.std()
    high_threshold = mean_articles + std_articles
    low_threshold = mean_articles - std_articles

    high_production_periods = monthly_articles[monthly_articles > high_threshold]
    low_production_periods = monthly_articles[monthly_articles < low_threshold]

    print("Mean Articles per Month:", mean_articles)
    print("Threshold for High Production:", high_threshold)
    print("Threshold for Low Production:", low_threshold)
    print("\nPeriods of High Production:")
    print(high_production_periods.to_string())
    print("\nPeriods of Low Production:")
    print(low_production_periods.to_string())

    plt.figure(figsize=(15, 7))
    monthly_articles.plot(kind='bar', color='gray', label='Normal')
    high_production_periods.plot(kind='bar', color='green', label='High Production', alpha=0.7)
    low_production_periods.plot(kind='bar', color='red', label='Low Production', alpha=0.7)

    plt.title('Article Production Analysis')
    plt.xlabel('Month-Year')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=90, fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Specify data source (uncomment the appropriate line based on data source)
    # query = "SELECT id, published_dt, clean_content, clean_headline FROM articles"
    # df_articles = load_data_from_db(query, 'processed_articles.db')
    
    # Using CSV file as a data source
    df_articles = load_data_from_csv('processed_articles(tokenized,stopwords,not lemmetised).csv')

    # Prepare data
    df_articles = prepare_articles_data(df_articles)

    # Plot monthly article count
    plot_articles_count(df_articles, freq='M', title='Number of Articles per Month')

    # Analyze article production
    analyze_article_production(df_articles)

    # Plot yearly article count
    plot_articles_count(df_articles, freq='Y', title='Number of Articles per Year')

if __name__ == "__main__":
    main()

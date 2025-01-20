import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Paths to your uploaded files
news_file_path = "financial_news1.csv"  # Replace with the path to your news file
stock_files = {
    "DJI": "DJI 2021_2024_D.csv",
    "Alphabet": "Alphabet 2021_2024_D.csv",
    "Amazon": "Amazon 2021_2024_D.csv",
    "Gold Futures": "Gold_Futures_2021_2024_D.csv",
    "SPY ETF": "SPY_ETF_2021_2024_D.csv",
    "Microsoft": "Microsoft 2021_2024_D.csv",
    "Apple": "Apple 2021_2024_D.csv",
    "Tesla": "Tesla 2021_2024_D.csv",
}

# Step 1: Load and Clean Financial News Data
def preprocess_text(text):
    """Clean text by removing special characters and extra spaces."""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip().lower()

def process_news(news_file):
    """Load and clean financial news data, and add sentiment scores."""
    print("Processing financial news data...")
    news_data = pd.read_csv(news_file)
    analyzer = SentimentIntensityAnalyzer()

    # Handle missing descriptions and clean text
    news_data['description'] = news_data['description'].fillna('')
    news_data['title_cleaned'] = news_data['title'].apply(preprocess_text)
    news_data['description_cleaned'] = news_data['description'].apply(preprocess_text)

    # Add sentiment scores
    news_data['title_sentiment'] = news_data['title_cleaned'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    news_data['description_sentiment'] = news_data['description_cleaned'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    # Save processed news data to a new CSV file
    news_data.to_csv("processed_financial_news.csv", index=False)
    print("News data processed and saved to 'processed_financial_news.csv'.")
    return news_data

# Step 2: Load Stock Data
def load_stock_data(stock_files):
    """Load and return stock data from multiple files."""
    print("Loading stock data...")
    stock_data = {}
    for name, file_path in stock_files.items():
        stock_data[name] = pd.read_csv(file_path)
        print(f"{name} data loaded with {len(stock_data[name])} rows.")
    return stock_data

# Step 3: Merge News and Stock Data
def merge_news_stock(news_data, stock_data):
    """Merge financial news with stock data based on dates."""
    print("Merging news and stock data...")

    # Convert published_at to date
    news_data['published_date'] = pd.to_datetime(news_data['published_at']).dt.date

    # Prepare merged datasets
    merged_data = {}
    for stock_name, data in stock_data.items():
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            merged_data[stock_name] = pd.merge(
                news_data,
                data,
                left_on='published_date',
                right_on='Date',
                how='inner'
            )
            print(f"Merged with {stock_name}: {len(merged_data[stock_name])} rows.")

    return merged_data

# Run All Steps
if __name__ == "__main__":
    # Step 1: Process Financial News
    financial_news = process_news(news_file_path)

    # Step 2: Load Stock Data
    stock_data = load_stock_data(stock_files)

    # Step 3: Merge News and Stock Data
    merged_data = merge_news_stock(financial_news, stock_data)

    # Save each merged dataset
    for stock_name, data in merged_data.items():
        file_name = f"merged_{stock_name}_data.csv"
        data.to_csv(file_name, index=False)
        print(f"Merged data saved to {file_name}.")

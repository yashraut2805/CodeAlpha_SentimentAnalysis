import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")

# Keep only required column
df = df[['reviews.text']].dropna()

# Function for sentiment
def get_sentiment(text):
    analysis = TextBlob(str(text))
    
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df['Sentiment'] = df['reviews.text'].apply(get_sentiment)

# Show results
print(df[['reviews.text', 'Sentiment']].head())

# Count sentiments
sentiment_counts = df['Sentiment'].value_counts()

# Plot graph
plt.figure(figsize=(6,5))
sentiment_counts.plot(kind='bar')

plt.title("Amazon Reviews Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.show()
# Pie chart
plt.figure(figsize=(6,6))

sentiment_counts.plot(
    kind='pie',
    autopct='%1.1f%%'
)

plt.title("Sentiment Distribution")
plt.ylabel("")

plt.show()
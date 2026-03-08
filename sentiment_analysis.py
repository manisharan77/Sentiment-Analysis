# ============================================================
# Sentiment Analysis using NLP
# Author: Mani Sharan Bommakanti
# GitHub: github.com/manisharan77
# Description: End-to-end sentiment analysis pipeline to
#              classify text reviews as Positive, Negative,
#              or Neutral using NLP & Machine Learning.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ─────────────────────────────────────────────
# 1. SAMPLE DATASET
# ─────────────────────────────────────────────
def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    data = {
        'review': [
            # Positive
            "This product is absolutely amazing! I love it so much.",
            "Great quality, fast delivery. Highly recommend to everyone!",
            "Fantastic experience, the customer service was top notch.",
            "Best purchase I have made in years. Works perfectly.",
            "Excellent product, very happy with my purchase.",
            "Outstanding quality and great value for money.",
            "I am thoroughly impressed with this item. Five stars!",
            "Superb product, exceeded my expectations completely.",
            "Really good product. Will definitely buy again.",
            "Wonderful! My whole family loves it.",
            "Incredible quality and very fast shipping. Highly satisfied.",
            "Perfect gift. Arrived on time and looks great.",
            "Very happy with this purchase. Works as described.",
            "Loved every bit of it. Totally worth the price.",
            "Brilliant product. Smooth and easy to use.",

            # Negative
            "Terrible product. Broke after one day of use.",
            "Very disappointed. Does not work as advertised at all.",
            "Worst purchase ever. Total waste of money.",
            "Poor quality, arrived damaged and customer support unhelpful.",
            "Awful experience. Would not recommend to anyone.",
            "Completely useless. Returned it immediately.",
            "Very bad quality. Fell apart within a week.",
            "Not happy at all. Product is nothing like the description.",
            "Horrible. Stopped working after two days.",
            "Do not buy this. It is a complete scam.",
            "Extremely poor build quality. Very fragile.",
            "Defective product. Customer service was no help.",
            "Disappointed with this purchase. Very cheap quality.",
            "Would give zero stars if possible. Absolute garbage.",
            "Broke immediately. Complete waste of money.",

            # Neutral
            "It is okay. Not great but not terrible either.",
            "Average product. Does what it says, nothing more.",
            "Decent quality for the price. Pretty standard.",
            "It works fine. Nothing special about it.",
            "Okay product. Expected a bit more for the price.",
            "Neither good nor bad. Just an average product.",
            "Product is acceptable. Delivery was on time.",
            "It does the job. Not very impressive though.",
            "Fairly ordinary. Met basic expectations only.",
            "Standard product. No complaints but nothing exciting.",
            "Mediocre quality but the price is fair enough.",
            "It arrived on time. Quality is just average.",
            "Works as expected. Nothing extraordinary.",
            "Pretty average overall. Might look for alternatives.",
            "Okay for the price. Would not go out of my way to recommend.",
        ],
        'sentiment': (
            ['Positive'] * 15 +
            ['Negative'] * 15 +
            ['Neutral']  * 15
        )
    }
    return pd.DataFrame(data)


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────
class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Full preprocessing pipeline for a single text string."""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)

    def fit_transform(self, texts):
        return [self.clean_text(t) for t in texts]


# ─────────────────────────────────────────────
# 3. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train multiple classifiers and compare performance."""

    models = {
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf',   LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf',   MultinomialNB())
        ]),
        'Support Vector Machine': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf',   LinearSVC(random_state=42, max_iter=2000))
        ]),
    }

    results = {}
    best_model = None
    best_acc = 0

    print("\n" + "="*60)
    print("        MODEL TRAINING & EVALUATION RESULTS")
    print("="*60)

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'accuracy': acc, 'pipeline': pipeline, 'y_pred': y_pred}

        print(f"\n📊 {name}")
        print(f"   Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"\n{classification_report(y_test, y_pred)}")

        if acc > best_acc:
            best_acc = acc
            best_model = name

    print(f"\n🏆 Best Model: {best_model} with accuracy {best_acc*100:.2f}%")
    return results, best_model


# ─────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────
def visualize_results(df, results, best_model, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sentiment Analysis — Results Dashboard', fontsize=16, fontweight='bold')

    # Plot 1: Sentiment Distribution
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    axes[0, 0].pie(sentiment_counts, labels=sentiment_counts.index,
                   autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 0].set_title('Sentiment Distribution in Dataset')

    # Plot 2: Model Accuracy Comparison
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in model_names]
    short_names = ['Logistic\nRegression', 'Naive\nBayes', 'SVM']
    bars = axes[0, 1].bar(short_names, accuracies, color=['#3498db', '#e67e22', '#9b59b6'], edgecolor='black')
    axes[0, 1].set_title('Model Accuracy Comparison')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_ylim(0, 110)
    for bar, acc in zip(bars, accuracies):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', fontweight='bold')

    # Plot 3: Confusion Matrix for best model
    cm = confusion_matrix(y_test, results[best_model]['y_pred'],
                          labels=['Positive', 'Negative', 'Neutral'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Negative', 'Neutral'],
                yticklabels=['Positive', 'Negative', 'Neutral'], ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix — {best_model}')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xlabel('Predicted')

    # Plot 4: Review Length Distribution by Sentiment
    df['review_length'] = df['review'].apply(lambda x: len(x.split()))
    for sentiment, color in zip(['Positive', 'Negative', 'Neutral'], colors):
        subset = df[df['sentiment'] == sentiment]['review_length']
        axes[1, 1].hist(subset, alpha=0.6, label=sentiment, color=color, bins=10)
    axes[1, 1].set_title('Review Length Distribution by Sentiment')
    axes[1, 1].set_xlabel('Number of Words')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Visualization saved as 'sentiment_analysis_results.png'")
    plt.show()


# ─────────────────────────────────────────────
# 5. PREDICT NEW REVIEWS
# ─────────────────────────────────────────────
def predict_sentiment(pipeline, preprocessor, reviews):
    """Predict sentiment for new custom reviews."""
    print("\n" + "="*60)
    print("        PREDICTING SENTIMENT FOR NEW REVIEWS")
    print("="*60)
    cleaned = preprocessor.fit_transform(reviews)
    predictions = pipeline.predict(cleaned)
    for review, pred in zip(reviews, predictions):
        emoji = "😊" if pred == "Positive" else "😠" if pred == "Negative" else "😐"
        print(f"\n📝 Review  : {review}")
        print(f"   Sentiment: {pred} {emoji}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("   SENTIMENT ANALYSIS USING NLP & MACHINE LEARNING")
    print("   Author: Mani Sharan Bommakanti")
    print("="*60)

    # Load data
    df = create_sample_dataset()
    print(f"\n✅ Dataset loaded: {df.shape[0]} reviews")
    print(df['sentiment'].value_counts().to_string())

    # Preprocess
    preprocessor = TextPreprocessor()
    df['cleaned_review'] = preprocessor.fit_transform(df['review'])
    print("\n✅ Text preprocessing complete")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], df['sentiment'],
        test_size=0.25, random_state=42, stratify=df['sentiment']
    )
    print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")

    # Train & evaluate
    results, best_model = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Visualize
    visualize_results(df, results, best_model, y_test)

    # Predict new reviews
    new_reviews = [
        "This is the best product I have ever bought! Absolutely love it.",
        "Very poor quality. Broke after just one day. Terrible.",
        "It is okay. Nothing special but works fine for the price.",
        "Incredible value for money. Highly recommend to everyone!",
        "Disappointed. Expected much better quality than this.",
    ]
    best_pipeline = results[best_model]['pipeline']
    predict_sentiment(best_pipeline, TextPreprocessor(), new_reviews)

    print("\n✅ Sentiment Analysis Complete!")

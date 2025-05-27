# Research-paper
Detection of fake news using Machine Learning and NLP

#Data set
 Taking Dataset in Google collab.

#Codes for Implementation
# 1. Basic Cleaning and Setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
file_path = "/content/archive (1)mmm.zip"
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully.")
except FileNotFoundError:
    print("File not found at: {file_path}")

# 2. Convert date column
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 3. Missing Value Analysis
plt.figure(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# 4. Label Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label')
plt.title("Fake vs Real News Distribution")
plt.show()

# 5. Category-wise Article Count
plt.figure(figsize=(10, 5))
df['category'].value_counts().plot(kind='bar')
plt.title("Articles per Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 6. Word Clouds for Fake vs Real
for label in ['fake', 'real']:
    text = " ".join(df[df['label'] == label]['text'].astype(str))
    wordcloud = WordCloud(max_words=200, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {label.title()} News")
    plt.show()

# 7. Top Authors by Article Count
top_authors = df['author'].value_counts().head(10)
top_authors.plot(kind='barh')
plt.title("Top 10 Authors by Number of Articles")
plt.xlabel("Number of Articles")
plt.gca().invert_yaxis()
plt.show()

# 8. Text Length Distribution
df['text_len'] = df['text'].astype(str).apply(len)
sns.histplot(df['text_len'], bins=50, kde=True)
plt.title("Distribution of Article Lengths")
plt.xlabel("Text Length")
plt.show()

# 9. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'].astype(str))
y = df['label'].map({'real': 1, 'fake': 0})

# 10. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11. Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
print("Logistic Regression Report:\n", classification_report(y_test, lr_preds))

# 12. Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
print("Naive Bayes Report:\n", classification_report(y_test, nb_preds))

# 13. Decision Tree Model
dt_model = DecisionTreeClassifier(max_depth=10)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
print("Decision Tree Report:\n", classification_report(y_test, dt_preds))

# 14. Compare Accuracy
models = ['Logistic Regression', 'Naive Bayes', 'Decision Tree']
accuracies = [accuracy_score(y_test, lr_preds),
              accuracy_score(y_test, nb_preds),
              accuracy_score(y_test, dt_preds)]
plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim(0, 1)
plt.show()

# 15. Confusion Matrix for Best Model (Choose one)
sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

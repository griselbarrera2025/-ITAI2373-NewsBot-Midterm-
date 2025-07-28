Professional News Bot NLP System 
ITAI2373-NewsBot-Midterm/
│
├── NewsBot_Midterm_Notebook.ipynb         ← Full project notebook
├── newsbot_dataset.csv                    ← Cleaned dataset (max 2K rows)
├── README.md                              ← Overview, instructions, contributions
├── images/                                ← Visualizations, charts, NER maps, etc.
│   └── tfidf_barplot.png
│   └── pos_distribution.png
├── utils/                                 ← Optional: helper functions
│   └── preprocessing.py
├── NewsBot_Reflection_[GRISEL_BARRERA].pdf      ← Final group reflection
└── MT_Video_Group_GRISEL+BARRERA].md  

!pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy textblob wordcloud --quiet
!python -m spacy download en_core_web_sm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Business context - markdown for reporting
from IPython.display import display, Markdown

display(Markdown("""
# Business Case & Application Context
**Objective:** Automatically categorize news articles and extract actionable insights to support media monitoring, editorial decision-making, and trend analysis.  
**Industry:** Media, Business Intelligence  
**Target Users:** Editors, researchers, analysts  
**Value Proposition:** Save time, improve accuracy, and reveal trends from large news datasets.
"""))

# Load data
df = pd.read_csv("BBC News Train.csv")
print("Data Sample:")
display(df.head())
print("Class Distribution:")
df['Category'].value_counts().plot(kind='bar', title='Category Distribution')
plt.show()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    # Lowercase, remove non-letters
    text = str(text).lower()
    text = ''.join([c if c.isalpha() or c.isspace() else ' ' for c in text])
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['Article'].apply(preprocess)
display(df[['Article','clean_text']].head())

tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df['clean_text'])

# Most important words per category
def top_tfidf_feats(row, features, top_n=10):
    d = row.toarray().flatten()
    topn_ids = np.argsort(d)[::-1][:top_n]
    top_feats = [(features[i], d[i]) for i in topn_ids]
    return top_feats

for cat in df['Category'].unique():
    idx = df[df['Category']==cat].index[0]
    feats = top_tfidf_feats(X_tfidf[idx], tfidf.get_feature_names_out(), 10)
    print(f"Top TF-IDF terms for {cat}:")
    print(feats)

# Visualize wordcloud by category
for cat in df['Category'].unique():
    text = " ".join(df[df['Category']==cat]['clean_text'])
    wc = WordCloud(width=800, height=400).generate(text)
    plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f"WordCloud - {cat}")
    plt.axis('off')
    plt.show()

    def pos_counts(text):
    doc = nlp(text)
    pos_freq = {}
    for token in doc:
        pos_freq[token.pos_] = pos_freq.get(token.pos_, 0) + 1
    return pos_freq

df['pos_counts'] = df['clean_text'].apply(pos_counts)

# Aggregate and plot POS distribution per category
pos_df = pd.DataFrame(df['pos_counts'].tolist()).fillna(0)
pos_df['Category'] = df['Category']
pos_means = pos_df.groupby('Category').mean().T
pos_means.plot(kind='bar', figsize=(14,6), title='POS Distribution per Category')
plt.ylabel('Avg Count')
plt.show()

def dependency_features(text):
    doc = nlp(text)
    # Extract subject/object pairs
    subjects = [tok.text for tok in doc if tok.dep_=="nsubj"]
    objects = [tok.text for tok in doc if tok.dep_=="dobj"]
    return {'subjects': subjects, 'objects': objects}

df['dep_features'] = df['clean_text'].apply(dependency_features)
display(df[['clean_text','dep_features']].head())

def dependency_features(text):
    doc = nlp(text)
    # Extract subject/object pairs
    subjects = [tok.text for tok in doc if tok.dep_=="nsubj"]
    objects = [tok.text for tok in doc if tok.dep_=="dobj"]
    return {'subjects': subjects, 'objects': objects}

df['dep_features'] = df['clean_text'].apply(dependency_features)
display(df[['clean_text','dep_features']].head())

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment'] = df['Article'].apply(get_sentiment)

plt.figure(figsize=(10,6))
sns.boxplot(x='Category', y='sentiment', data=df)
plt.title('Sentiment Distribution by Category')
plt.show()

# Prepare features and labels
X = X_tfidf
y = df['Category']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"{name} Accuracy: {acc:.3f}")
    print(classification_report(y_val, preds))
    print(confusion_matrix(y_val, preds))

    def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

df['entities'] = df['Article'].apply(extract_entities)

# Entity frequency
all_entities = [ent[1] for ents in df['entities'] for ent in ents]
entity_freq = pd.Series(all_entities).value_counts()
print(entity_freq)

# Example: Show most common entities per category
from collections import Counter
for cat in df['Category'].unique():
    ents = [ent for ents in df[df['Category']==cat]['entities'] for ent in ents]
    freq = Counter([e[1] for e in ents])
    print(f"Entities in {cat}: {freq}")

    display(Markdown("""
## Key Business Insights

- **Most frequent news topics:** {TOPICS}
- **Sentiment overview:** {SENTIMENT}
- **Entity patterns:** {ENTITY}
- **Model performance:** {MODEL}







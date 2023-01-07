# %%
import nltk
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import sent_tokenize

# %% [markdown]
#### Task 1 - Counting characters, words, and vocabulary

# %%
data = fetch_20newsgroups(subset="train")
df = pd.DataFrame(data.data, columns=["text"])

# %%
df["num_char"] = df.text.str.len()
df["num_words"] = df.text.str.split().str.len()
# %%
df["num_vocab"] = df.text.str.lower().str.split().apply(set).str.len()
# %%
df["lexical_div"] = df["num_words"] / df["num_vocab"]
df["ave_word_length"] = df["num_char"] / df["num_words"]

# %% [markdown]
### Task 2 - Estimating text complexity by counting sentences
# %%
sample_data = df.loc[1:10]
sample_data["text"] = sample_data["text"].str.split("Lines:").apply(lambda x: x[1])
sample_data["num_sents"] = sample_data["text"].apply(sent_tokenize).apply(len)

# %%
df["text"] = df["text"].str.replace("[^\w\s]", "").str.replace("\d+", "")
# %%
vectorizer = CountVectorizer(
    lowercase=True, stop_words="english", ngram_range=(1, 1), min_df=0.05
)
# %%
X = vectorizer.fit_transform(df["text"])
bagofwords = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

# %% [markdown]
### Task 3 - Implementing term frequency-inverse document frequency

# %%
vectorizer = TfidfVectorizer(
    lowercase=True, stop_words="english", ngram_range=(1, 1), min_df=0.05
)

X = vectorizer.fit_transform(df["text"])

tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

# %%

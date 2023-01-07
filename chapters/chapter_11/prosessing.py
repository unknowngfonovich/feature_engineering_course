# %%
import nltk
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# %% [markdown]
#### Task 1 - Counting characters, words, and vocabulary

# %%
data = fetch_20newsgroups(subset='train')
df = pd.DataFrame(data.data, columns=['text'])

# %%
df['num_char'] = df.text.str.len()
df['num_words'] = df.text.str.split().str.len()
# %%
df['num_vocab'] = df.text.str.lower() \
    .str.split() \
    .apply(set) \
    .str.len()
# %%
df['lexical_div'] = df['num_words'] / df['num_vocab']
df['ave_word_length'] = df['num_char'] / df['num_words']
import nltk
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Load Dataset
data = pd.read_csv("covid19_tweets.csv")

def data_source(feature, title, df, size):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()

stopwords = set(STOPWORDS)
def display_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=50,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=1.3)

    plt.imshow(wordcloud)
    plt.show()

# Dataset feature/column info
data.info()

# Heatmap of missing values in different features
missing_graph = sns.heatmap(data.isnull(), cbar=True, yticklabels=False, cmap="GnBu")
plt.show()

data_source("user_location", "Data from different locations", data, 4)

# Term Frequency WordCloud from different locations
bang_df1 = data.loc[data.user_location == "Bengaluru"]
bang_df2 = data.loc[data.user_location == "Bangalore"]
dfs = [bang_df2, bang_df1]
bang_df = pd.concat(dfs, axis=0, join='inner', ignore_index=False, keys=None,
                    levels=None, names=None, verify_integrity=False, copy=True)
display_wordcloud(bang_df['text'], title = 'Most used words in COVID related tweets from Bengaluru')

india_df1 = data.loc[data.user_location=="India"]
india_df2 = data.loc[data.user_location=="New Delhi, India"]
india_df3 = data.loc[data.user_location=="Mumbai, India"]
india_df4 = data.loc[data.user_location=="New Delhi"]
dfs1 = [india_df1, india_df2, india_df3, india_df4]
india_df = pd.concat(dfs1, axis=0, join='inner', ignore_index=False, keys=None,
                    levels=None, names=None, verify_integrity=False, copy=True)
display_wordcloud(india_df['text'], title = 'Most used words in COVID related tweets from India')


usa_df1 = data.loc[data.user_location=="United States"]
usa_df2 = data.loc[data.user_location=="Washington, DC"]
usa_df3 = data.loc[data.user_location=="New York, NY"]
usa_df4 = data.loc[data.user_location=="Los Angeles, CA"]
usa_df5 = data.loc[data.user_location=="USA"]
usa_df6 = data.loc[data.user_location=="California, USA"]
dfs2 = [usa_df1, usa_df2, usa_df3, usa_df4, usa_df5, usa_df6]
usa_df = pd.concat(dfs2, axis=0, join='inner', ignore_index=False, keys=None,
                    levels=None, names=None, verify_integrity=False, copy=True)
display_wordcloud(usa_df['text'], title = 'Most used words in COVID related tweets from United States')


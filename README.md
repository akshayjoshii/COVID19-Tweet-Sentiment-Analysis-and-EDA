**The project is currently in progress, README is not complete yet**

## Abstract:
The [**COVID-19 Tweets**](https://www.kaggle.com/gpreda/covid19-tweets) dataset hosted on Kaggle has **92,276** unique tweets related to the COVID-19 pandemic. Each tweet containes the high-frequency hashtag (#covid19) and are scrapped using Twitter API. The dataset **does not** contain sentiment labels corresponding to each tweet. Thus, supervised learning (ML/DL) methods cannot be used directly for training.
The following tasks are implemented in this project:
1. Perform Exploratory Data Analysis
    * Pre-processing the tweets to perform Normalization, Stop Word Removal, Stemming & Lammetization
    * Plot a wordcloud of most frequent words used in tweets (location-wise).
    * Plot geographical distribution of tweets.
    * Plot frequency of tweets/user and so on.

2. Unsupervised Sentiment Analysis using Density-based Spatial Clustering methods. [In Progress]
    * Projecting the tweets into vector space using pre-trained Word2Vec model.
    * Apply Linear & Manifold Dimentionality Reduction techniques to reduce the predictors from 13 to perhaps 2.
    * Perform DBSCAN clustering to cluster the un-labelled tweets into 4 categories: happy, sad, angry, neutral

3. Explore Transfer Learning with XGBoost (Machine Learning) [In Progress]
    * Train gradient boosted decision trees on a similar but [**labelled**](https://www.kaggle.com/surajkum1198/twitterdata) dataset.
    * Use the trained model for inference on our task's dataset.

4. Explore Transfer Learning with Self-Attention Networks (Deep Learning) [In Progress]
    * Build dataloader to process & consume the dataset into train/test/validation.
    * Train self-attention based transformer network using PyTorch.
    * Use the trained model for inference on our task's dataset.

## Instruction:
1. Clone the repository.
2. Install Python & PIP.
3. Install project dependencies: "pip install -r requirements.txt"
4. Perform Exploratory Data Analysis: "python analysis.py"
#importing necessary libraries
from afinn import Afinn
import pandas as pd

#instantiate afinn
afn = Afinn()
		

def getAfinnSent(df):
    # compute scores (polarity) and labels
    scores = [afn.score(article) for article in df['reply'].tolist()]
    sentiment = ['positive' if score > 0
                            else 'negative' if score < 0
                                else 'neutral'
                                    for score in scores]
        
    # dataframe creation
    df['AfinnSent'] = sentiment
    # print(df)
    return df, sentiment
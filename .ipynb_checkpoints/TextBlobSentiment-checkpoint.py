from textblob import TextBlob

def getTextBobSent(df):
    # compute scores (polarity) 
    scores = [TextBlob(article).sentiment.polarity for article in df['reply'].tolist()]
    sentiment = ['positive' if score > 0
                            else 'neutral' if score == 0
                                else 'negative'
                                    for score in scores]
        
    # dataframe creation
    df['TextBlobSent'] = sentiment
    # print(df)
    return df, sentiment
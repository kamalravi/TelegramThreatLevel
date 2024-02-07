# import SentimentIntensityAnalyzer class
# from vaderSentiment.vaderSentiment module.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# function to print sentiments
# of the sentence.
def sentimentVADER(sentence):

	# Create a SentimentIntensityAnalyzer object.
	sid_obj = SentimentIntensityAnalyzer()

	# polarity_scores method of SentimentIntensityAnalyzer
	# object gives a sentiment dictionary.
	# which contains pos, neg, neu, and compound scores.
	sentiment_dict = sid_obj.polarity_scores(sentence)

	# decide sentiment as positive, negative and neutral
	if sentiment_dict['compound'] >= 0.05 :
		return "positive"

	elif sentiment_dict['compound'] <= - 0.05 :
		return "negative"

	else :
		return "neutral"


def getVADERSentiments(df):
    # print(df.shape)
    preds = []
    for count, row in df.iterrows():
        # print(count, row[0])
        preds.append(sentimentVADER(row['reply']))
        del row, count

    # flat_list = [item for sublist in preds for item in sublist]
    # print(len(preds))
    
    df['VADERSentPreds']=preds

    return df, preds
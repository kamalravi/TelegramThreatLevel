from transformersSentiment import gettfSentiments
import pandas as pd

df=pd.read_json("/home/ravi/PROJECTS_DATA/HarmDetection/allChatsReplies.json", orient='records')

tfSentPredDF=gettfSentiments(df)

tfSentPredDF.to_json("/home/ravi/PROJECTS_DATA/HarmDetection/tfSentPredDF.json", orient='records')
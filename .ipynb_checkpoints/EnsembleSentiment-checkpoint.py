from transformersSentiment import gettfSentiments
from VADERSentiment import getVADERSentiments
from afinnSentiment import getAfinnSent
from TextBlobSentiment import getTextBobSent

import pandas as pd
import glob

# json_files=glob.glob("/home/ravi/PROJECTS_DATA/HarmDetection/allChatsReplies/*.json")

json_files = ["/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/allChatsRepliesFilteredUrlNaN.json"]

for count, file in enumerate(json_files):

    df = pd.read_json(file, orient='records')

    print(count, df.shape, file.split('/')[-1])

    a, df['tfSent']=gettfSentiments(df.copy())
    del a
    a, df['VADERSent']=getVADERSentiments(df.copy())    
    del a
    a, df['AfinnSent']=getAfinnSent(df.copy())
    del a
    a, df['TextBlobSent']=getTextBobSent(df.copy())
    del a

    # df.to_json("/home/ravi/PROJECTS_DATA/HarmDetection/allChatsRepliesEnsemblePreds/"+file.split('/')[-1], orient='records')
df.to_json("/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/allChatsRepliesFilteredUrlNaNEnsemblePreds.json", orient='records')

print(df.shape)

    # break
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

task='sentiment'
# MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
MODEL="/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/tfSentiment/twitter-roberta-base-sentiment/"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def gettfSentiments(df):
    
    preds = []
    for count, batch_df in df.groupby(np.arange(len(df)) // 25):
        # print(count, batch_df.shape)
        encoded_input = tokenizer(batch_df['reply'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
        output = model(**encoded_input)
        scores = output[0].detach().numpy()
        scores = softmax(scores, axis=1)
        max_score_labels = [labels[np.argmax(score)] for score in scores]
        preds.append(max_score_labels)
        del batch_df, encoded_input, output, scores, max_score_labels

    flat_list = [item for sublist in preds for item in sublist]
    
    df['tfSentPreds']=flat_list

    return df, flat_list




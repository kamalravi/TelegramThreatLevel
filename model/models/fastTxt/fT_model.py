# warnings
from warnings import simplefilter
simplefilter("ignore", category=FutureWarning)
simplefilter("ignore", category=DeprecationWarning)

# import libraries
seed=42
import os
os.environ['PYTHONHASHSEED'] = str(seed)
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import time

import time
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

import pickle
import json
import io
import shutil

# preprocess
import pandas as pd
import csv

# train
import fasttext
# prediction

# Functions


import glob
from natsort import natsorted
import pandas as pd

# Functions

def fastText_HoldOut(logger, pred_HoldOut, model_folder, root_dir):
    ## Load trained model and predict
    if pred_HoldOut == 1:
        logger.info("=========================================================")  
        logger.info("================ predict on hold out data ==================")
        logger.info("=========================================================")

        predict_st = time.time()
        logger.info("get input data")

        logger.info("HoldOutData json_files dir: \n {}".format(root_dir+"HoldOutData/*.json"))
        json_files = glob.glob("/home/ravi/HoldOutData/*.json")
        json_files = natsorted(json_files)
        logger.info("HoldOutDatajson_files: \n {}".format(json_files))
        logger.info("HoldOutDatajson_files len: \n {}".format(len(json_files)))

        logger.info("loading trained model")
        model = fasttext.load_model(model_folder+'model.bin')
        
        logger.info("Predicting on the trained LGBMmodel")
        for file in json_files:
            logger.info("===== file: {} =====".format(file.split('/')[-1]))
            df = pd.read_json(file)
            logger.info("df.shape: {}".format(df.shape))

            # y_pred = model.predict(df.article)
            predictions = []
            for tweet in df['text']:
                tweet = ''.join(tweet.replace('\n', ' '))
                yPred=model.predict(tweet)[0][0]
                # print()
                predictions.append(int(yPred.split('_')[-1]))
                # break

            y_pred=np.array(predictions)

            # save y_pred
            logger.info("Saving Predictions along with df in json format")
            test_data_yTrue_yPred = df.copy()
            test_data_yTrue_yPred["y_pred"]=y_pred
            test_data_yTrue_yPred.to_json(root_dir+'/Results/fastText/HoldOutDataResults/'+file.split('/')[-1], orient = 'records')

        logger.info("hold out prediction time {} seconds".format(time.time()-predict_st))


def testfastText(logger, model_predict, model_folder, test_data, test_data_yTrue_yPred):

    if model_predict == 1:
        # Predicting on a single input
        # logger.info("model.predict test_data.iloc[2, 1] {}".format(model.predict(test_data.iloc[2, 1])))
        
        logger.info("=========================================================")  
        logger.info("================Prediction on Test data==================")
        logger.info("=========================================================")
        predict_st = time.time()

        # Save the model
        logger.info("================ load model ==================")
        model = fasttext.load_model(model_folder+'model.bin')

        logger.info("================ Predict ==================")

        predictions = []
        for tweet in test_data['text']:
            tweet = ''.join(tweet.replace('\n', ' '))
            yPred=model.predict(tweet)[0][0]
            # print()
            predictions.append(int(yPred.split('_')[-1]))
            # break

        test_data_yTrue_yPred["y_pred"]=np.array(predictions)

        test_data_yTrue_yPred.to_json(model_folder+'/test_data_yTrue_yPred.json', orient = 'records')

        # metrics
        target_names = ['0', '1', '2']
        classi_report = classification_report(test_data_yTrue_yPred.label, test_data_yTrue_yPred.y_pred, target_names=target_names, digits=4)
        logger.info("classi_report:\n{}".format(classi_report))

        logger.info("confusion_matrix:\n {}".format(confusion_matrix(test_data_yTrue_yPred.label, test_data_yTrue_yPred.y_pred)))
        

        logger.info("Testing f1_weighted score: {}".format(f1_score(test_data_yTrue_yPred.label, test_data_yTrue_yPred.y_pred, average='weighted')))
        
        logger.info("Plot ConfusionMatrix")
        cm = confusion_matrix(test_data_yTrue_yPred.label, test_data_yTrue_yPred.y_pred)
        fig, ax = plt.subplots(figsize=(3,3))
        display_labels=target_names
        SVM_ConfusionMatrix = sns.heatmap(cm, annot=True, xticklabels=display_labels, yticklabels=display_labels, cmap='Blues', ax=ax, fmt='d')
        plt.yticks(va="center")
        plt.xticks(va="center")
        fig.savefig(model_folder+'fasText_ConfusionMatrix.png', format='png', dpi=1200, bbox_inches='tight')
            
        logger.info("train test time {} seconds".format(time.time()-predict_st))


def trainfastText(logger, model_train, model_folder):

    if model_train == 1:
        # and train a model on the all_train_data and save the scores
        # Then test it on the test_data and save the predictions and scores
        logger.info("=========================================================")  
        logger.info("================train on train data==================")
        logger.info("=========================================================")
        
        model_st = time.time()
        # Training the fastText classifier
        model = fasttext.train_supervised(
            model_folder+'train.txt', 
            lr=0.1, 
            epoch=1000, 
            wordNgrams=2, 
            bucket=200000, 
            dim=50, 
            loss='hs')

        # Save the model
        model.save_model(model_folder+'model.bin')

        # Evaluating performance on the entire test file
        logger.info("model.test {}".format(model.test(model_folder+'test.txt')))                      

        logger.info("train time {} seconds".format(time.time()-model_st))
            

def formatData(logger, data_format, model_folder, all_train_data, test_data):

    if data_format == 1:
        logger.info("=========================================================")  
        logger.info("===================== format Data========================")
        logger.info("=========================================================")
        format_st = time.time()

        # Saving the CSV file as a text file to train/test the classifier
        all_train_data[['label', 'text']].to_csv(model_folder+'train.txt', 
                                                index = False, 
                                                sep = ' ',
                                                header = None, 
                                                quoting = csv.QUOTE_NONE, 
                                                quotechar = "", 
                                                escapechar = " ")

        test_data[['label', 'text']].to_csv(model_folder+'test.txt', 
                                            index = False, 
                                            sep = ' ',
                                            header = None, 
                                            quoting = csv.QUOTE_NONE, 
                                            quotechar = "", 
                                            escapechar = " ")
        
        logger.info("data_format time {} seconds".format(time.time()-format_st))
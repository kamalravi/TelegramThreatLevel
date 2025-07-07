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

# train

# prediction

# Functions

def fastText_predict(logger, model_select, model_predict, test_data, model_folder):
    ## Load trained model and predict
    if model_predict == 1:
        logger.info("=========================================================")  
        logger.info("================Prediction on Test data==================")
        logger.info("=========================================================")
        predict_st = time.time()
        
        logger.info("=======loading FineTuned model==========")
        # Tokenize the text and return PyTorch tensors:
        tokenizer = AutoTokenizer.from_pretrained(model_folder)

        if model_select=="OpenAIGPT2": # https://github.com/huggingface/transformers/issues/3859 and https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/ 
            # default to left padding
            tokenizer.padding_side = "left"
            # Define PAD Token = EOS Token = 50256
            tokenizer.pad_token = tokenizer.eos_token
        
        # load model
        model = AutoModelForSequenceClassification.from_pretrained(model_folder)

        if model_select=="OpenAIGPT2": # https://github.com/huggingface/transformers/issues/3859 and https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/ 
            # resize model embedding to match new tokenizer
            model.resize_token_embeddings(len(tokenizer))

            # fix model padding token id
            model.config.pad_token_id = model.config.eos_token_id

        y_pred = []
        for text in test_data['article']:
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
            # Pass your inputs to the model and return the logits:
            with torch.no_grad():
                logits = model(**inputs).logits
            # predict
            predicted_class_id = logits.argmax().item()
            y_pred.append(predicted_class_id)
        
        # save y_pred
        logger.info("Saving Predictions along with test_data_yTrue in json format")
        test_data_yTrue_yPred = test_data.copy()
        test_data_yTrue_yPred['y_pred'] = y_pred
        test_data_yTrue_yPred.to_json(model_folder+"/test_data_yTrue_yPred.json", orient='records')
        
        # metrics
        logger.info("==========metrics===========")
        target_names = ['Liberal', 'Conservative', 'Restricted']
        classi_report = classification_report(test_data.label, y_pred, target_names=target_names, digits=4)
        logger.info("classi_report:\n{}".format(classi_report))
        logger.info("Testing f1_weighted score: {}".format(f1_score(test_data.label, y_pred, average='weighted')))
        logger.info("Plot ConfusionMatrix")
        cm = confusion_matrix(test_data.label, y_pred)
        # fig
        fig, ax = plt.subplots(figsize=(3,3))
        display_labels=target_names
        SVM_ConfusionMatrix = sns.heatmap(cm, annot=True, xticklabels=display_labels, yticklabels=display_labels, cmap='Blues', ax=ax, fmt='d')
        plt.yticks(va="center")
        plt.xticks(va="center")
        fig.savefig(model_folder+'/ConfusionMatrix.png', format='png', dpi=1200, bbox_inches='tight')
        
        logger.info("prediction time {} seconds".format(time.time()-predict_st))


def fastText_train(logger, model_folder):

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

    # Evaluating performance on the entire test file
    logger.info("model.test {}".format(model.test(model_folder+'test.txt')))                      

    # Predicting on a single input
    logger.info("model.predict test_data.iloc[2, 1] {}".format(model.predict(test_data.iloc[2, 1])))

    logger.info("train time {} seconds".format(time.time()-model_st))
    
    logger.info("=========================================================")  
    logger.info("================Prediction on Test data==================")
    logger.info("=========================================================")
    predict_st = time.time()
    predictions = []
    for tweet in test_data['article']:
        yPred=model.predict(tweet)[0][0]
        # print()
        predictions.append(int(yPred.split('_')[-1]))
        # break

    test_data_yTrue_yPred["y_pred"]=np.array(predictions)

    test_data_yTrue_yPred.to_json(model_folder+'/test_data_yTrue_yPred.json', orient = 'records')

    # metrics
    target_names = ['Liberal', 'Conservative', 'Restricted']
    classi_report = classification_report(test_data_yTrue_yPred.label, test_data_yTrue_yPred.y_pred, target_names=target_names, digits=4)
    logger.info("classi_report:\n{}".format(classi_report))
    logger.info("Testing f1_weighted score: {}".format(f1_score(test_data_yTrue_yPred.label, test_data_yTrue_yPred.y_pred, average='weighted')))
    logger.info("Plot ConfusionMatrix")
    cm = confusion_matrix(test_data_yTrue_yPred.label, test_data_yTrue_yPred.y_pred)
    fig, ax = plt.subplots(figsize=(3,3))
    display_labels=['Liberal', 'Conservative', 'Restricted']
    SVM_ConfusionMatrix = sns.heatmap(cm, annot=True, xticklabels=display_labels, yticklabels=display_labels, cmap='Blues', ax=ax, fmt='d')
    plt.yticks(va="center")
    plt.xticks(va="center")
    fig.savefig(model_folder+'fasText_ConfusionMatrix.png', format='png', dpi=1200, bbox_inches='tight')
        
    logger.info("train test time {} seconds".format(time.time()-predict_st))
        
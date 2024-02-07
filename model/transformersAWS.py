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
import pandas as pd

# preprocess
from datasets import Dataset, DatasetDict, concatenate_datasets

# train
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# prediction
import torch

# custom built functions
from logs.get_logs import logger
from dataPrep.get_data_fold import data_read
from models.TFs.Transformers_model import BatchTokenize, BatchTokenizeCombine, Transformers_train, Transformers_predict
from utils.utils import set_seed


# functions go here

def getModelType(model_select):
    
    if model_select=="RoBERTa": # batch size 10
        model_type="roberta-large" # https://huggingface.co/models?sort=downloads&search=roberta
    elif model_select=="Longformer": # batch size 1
        model_type="allenai/longformer-large-4096" # https://huggingface.co/models?sort=downloads&search=longformer-large
    elif model_select=="OpenAIGPT2": # batch size 6
        model_type="gpt2" # https://huggingface.co/models?sort=downloads&search=gpt2
    
    return model_type

# main
if __name__=="__main__":
    
    ## inputs
    # Choose model
    model_select = "OpenAIGPT2" # Options: RoBERTa, Longformer, OpenAIGPT2
    model_type = getModelType(model_select)

    # Choose
    model_tokenize=0
    TokenizeCombine=0
    model_train=0
    model_predict=1
    
    # logger
    task = "_Tokenize_Train_Test" # Train Test
    taskName = model_select + task
    root_dir = '/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/DataModelsResults/'
    model_folder = root_dir + "/Results/" + model_select + "/"
    log_dir_fname = model_folder + taskName +".log"
    print("log_dir_fname: {}".format(log_dir_fname))
    logger = logger(log_dir_fname=log_dir_fname)

    logger.info("=========================================================")  
    logger.info("==================== New execution ======================")
    logger.info("=========================================================")
    execution_st = time.time()

    if TokenizeCombine:
        BatchTokenizeCombine(logger, model_folder)
    elif model_tokenize: 
        # inputs
        logger.info("Get inputs data")
        # Load data. Get K-Fold data. Save 5 fold indices (80% train, 20% test)
        all_train_data = pd.read_json("/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/Annotate/Sample_10000_For_Training.json", orient='records')
        all_train_data['openAI-classification'] = all_train_data['openAI-classification'].astype('int64')
        logger.info("all_train_data.shape {}".format(all_train_data.shape))

        # format
        all_train_data = all_train_data.rename(columns={"openAI-classification": "label", "reply": "article"})

        # Sample the DataFrame with replacement
        n=2000
        val_data = all_train_data.sample(n=n, random_state=42, replace=False)
        # Drop the sampled rows from the DataFrame
        train_data = all_train_data.drop(val_data.index)

        train_data = train_data[['article','label']]
        val_data = val_data[['article','label']]

        train_data = train_data.rename(columns={'article': 'text'})
        val_data = val_data.rename(columns={'article': 'text'})
        
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True)
        
        # Tokenize
        BatchTokenize(logger, model_tokenize, model_type, model_select, model_folder, train_data, val_data)

        # end
        logger.info("Execution time {} seconds".format(time.time()-execution_st))

    elif model_train: 
        # inputs
        logger.info("Get inputs data")
        # Load data. Get K-Fold data. Save 5 fold indices (80% train, 20% test)
        all_train_data = pd.read_json("/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/Annotate/Sample_10000_For_Training.json", orient='records')
        all_train_data['openAI-classification'] = all_train_data['openAI-classification'].astype('int64')
        logger.info("all_train_data.shape {}".format(all_train_data.shape))

        # format
        all_train_data = all_train_data.rename(columns={"openAI-classification": "label", "reply": "article"})
        # Sample the DataFrame with replacement
        n=2000
        val_data = all_train_data.sample(n=n, random_state=42, replace=False)
        # Drop the sampled rows from the DataFrame
        train_data = all_train_data.drop(val_data.index)
        train_data = train_data[['article','label']]
        train_data = train_data.rename(columns={'article': 'text'})
        train_data.reset_index(drop=True, inplace=True)

        # class_weights = [1.7343, 1.5799, 0.5585]

        # # Train
        Transformers_train(logger, model_select, model_train, model_type, model_folder, train_data)

        # end
        logger.info("Execution time {} seconds".format(time.time()-execution_st))

    elif model_predict: 
        # inputs
        logger.info("Get inputs data")

        ## Load data. Get K-Fold data. Save 5 fold indices (80% train, 20% test)
        # all_train_data, train_data, val_data, test_data = data_read(logger, root_dir)
        # del all_train_data,train_data, val_data
        
        test_data = pd.read_json("/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/Annotate/Sample_1326_For_Testing.json", orient='records')
        test_data['openAI-classification'] = test_data['openAI-classification'].astype('int64')
        logger.info("test_data.shape {}".format(test_data.shape))
        test_data = test_data.rename(columns={"openAI-classification": "label", "reply": "article"})
        # test_data = test_data[['article','label']]
        test_data = test_data.rename(columns={'article': 'text'})
        test_data.reset_index(drop=True)
        logger.info("test_data.shape {}".format(test_data.shape))

        # # Prediction
        Transformers_predict(logger, model_select, model_predict, test_data, model_folder)
        
        # end
        logger.info("Execution time {} seconds".format(time.time()-execution_st))
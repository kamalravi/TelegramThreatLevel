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

import torch
# Clear GPU memory
# torch.cuda.empty_cache()

# preprocess
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split

# train
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# prediction

# custom built functions
from logs.get_logs import setup_logger
from dataPrep.get_data_fold import data_read
from models.TFs.TFs_model import BatchTokenize, BatchTokenizeCombine, Transformers_train, Transformers_predict, Transformers_preTrain
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
    ite = 1

    # Choose model
    model_select = "OpenAIGPT2" # Options: RoBERTa, Longformer, OpenAIGPT2
    model_type = getModelType(model_select)

    # if preTrain
    model_preTrain = 0

    # Choose
    model_tokenize=0
    TokenizeCombine=0
    model_train=0
    model_predict=1
    
    # logger
    task = "_Tokenize_Train_Test_"+str(ite) # Train Test
    taskName = model_select + task
    root_dir = '/home/ravi/raviProject/DataModelsResults/'
    model_folder = root_dir + "/Results/" + "FineTune_" + model_select + "_" + str(ite) + "/"
    print("model_folder: {}".format(model_folder))
    log_dir_fname = model_folder + taskName +".log"
    print("log_dir_fname: {}".format(log_dir_fname))
    logger = setup_logger(log_dir_fname=log_dir_fname)

    logger.info("=========================================================")  
    logger.info("==================== New execution ======================")
    logger.info("=========================================================")
    execution_st = time.time()

    if model_preTrain:
        # Call the function
        Transformers_preTrain(
            logger=logger,
            model_select=model_select, 
            model_type=model_type,
            model_folder=model_folder,
            bigData=bigData
        )
    elif TokenizeCombine:
        BatchTokenizeCombine(logger, model_folder)
    elif model_tokenize: 
        # inputs
        logger.info("Get inputs data")
        # Load data. 
        train_df = pd.read_json('/home/ravi/raviProject/DATA/Annotate/iterData/Labeled_10554_train.json', orient='records')
        validate_df=pd.read_json('/home/ravi/raviProject/DATA/Annotate/iterData/Labeled_2261_dev.json', orient='records')        
        logger.info("train_df.shape {}".format(train_df.shape))
        logger.info("validate_df.shape {}".format(validate_df.shape))

        # format
        train_df = train_df.rename(columns={"reply": "text", "Label": "label"})
        validate_df = validate_df.rename(columns={"reply": "text", "Label": "label"})

        train_data = train_df[['text','label']]
        val_data = validate_df[['text','label']]
        
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True)
        
        # Tokenize
        BatchTokenize(logger, model_tokenize, model_type, model_select, model_folder, train_data, val_data)

        # end
        logger.info("Execution time {} seconds".format(time.time()-execution_st))

    elif model_train: 
        # # Train
        Transformers_train(logger, model_select, model_train, model_type, model_folder)

        # end
        logger.info("Execution time {} seconds".format(time.time()-execution_st))

    elif model_predict: 
        # inputs
        logger.info("Get inputs data")

        ## Load data. Get K-Fold data. Save 5 fold indices (80% train, 20% test)
        # all_train_data, train_data, val_data, test_data = data_read(logger, root_dir)
        # del all_train_data,train_data, val_data
        
        fileName = '/home/ravi/raviProject/DATA/Annotate/iterData/Labeled_2261_test.json'
        test_data=pd.read_json(fileName, orient='records')   
        # test_data['openAI-classification'] = test_data['openAI-classification'].astype('int64')
        logger.info("test_data.shape {}".format(test_data.shape))
        
        test_data = test_data.rename(columns={"reply": "text"})

        test_data.reset_index(drop=True)
        logger.info("test_data.shape {}".format(test_data.shape))

        # # Prediction
        Transformers_predict(logger, model_select, model_predict, test_data, model_folder, fileName)
        
        # end
        logger.info("Execution time {} seconds".format(time.time()-execution_st))
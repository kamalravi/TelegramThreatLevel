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
from models.TFs.TFs_model import BatchTokenize, BatchTokenizeCombine, Transformers_train, Transformers_predict
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
    ite = 6

    # Choose model
    model_select = "RoBERTa" # Options: RoBERTa, Longformer, OpenAIGPT2
    model_type = getModelType(model_select)

    # Choose
    model_tokenize=0
    TokenizeCombine=0
    model_train=0
    model_predict=1
    
    # logger
    task = "_Tokenize_Train_Test_"+str(ite) # Train Test
    taskName = model_select + task
    root_dir = '/home/ravi/raviProject/DataModelsResults/'
    model_folder = root_dir + "/Results/" + model_select + "_" + str(ite) + "/"
    log_dir_fname = model_folder + taskName +".log"
    print("log_dir_fname: {}".format(log_dir_fname))
    logger = setup_logger(log_dir_fname=log_dir_fname)

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
        # all_train_data = pd.read_json("/home/ravi/raviProject/DataModelsResults/Data/iter6_Labeled_7605_sampled_forNextIter.json", orient='records')
        all_train_data = pd.read_json("/home/ravi/raviProject/DataModelsResults/Data/iter7_Labeled_15076_sampled_forNextIter_yPred_preTrainFT_RoBERTa_NoisyLabelsRelabeled.json", orient='records')
        # V1_Labeled_300_sampled.json
        # iter2_Labeled_600_sampled.json
        # all_train_data = all_train_data.drop(columns=['label'])
        all_train_data['FinalLabel'] = all_train_data['FinalLabel'].astype('int64')
        logger.info("all_train_data.shape {}".format(all_train_data.shape))

        # format
        all_train_data = all_train_data.rename(columns={"FinalLabel": "label"})

        # Sample the DataFrame with replacement
        # n=2000
        # val_data = all_train_data.sample(frac=0.2, random_state=42, replace=False)
        # Drop the sampled rows from the DataFrame
        # train_data = all_train_data.drop(val_data.index)
        # Perform stratified sampling
        train_data, val_data = train_test_split(
            all_train_data, 
            test_size=0.2, 
            stratify=all_train_data['label'], 
            random_state=42
        )       

        train_data = train_data[['text','label']]
        val_data = val_data[['text','label']]
        
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True)
        
        # Tokenize
        BatchTokenize(logger, model_tokenize, model_type, model_select, model_folder, train_data, val_data)

        # end
        logger.info("Execution time {} seconds".format(time.time()-execution_st))

    elif model_train: 
        # # only for class_weights. we already tokenized our train and val data
        # # inputs
        # logger.info("Get inputs data")
        # # Load data. Get K-Fold data. Save 5 fold indices (80% train, 20% test)
        # all_train_data = pd.read_json("/home/ravi/raviProject/DataModelsResults/Data/V1_Labeled_300_sampled.json", orient='records')
        # all_train_data = all_train_data.drop(columns=['label'])
        # all_train_data['FinalLabel'] = all_train_data['FinalLabel'].astype('int64')
        # logger.info("all_train_data.shape {}".format(all_train_data.shape))
        # # format
        # all_train_data = all_train_data.rename(columns={"FinalLabel": "label"})
        # # Sample the DataFrame with replacement
        # val_data = all_train_data.sample(frac=0.2, random_state=42, replace=False)
        # # Drop the sampled rows from the DataFrame
        # train_data = all_train_data.drop(val_data.index)
        # train_data = train_data[['text','label']]
        # val_data = val_data[['text','label']]
        # train_data.reset_index(drop=True, inplace=True)
        # val_data.reset_index(drop=True)
        # # class_weights = [1.7343, 1.5799, 0.5585]

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
        
        # fileName = "/home/ravi/raviProject/DATA/Annotate/sampled_V7_151110.json"
        fileName = "/home/ravi/raviProject/DATA/Annotate/iterData/iter7_Labeled_15076_sampled_forNextIter.json"
        test_data = pd.read_json(fileName, orient='records')
        # test_data['openAI-classification'] = test_data['openAI-classification'].astype('int64')
        logger.info("test_data.shape {}".format(test_data.shape))
        test_data = test_data.rename(columns={"reply": "text"})

        test_data.reset_index(drop=True)
        logger.info("test_data.shape {}".format(test_data.shape))

        # # Prediction
        Transformers_predict(logger, model_select, model_predict, test_data, model_folder, fileName)
        
        # end
        logger.info("Execution time {} seconds".format(time.time()-execution_st))
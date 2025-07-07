# warnings
from warnings import simplefilter
simplefilter("ignore", category=FutureWarning)
simplefilter("ignore", category=DeprecationWarning)

# custom built functions
from logs.get_logs import setup_logger
from dataPrep.get_data_fold import data_read
from utils.utils import set_seed

# import libraries
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

# NLP Preprocessing

import csv

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

from models.fastTxt.fT_model import trainfastText, testfastText, formatData, fastText_HoldOut

# main
if __name__=="__main__":
    
    ## inputs
    # Choose model
    model_select = "fastText" # Options: fastText, OpenAIGPT2

    # Choose
    data_format= 0
    model_train = 0
    model_predict = 1

    # logger
    task = "_Train_Test" # Train Test
    taskName = model_select + task
    root_dir = '/home/ravi/raviProject/DataModelsResults'
    model_folder = root_dir + "/Results/" + model_select + "/"
    print(model_folder)
    log_dir_fname = model_folder + taskName +".log"
    print("log_dir_fname: {}".format(log_dir_fname))
    logger = setup_logger(log_dir_fname=log_dir_fname)
    
    logger.info("=========================================================")  
    logger.info("==================== New execution ======================")
    logger.info("=========================================================")
    execution_st = time.time()
    
    logger.info("Get inputs data")
    # Load data. 
    train_df = pd.read_json('/home/ravi/raviProject/DATA/Annotate/iterData/Labeled_10554_train.json', orient='records')
    validate_df=pd.read_json('/home/ravi/raviProject/DATA/Annotate/iterData/Labeled_2261_dev.json', orient='records') 
    # Concatenate DataFrames vertically
    all_train_data = pd.concat([train_df, validate_df], ignore_index=True)
    # Shuffle the combined DataFrame
    all_train_data = all_train_data.sample(frac=1).reset_index(drop=True)
    all_train_data['Label'] = all_train_data['Label'].astype('int64')
    
    fileName = '/home/ravi/raviProject/DATA/Annotate/iterData/Labeled_2261_test.json'
    test_data=pd.read_json(fileName, orient='records')   
    test_data['Label'] = test_data['Label'].astype('int64')

    logger.info("all_train_data.shape {}".format(all_train_data.shape))
    logger.info("test_data.shape {}".format(test_data.shape))

    # # For Trial run
    # frac = 0.01
    # # for id-ying the threshold of compute to run models
    # all_train_data = all_train_data.sample(frac=frac, replace=True, random_state=42)
    # test_data = test_data.sample(frac=frac, replace=True, random_state=42)
    # logger.info("all_train_data {}, all_train_data.shape {}".format(frac, all_train_data.shape))
    # logger.info("frac {}, test_data.shape {}".format(frac, test_data.shape))

    all_train_data = all_train_data.rename(columns={"Label": "label", "reply": "text"})
    test_data = test_data.rename(columns={"Label": "label", "reply": "text"})

    all_train_data = all_train_data[["label", "text"]]

    test_data_yTrue_yPred = test_data.copy()

    logger.info("all_train_data.head() {}".format(all_train_data.head()))

    logger.info("================ Formatting data==================")
    # Prefixing each row of the category column with '__label__'
    all_train_data.iloc[:, 0] = all_train_data.iloc[:, 0].apply(lambda x: '__label__' + str(x))
    test_data.iloc[:, 0] = test_data.iloc[:, 0].apply(lambda x: '__label__' + str(x))

    logger.info("all_train_data.head() {}".format(all_train_data.head()))

    formatData(logger, data_format, model_folder, all_train_data, test_data)
    
    trainfastText(logger, model_train, model_folder)

    testfastText(logger, model_predict, model_folder, test_data, test_data_yTrue_yPred)

    
    # end
    logger.info("Execution time {} seconds".format(time.time()-execution_st))

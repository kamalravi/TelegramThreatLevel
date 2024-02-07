# warnings
from warnings import simplefilter
simplefilter("ignore", category=FutureWarning)
simplefilter("ignore", category=DeprecationWarning)

# custom built functions
from logs.get_logs import logger
from dataPrep.get_data_fold import data_read
from models.LSTM.LSTM_model import getData, FineTuneLM, ClassifierTrain, LSTM_predict, LSTM_HoldOut
from utils.utils import set_seed

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

# functions go here

# main
if __name__=="__main__":


    ## inputs
    # Choose model
    model_select = "LSTM" # Options: SVM, RoBERTa, LSTM, Longformer, OpenAIGPT2

    # Choose
    Tokenize = 0
    FineTune = 0
    Classifier = 0
    lstm_predict = 1

    # logger
    frac = 1 # 0.0001 # For Trial run otherwise 1
    task = "_Tokenize_TrainClassifer" # Train Test
    taskName = model_select + task
    root_dir = '/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/DataModelsResults'
    model_folder = root_dir + "/Results/" + model_select + "/"
    log_dir_fname = model_folder + taskName +".log"
    # print("log_dir_fname: {}".format(log_dir_fname))
    logger = logger(log_dir_fname=log_dir_fname)

    logger.info("=========================================================")  
    logger.info("==================== New execution ======================")
    logger.info("=========================================================")
    execution_st = time.time()

    logger.info("root_dir: {} ".format(root_dir))
    
    ### 
    logger.info("==================== Tokenize ======================")
    
    getData(logger, Tokenize, root_dir, frac)
    
    ### 
    logger.info("==================== LSTM Train ======================")

    FineTuneLM(logger, FineTune, model_folder)
    
    ClassifierTrain(logger, Classifier, model_folder)

#     ## Load trained model and predict
    ### 
    logger.info("==================== LSTM Predict ======================")
    LSTM_predict(logger, lstm_predict, root_dir, frac)
    
    # end
    logger.info("main.py Execution time {} seconds".format(time.time()-execution_st))
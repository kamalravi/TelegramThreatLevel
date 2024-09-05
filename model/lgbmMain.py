# warnings
from warnings import simplefilter
simplefilter("ignore", category=FutureWarning)
simplefilter("ignore", category=DeprecationWarning)

# custom built functions
from logs.get_logs import setup_logger
from dataPrep.get_data_fold import data_read
from models.GBM.LGBM_model import LGBM_GridSearchCV, LGBM_train, LGBM_predict, LGBM_HoldOut
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

    # Choose model
    model_select = "LGBM" # Options: "SVM_MODEL", "LSTM_MODEL", "TRANSFORMER_MODEL"

    # SVM inputs
    GridSearch= 1
    lgbm_train = 0
    lgbm_predict = 0
    pred_HoldOut = 0

    # logger
    task = "TrainTest" # GridSearchCV Train Test
    # frac = 0.01
    taskName = model_select + task
    log_dir_fname = "/home/ravi/raviProject/DataModelsResults/Results/GBM/" + taskName +".log"
    logger = setup_logger(log_dir_fname=log_dir_fname)
    logger.info("=========================================================")  
    logger.info("==================== New execution ======================")
    logger.info("=========================================================")
    execution_st = time.time()

    # inputs
    logger.info("Get inputs values")
    root_dir = '/home/ravi/raviProject/DataModelsResults/'

    # Load data. Get K-Fold data. Save 5 fold indices (80% train, 20% test)
    # Load data. 
    train_df = pd.read_json('/home/ravi/raviProject/DATA/Annotate/iterData/Labeled_10554_train.json', orient='records')
    validate_df=pd.read_json('/home/ravi/raviProject/DATA/Annotate/iterData/Labeled_2261_dev.json', orient='records') 
    # Concatenate DataFrames vertically
    all_train_data = pd.concat([train_df, validate_df], ignore_index=True)
    # Shuffle the combined DataFrame
    all_train_data = all_train_data.sample(frac=1).reset_index(drop=True)
    # all_train_data['Label'] = all_train_data['Label'].astype('int64')
    
    fileName = '/home/ravi/raviProject/DATA/Annotate/iterData/Labeled_2261_test.json'
    test_data=pd.read_json(fileName, orient='records')   
    # test_data['Label'] = test_data['Label'].astype('int64')

    # all_train_data, train_data, dev_data, test_data = data_read(logger, root_dir)
    del train_df, validate_df

    # for id-ying the threshold of data to run HYPERPARAm tuning with 375 models
    frac = 1
    all_train_data = all_train_data.sample(frac=frac, replace=True, random_state=42)
    test_data = test_data.sample(frac=frac, replace=True, random_state=42)

    logger.info("frac {}, all_train_data.shape {}".format(frac, all_train_data.shape))
    logger.info("frac {}, test_data.shape {}".format(frac, test_data.shape))

    ### SVM
    logger.info("=========== LGBM ===========")
    # Train one KFold Model: LGBM and save the CV results, and model
    # Our pipeline consists of two phases. First, data will be transformed into vector. Afterwards, it is fed to a LGBM classifier. For the LGBM classifier, we tune the hyper-parameters.
    LGBM_GridSearchCV(logger, GridSearch, all_train_data, root_dir)
    # Analyse the above results by loading gs_pipeline_object.pkl
    # check the jupyter notebook: GridSearchCV_Analyse.ipynb
    # Use the best model params: 

    # and train a model on the all_train_data and save the scores
    # Then test it on the test_data and save the predictions and scores
    # LGBM_train(logger, lgbm_train, all_train_data, test_data, root_dir)

    ## Load trained model and predict
    # LGBM_predict(logger, lgbm_predict, test_data, root_dir)

    ## Load trained model and predict on hold out set
    # LGBM_HoldOut(logger, pred_HoldOut, root_dir)

    # end
    logger.info("main.py Execution time {} seconds".format(time.time()-execution_st))
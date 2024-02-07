# warnings
from warnings import simplefilter
simplefilter("ignore", category=FutureWarning)
simplefilter("ignore", category=DeprecationWarning)

# custom built functions
from logs.get_logs import logger
from dataPrep.get_data_fold import data_read
from models.SVM.SVM_model import SVM_train, SVM_predict, SVM_HoldOut, SVM_predictSample
from utils.utils import set_seed

# import libraries
seed=42
import os
os.environ['PYTHONHASHSEED'] = str(seed)
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import pandas as pd

import time

# functions go here

# main
if __name__=="__main__":

    # Choose model
    model_select = "SVM" # Options: "SVM_MODEL", "LSTM_MODEL", "TRANSFORMER_MODEL"

    # SVM inputs
    svm_train = 0
    svm_predict = 1

    # logger
    task = "Train_Test"
    # frac = 0.01
    taskName = model_select + task
    log_dir_fname = "/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/DataModelsResults/Results/SVM/" + taskName +".log"
    # log_dir_fname = "/home/ravi/PROJECTS_DATA/DataModelsResults/Results/SVM/" + taskName +".log"
    logger = logger(log_dir_fname=log_dir_fname)
    logger.info("=============== New execution ====================")
    execution_st = time.time()

    # inputs
    logger.info("Get inputs values")
    root_dir = '/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/DataModelsResults/'

    logger.info("Get inputs data")
    # Load data. Get K-Fold data. Save 5 fold indices (80% train, 20% test)
    all_train_data = pd.read_json("/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/Annotate/Sample_10000_For_Training.json", orient='records')
    all_train_data['openAI-classification'] = all_train_data['openAI-classification'].astype('int64')
    test_data = pd.read_json("/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/Annotate/Sample_1326_For_Testing.json", orient='records')
    test_data['openAI-classification'] = test_data['openAI-classification'].astype('int64')
    logger.info("all_train_data.shape {}".format(all_train_data.shape))
    logger.info("test_data.shape {}".format(test_data.shape))


    all_train_data = all_train_data.rename(columns={"openAI-classification": "label", "reply": "article"})
    test_data = test_data.rename(columns={"openAI-classification": "label", "reply": "article"})

    all_train_data = all_train_data[["label", "article"]]
    # test_data = test_data[["label", "article"]]
    test_data_yTrue_yPred = test_data.copy()

    logger.info("all_train_data.head() {}".format(all_train_data.head()))

    ## Load data. Get K-Fold data. Save 5 fold indices (80% train, 20% test)
    # all_train_data, test_data = data_read(logger, root_dir)

    # # for id-ying the threshold of data to run HYPERPARAm tuning with 375 models
    # all_train_data = all_train_data.sample(frac=frac, replace=True, random_state=42)
    # test_data = test_data.sample(frac=frac, replace=True, random_state=42)
    # logger.info("frac {}, all_train_data.shape {}".format(frac, all_train_data.shape))
    # logger.info("frac {}, test_data.shape {}".format(frac, test_data.shape))

    ### SVM
    logger.info("=========== SVM ===========")
    ## Train one KFold Model: SVM and save the CV results, and model
    # Our pipeline consists of two phases. First, data will be transformed into vector. Afterwards, it is fed to a SVM classifier. For the SVM classifier, we tune the n_neighbors and weights hyper-parameters.
    
    # SVM_GridSearchCV(logger, GridSearch, KFold, all_train_data, root_dir, ratio)
    
    ## Analyse the above results by loading gs_pipeline_object.pkl
    # check the jupyter notebook: GridSearchCV_Analyse.ipynb
    ## Use the best model params: {'clf__C': 10, 'clf__gamma': 1, 'clf__kernel': 'rbf'}

    # and train a model on the all_train_data and save the scores
    # Then test it on the test_data and save the predictions and scores
    SVM_train(logger, svm_train, all_train_data, test_data, root_dir)

    ## Load trained model and predict
    SVM_predict(logger, svm_predict, test_data, root_dir)

    # # saple predict
    # Sample = pd.read_csv("/home/ravi/advExamples.csv")
    # print(Sample.shape)
    # Sample = pd.DataFrame(Sample)
    # print(Sample.shape)
    # print(Sample)
    # SVM_predictSample(logger, svm_predict, Sample, root_dir)

    # end
    logger.info("main.py Execution time {} seconds".format(time.time()-execution_st))

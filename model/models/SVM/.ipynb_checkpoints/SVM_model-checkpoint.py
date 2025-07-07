# import libraries
import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

import glob
from natsort import natsorted
import pandas as pd

# Functions

def dataTruncationTEXT(texts):
    truncated_texts = []
    for text in texts:
        if len(text.split()) > 4000:
            text = ' '.join(text.split()[:4000])
        truncated_texts.append(text)

    return truncated_texts

def SVM_HoldOut(logger, pred_HoldOut, root_dir):
    ## Load trained model and predict
    if pred_HoldOut == 1:
        logger.info("=========================================================")  
        logger.info("================Prediction on HoldOut data==================")
        logger.info("=========================================================")
        predict_st = time.time()

        logger.info("loading trained SVMmodel")
        loaded_SVMmodel = joblib.load(root_dir+'/Results/SVM/SVMmodel.pkl')
        logger.info("Predicting on the trained SVMmodel")

        logger.info("=======Prediction starts==========")
        json_files = glob.glob(root_dir+"/HoldOutData/*.json")
        json_files = natsorted(json_files)
        # json_files = json_files[75:]
        logger.info("HoldOutDatajson_files: \n {}".format(json_files))

        for file in json_files:
            logger.info("===== file: {} =====".format(file.split('/')[-1]))
            df = pd.read_json(file)
            logger.info("df.shape: {}".format(df.shape))

            y_pred = loaded_SVMmodel.predict(df.article)
            # save y_pred
            logger.info("Saving Predictions along with test_data_yTrue in json format")
            test_data_yTrue_yPred = df.copy()
            test_data_yTrue_yPred["y_pred"]=y_pred
            test_data_yTrue_yPred.to_json(root_dir+'/Results/SVM/HoldOutDataResults/'+file.split('/')[-1], orient = 'records')
            del test_data_yTrue_yPred, df, y_pred, file

        logger.info("HoldOut prediction time {} seconds".format(time.time()-predict_st))

def SVM_predict(logger, svm_predict, test_data, root_dir):
    ## Load trained model and predict
    if svm_predict == 1:
        SVMmodel__predict_st = time.time()
        logger.info("loading trained SVMmodel")
        loaded_SVMmodel = joblib.load(root_dir+'/Results/SVM/SVMmodel.pkl')
        logger.info("Predicting on the trained SVMmodel")
        y_pred = loaded_SVMmodel.predict(test_data.article)
        # save y_pred
        logger.info("Saving Predictions along with test_data_yTrue in json format")
        test_data_yTrue_yPred = test_data.copy()
        test_data_yTrue_yPred["y_pred"]=y_pred
        test_data_yTrue_yPred.to_json(root_dir+'/Results/SVM/test_data_yTrue_yPred.json', orient = 'records')
        # metrics
        target_names = ['Liberal', 'Conservative', 'Restricted']
        classi_report = classification_report(test_data.label, y_pred, target_names=target_names, digits=4)
        logger.info("classi_report:\n{}".format(classi_report))
        if len(target_names)==2:
            tn, fp, fn, tp = confusion_matrix(test_data.label, y_pred).ravel()
            logger.info("tn:{}, fp:{}, fn:{}, tp: {}".format(tn, fp, fn, tp ))
            logger.info("Testing f1_weighted score: {}".format(f1_score(test_data.label, y_pred)))
        logger.info("Testing f1_weighted score: {}".format(f1_score(test_data.label, y_pred, average='weighted')))
        logger.info("Plot ConfusionMatrix")
        cm = confusion_matrix(test_data.label, y_pred, labels=loaded_SVMmodel.classes_)
        fig, ax = plt.subplots(figsize=(3,3))
        display_labels=['Liberal', 'Conservative', 'Restricted']
        SVM_ConfusionMatrix = sns.heatmap(cm, annot=True, xticklabels=display_labels, yticklabels=display_labels, cmap='Blues', ax=ax, fmt='d')
        plt.yticks(va="center")
        plt.xticks(va="center")
        fig.savefig(root_dir+'/Results/SVM/SVM_ConfusionMatrix.png', format='png', dpi=1200, bbox_inches='tight')
        logger.info("SVMmodel prediction time {} seconds".format(time.time()-SVMmodel__predict_st))

def SVM_train(logger, svm_train, all_train_data, test_data, root_dir):
    ## Use the best model params: {'clf__C': 10, 'clf__gamma': 0.1, 'clf__kernel': 'rbf'}
    # and train a model on the all_train_data and save the scores
    # Then test it on the test_data and save the predictions and scores
    if svm_train == 1:
        logger.info("Getting SVM Model")
        SVMmodel_st = time.time()
        SVMmodel = Pipeline([('tfidfvect',
                        TfidfVectorizer(encoding='utf-8', lowercase=False, min_df=5,
                                        ngram_range=(1, 2), stop_words='english',
                                        sublinear_tf=True)),
                        ('clf', SVC(kernel='rbf', C=10, gamma=0.1, class_weight = "balanced"))])
        SVMmodel.fit(all_train_data.article, all_train_data.label)
        logger.info("Training accuracy: {}".format(SVMmodel.score(all_train_data.article, all_train_data.label)))
        logger.info("Testing accuracy: {}".format(SVMmodel.score(test_data.article, test_data.label)))
        # Save trained model with fold number
        joblib.dump(SVMmodel, root_dir+'/Results/SVM/SVMmodel.pkl')
        logger.info("SVMmodel train time {} seconds".format(time.time()-SVMmodel_st))

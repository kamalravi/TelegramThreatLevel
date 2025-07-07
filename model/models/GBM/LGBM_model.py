# import libraries
import time

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from skopt import BayesSearchCV

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import glob
from natsort import natsorted
import pandas as pd

# Functions

def LGBM_HoldOut(logger, pred_HoldOut, root_dir):
    ## Load trained model and predict
    if pred_HoldOut == 1:
        logger.info("=========================================================")  
        logger.info("================ predict on LGBM Model ==================")
        logger.info("=========================================================")

        predict_st = time.time()
        logger.info("get input data")

        logger.info("HoldOutData json_files dir: \n {}".format(root_dir+"HoldOutData/*.json"))
        json_files = glob.glob("/home/ravi/HoldOutData/*.json")
        json_files = natsorted(json_files)
        logger.info("HoldOutDatajson_files: \n {}".format(json_files))
        

        logger.info("loading trained model")
        loaded_LGBMmodel = joblib.load(root_dir+'/Results/GBM/LGBMmodel.pkl')
        
        
        logger.info("Predicting on the trained LGBMmodel")
        for file in json_files:
            logger.info("===== file: {} =====".format(file.split('/')[-1]))
            df = pd.read_json(file)
            logger.info("df.shape: {}".format(df.shape))
            y_pred = loaded_LGBMmodel.predict(df.reply)
            # save y_pred
            logger.info("Saving Predictions along with test_data_yTrue in json format")
            test_data_yTrue_yPred = df.copy()
            test_data_yTrue_yPred["y_pred"]=y_pred
            test_data_yTrue_yPred.to_json(root_dir+'/Results/GBM/HoldOutDataResults/'+file.split('/')[-1], orient = 'records')

        logger.info("hold out prediction time {} seconds".format(time.time()-predict_st))

def LGBM_predict(logger, lgbm_predict, test_data, root_dir):
    ## Load trained model and predict
    if lgbm_predict == 1:
        logger.info("=========================================================")  
        logger.info("================ predict on LGBM Model ==================")
        logger.info("=========================================================")

        LGBMmodel__predict_st = time.time()
        logger.info("loading trained SVMmodel")
        loaded_LGBMmodel = joblib.load(root_dir+'/Results/GBM/LGBMmodel.pkl')
        logger.info("Predicting on the trained LGBMmodel")
        y_pred = loaded_LGBMmodel.predict(test_data.reply)
        # save y_pred
        logger.info("Saving Predictions along with test_data_yTrue in json format")
        test_data_yTrue_yPred = test_data.copy()
        test_data_yTrue_yPred["y_pred"]=y_pred
        test_data_yTrue_yPred.to_json(root_dir+'/Results/GBM/test_data_yTrue_yPred.json', orient = 'records')
        # metrics
        target_names = ['Liberal', 'Conservative', 'Restricted']
        classi_report = classification_report(test_data.Label, y_pred, target_names=target_names, digits=4)
        logger.info("classi_report:\n{}".format(classi_report))
        logger.info("Testing f1_weighted score: {}".format(f1_score(test_data.Label, y_pred, average='weighted')))
        logger.info("Plot ConfusionMatrix")
        cm = confusion_matrix(test_data.Label, y_pred, labels=loaded_LGBMmodel.classes_)
        fig, ax = plt.subplots(figsize=(3,3))
        display_labels=['Liberal', 'Conservative', 'Restricted']
        SVM_ConfusionMatrix = sns.heatmap(cm, annot=True, xticklabels=display_labels, yticklabels=display_labels, cmap='Blues', ax=ax, fmt='d')
        plt.yticks(va="center")
        plt.xticks(va="center")
        fig.savefig(root_dir+'/Results/GBM/LGBM_ConfusionMatrix.png', format='png', dpi=1200, bbox_inches='tight')
        logger.info("LGBMmodel prediction time {} seconds".format(time.time()-LGBMmodel__predict_st))

def LGBM_train(logger, lgbm_train, all_train_data, test_data, root_dir):
    ## Use the best model params: 
    # and train a model on the all_train_data and save the scores
    # Then test it on the test_data and save the predictions and scores
    if lgbm_train == 1:
        logger.info("=========================================================")  
        logger.info("================ Training LGBM Model ====================")
        logger.info("=========================================================")


        LGBMmodel_st = time.time()
        estimator = lgb.LGBMClassifier(
            objective='multiclass',
            random_state=42,
            class_weight="balanced",
            learning_rate=0.1,  # Updated from best_params_
            max_depth=7,        # No change, remains the same
            min_child_samples=1, # No change, remains the same
            min_data_in_leaf=100, # No change, remains the same           
            n_estimators=500,   # Updated from best_params_
            num_leaves=31,      # No change, remains the same
            reg_alpha=0,        # Updated from best_params_
            reg_lambda=0.1,     # Updated from best_params_
            n_jobs=40,
            verbose=1,
            verbose_eval=10
        )

        LGBMmodel = Pipeline([('tfidfvect', 
                        TfidfVectorizer(sublinear_tf=True, min_df=5, encoding='utf-8', 
                            ngram_range=(1, 2), stop_words='english')), 
                        ('clf', estimator)])
        
        logger.info("================ Fit starts ====================")
        LGBMmodel.fit(all_train_data.reply, all_train_data.Label)
        logger.info("================ Fit ends ====================")

        logger.info("Training accuracy: {}".format(LGBMmodel.score(all_train_data.reply, all_train_data.Label)))
        logger.info("Testing accuracy: {}".format(LGBMmodel.score(test_data.reply, test_data.Label)))
        # Save trained model with fold number
        joblib.dump(LGBMmodel, root_dir+'/Results/GBM/LGBMmodel.pkl')
        logger.info("SVMmodel train time {} seconds".format(time.time()-LGBMmodel_st))

def LGBM_GridSearchCV(logger, GridSearch, all_train_data, root_dir):
    ## Train Model 1: LGBM and save the CV results, and model
    # Our pipeline consists of two phases. First, data will be transformed into vector. Afterwards, it is fed to a LGBM classifier. For the LGBM classifier, we tune the hyper-parameters.
    if GridSearch == 1:
        logger.info("=========================================================")  
        logger.info("============ Getting LGBM GridSearchCV ==================")
        logger.info("=========================================================")

        LGBM_GridSearchCV_st = time.time()

        estimator = lgb.LGBMClassifier(
            objective='multiclass',
            random_state=42,
            class_weight="balanced"        )

        n_splits = 5

        # parameters_lgbm = {
        #     'clf__max_depth': [3, 7, 10, 15],
        #     'clf__num_leaves': [31, 63, 127, 255],
        #     'clf__min_data_in_leaf': [20, 100, 500, 1000],
        #     'clf__learning_rate': [0.01, 0.05, 0.1],
        #     'clf__n_estimators': [100, 500, 1000, 2000],
        #     'clf__reg_alpha': [0, 0.1, 0.5, 1],
        #     'clf__reg_lambda': [0, 0.1, 0.5, 1],
        #     'clf__min_child_samples': [1, 5, 10, 20]
        # }

        parameters_lgbm_reduced = {
            'clf__max_depth': [3, 7],
            'clf__num_leaves': [31, 63],
            'clf__min_data_in_leaf': [100, 500],
            'clf__learning_rate': [0.01, 0.1],
            'clf__n_estimators': [100, 500],
            'clf__reg_alpha': [0, 0.1],
            'clf__reg_lambda': [0, 0.1],
            'clf__min_child_samples': [1, 5]
        }

        # pipeline = GridSearchCV(
        #     Pipeline([
        #             ('tfidfvect', TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', 
        #             encoding='utf-8', ngram_range=(1, 2), stop_words='english')),
        #             ('clf', estimator)
        #     ]),
        #     param_grid=parameters_lgbm,
        #     cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        #     scoring='f1_weighted',
        #     n_jobs=10, # 40 maxes out, 48 CPU cores available
        #     verbose=2,
        #     return_train_score=True
        # )

        pipeline = GridSearchCV(
            Pipeline([
                ('tfidfvect', TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', 
                encoding='utf-8', ngram_range=(1, 2), stop_words='english')),
                ('clf', estimator)
            ]),
            param_grid=parameters_lgbm_reduced,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
            scoring='f1_weighted',
            n_jobs=10,  # Adjust based on available resources
            verbose=2,
            return_train_score=True
        )

        # Fit our pipeline
        logger.info('Performing hyper-parameter tuning of LGBM classifiers... ')
        pipeline.fit(all_train_data.reply, all_train_data.Label)
        logger.info("pipeline.best_estimator_ {}".format(pipeline.best_estimator_)) # save, Estimator that was chosen by the search, i.e. estimator which gave highest score
        logger.info("pipeline.best_score_{}".format(pipeline.best_score_)) # Mean cross-validated score of the best_estimator
        logger.info("pipeline.best_params_{}".format(pipeline.best_params_)) # params of the best best_estimator(model)
        logger.info("saving GridSearch object as pickle")
        joblib.dump(pipeline, root_dir+'/Results/GBM/LGBM_GridSearchCV_object.pkl')
        logger.info("LGBM_GridSearchCV time {} seconds".format(time.time()-LGBM_GridSearchCV_st))
# import libraries
import time

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from skopt import BayesSearchCV

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# warnings
from warnings import simplefilter
simplefilter("ignore", category=FutureWarning)
simplefilter("ignore", category=DeprecationWarning)

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

# import libraries
from fastai.text.all import *
from functools import partial
import io

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns

import glob
from natsort import natsorted
import pandas as pd

# Functions

def dataTruncation(data):
    
    data['article'] = data['article'].apply(lambda x: x[:4000])
    
    return data

def getTestData(logger, root_dir, frac):
    
    logger.info("=========================================================")  
    logger.info("==================== getTestData ======================")
    logger.info("=========================================================")
    execution_st = time.time()
    
    test_data = pd.read_json("/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/Annotate/Sample_1326_For_Testing.json", orient='records')
    test_data['openAI-classification'] = test_data['openAI-classification'].astype('int64')
    test_data = test_data.rename(columns={"openAI-classification": "label", "reply": "article"})
    logger.info("test_data.shape {}".format(test_data.shape))

    # test_data = dataTruncation(test_data)
    logger.info("test_data.shape {}".format(test_data.shape))
    # test_data = test_data.sample(frac=frac, replace=True, random_state=42)
    logger.info("frac {}, test_data.shape {}".format(frac, test_data.shape))
    # test_data = test_data[["label", "article"]]
    
    # end
    logger.info("getTestData Execution time {} seconds".format(time.time()-execution_st))
    
    return test_data

def getData(logger, Tokenize, root_dir, frac):

    if Tokenize==1:
        logger.info("=========================================================")  
        logger.info("==================== getData ======================")
        logger.info("=========================================================")
        execution_st = time.time()

        logger.info("Get inputs data")
        # Load data. Get K-Fold data. Save 5 fold indices (80% train, 20% test)
        all_train_data = pd.read_json("/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/Annotate/Sample_10000_For_Training.json", orient='records')
        all_train_data['openAI-classification'] = all_train_data['openAI-classification'].astype('int64')
        logger.info("all_train_data.shape {}".format(all_train_data.shape))
        
        test_data = pd.read_json("/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/Annotate/Sample_1326_For_Testing.json", orient='records')
        test_data['openAI-classification'] = test_data['openAI-classification'].astype('int64')
        logger.info("test_data.shape {}".format(test_data.shape))

        all_train_data = all_train_data.rename(columns={"openAI-classification": "label", "reply": "article"})
        test_data = test_data.rename(columns={"openAI-classification": "label", "reply": "article"})
        
        df=all_train_data.copy()
        val_data = df.sample(2000, random_state=42)
        # Drop the sampled rows from the DataFrame
        train_data = df.drop(val_data.index)
        print(len(train_data))
        print(len(val_data))

        logger.info("==================== train_data ======================")
        train_data = dataTruncation(train_data)
        logger.info("train_data.shape {}".format(train_data.shape))
        train_data = train_data.sample(frac=frac, replace=True, random_state=42)
        logger.info("frac {}, train_data.shape {}".format(frac, train_data.shape))

        logger.info("==================== val_data ======================")
        val_data=dataTruncation(val_data)
        logger.info("val_data.shape {}".format(val_data.shape))
        val_data = val_data.sample(frac=frac, replace=True, random_state=42)
        logger.info("frac {}, val_data.shape {}".format(frac, val_data.shape))

        # logger.info("train_data.head() {}".format(train_data.head()))
        logger.info("==================== format ======================")
        train_data = train_data[["label", "article"]]
        val_data = val_data[["label", "article"]]

        # logger.info("train_data.head() {}".format(train_data.head()))

        train_data["is_valid"]=0
        val_data["is_valid"]=1
        data_df = train_data.append(val_data, ignore_index=True)

        del train_data, val_data

        # end
        logger.info("getData Execution time {} seconds".format(time.time()-execution_st))

        logger.info("=========================================================")  
        logger.info("==================== getDataLoaded ======================")
        logger.info("=========================================================")
        execution_st = time.time()

        logger.info("==================== data_lm ======================")
        data_lm = TextDataLoaders.from_df(
            data_df, 
            path=root_dir+'/Results/LSTM/',
            text_col='article', 
            train_ds='label', 
            valid_col='is_valid', 
            is_lm=True,
            bs=2
        )

        logger.info("len data_lm.train_ds {}".format(len(data_lm.train_ds)))
        logger.info("len data_lm.valid_ds {}".format(len(data_lm.valid_ds)))
        logger.info("len data_lm.train_ds[0][0] {}".format(len(data_lm.train_ds[0][0])))
        logger.info("len data_lm.valid_ds[0][0] {}".format(len(data_lm.valid_ds[0][0])))

        logger.info("==================== save data_lm ======================")
        joblib.dump(data_lm, root_dir+'/Results/LSTM/data_lm.pkl')

        logger.info("==================== data_clas ======================")
        data_clas = TextDataLoaders.from_df(
            data_df, 
            path=root_dir+'/Results/LSTM/',
            text_col='article',                                 
            label_col='label', 
            valid_col='is_valid',
            text_vocab=data_lm.train_ds.vocab, 
            bs=200
        )

        logger.info("len data_clas.train_ds {}".format(len(data_clas.train_ds)))
        logger.info("len data_clas.valid_ds {}".format(len(data_clas.valid_ds)))
        logger.info("len data_clas.train_ds[0][0] {}".format(len(data_clas.train_ds[0][0])))
        logger.info("len data_clas.valid_ds[0][0] {}".format(len(data_clas.valid_ds[0][0])))

        logger.info("==================== save data_clas ======================")
        joblib.dump(data_clas, root_dir+'/Results/LSTM/data_clas.pkl')

        # end
        logger.info("getDataLoaded Execution time {} seconds".format(time.time()-execution_st))
        
def LSTM_HoldOut(logger, pred_HoldOut, root_dir):
    ## Load trained model and predict
    if pred_HoldOut == 1:
        logger.info("=========================================================")  
        logger.info("================ pred_HoldOut on LSTM Model ==================")
        logger.info("=========================================================")

        predict_st = time.time()
        
        # inputs
        logger.info("Get inputs values")                

        logger.info("HoldOutData json_files dir: \n {}".format(root_dir+"HoldOutData/*.json"))
        json_files = glob.glob(root_dir+'/HoldOutData/*.json')
        json_files = natsorted(json_files)
        logger.info("HoldOutDatajson_files: \n {}".format(json_files))
        logger.info("HoldOutDatajson_files len: \n {}".format(len(json_files)))

        logger.info("loading trained model")      
        FTclassifier = load_learner(root_dir+'/Results/LSTM/finetuned_ULMFit_classifer')

        logger.info("Predicting on the trained LSTM model")           
        # GPU prediction
        
        logger.info("Predicting on the trained LGBMmodel")
        for file in json_files:
            logger.info("===== file: {} =====".format(file.split('/')[-1]))
            df = pd.read_json(file)
            df1 = dataTruncation(df.copy())
            logger.info("df.shape: {}".format(df.shape))

            # y_pred = model.predict(df.article)
            # Make predictions on test data
            logger.info("Predicting ...")
            test_dl = FTclassifier.dls.test_dl(df1['article'])
            preds, _ = FTclassifier.get_preds(dl=test_dl)
            Pred_labels = preds.argmax(dim=1).numpy()
            del df1, preds, test_dl

            # save y_pred
            logger.info("Saving Predictions along with df in json format")
            test_data_yTrue_yPred = df.copy()
            test_data_yTrue_yPred["y_pred"]=Pred_labels
            test_data_yTrue_yPred.to_json(root_dir+'/Results/LSTM/HoldOutDataResults/'+file.split('/')[-1], orient = 'records')
            del test_data_yTrue_yPred, df, Pred_labels

        logger.info("hold out prediction time {} seconds".format(time.time()-predict_st))
        
        
def LSTM_predict(logger, lstm_predict, root_dir, frac):
    ## Load trained model and predict
    if lstm_predict == 1:

        # inputs
        logger.info("Get inputs values")                
        test_data = getTestData(logger, root_dir, frac)
        
        logger.info("=========================================================")  
        logger.info("================ predict on LSTM Model ==================")
        logger.info("=========================================================")

        predict_st = time.time()
        
        logger.info("loading trained model")      
        FTclassifier = load_learner(root_dir+'/Results/LSTM/finetuned_ULMFit_classifer')

        logger.info("Predicting on the trained LSTM model")           
        # GPU prediction
        
        # Make predictions on test data
        test_dl = FTclassifier.dls.test_dl(test_data['article'])
        preds, _ = FTclassifier.get_preds(dl=test_dl)
        Pred_labels = preds.argmax(dim=1).numpy()
        
#         # Add the predictions to the DataFrame
        test_data_yTrue_yPred = test_data.copy()
        del test_data
        test_data_yTrue_yPred["y_pred"] = Pred_labels
            
        # save y_pred
        logger.info("Saving Predictions along with test_data_yTrue in json format")

        test_data_yTrue_yPred.to_json(root_dir+'/Results/LSTM/test_data_yTrue_yPred.json', orient = 'records')

        # metrics
        target_names = ['0', '1', '2', '3', '4', '5']
        classi_report = classification_report(test_data_yTrue_yPred.label, test_data_yTrue_yPred.y_pred, target_names=target_names, digits=4)
        logger.info("classi_report:\n{}".format(classi_report))
        logger.info("Testing f1_weighted score: {}".format(f1_score(test_data_yTrue_yPred.label, test_data_yTrue_yPred.y_pred, average='weighted')))
        logger.info("Plot ConfusionMatrix")
        cm = confusion_matrix(test_data_yTrue_yPred.label, test_data_yTrue_yPred.y_pred)
        fig, ax = plt.subplots(figsize=(3,3))
        display_labels=target_names
        SVM_ConfusionMatrix = sns.heatmap(cm, annot=True, xticklabels=display_labels, yticklabels=display_labels, cmap='Blues', ax=ax, fmt='d')
        plt.yticks(va="center")
        plt.xticks(va="center")
        fig.savefig(root_dir+'/Results/LSTM/ConfusionMatrix.png', format='png', dpi=1200, bbox_inches='tight')
        logger.info("LSTM model prediction time {} seconds".format(time.time()-predict_st))

        
def FineTuneLM(logger, FineTune, model_folder):
    ## Use the best model params: 
    # and train a model on the all_train_data and save the scores
    # Then test it on the test_data and save the predictions and scores
    if FineTune == 1:

        # inputs
        logger.info("Get inputs values")    
        
        logger.info("==================== load data_lm ======================")
        data_lm = joblib.load(model_folder+'data_lm.pkl')
        
        logger.info("=========================================================")  
        logger.info("========= Finetune a pretrained Language Model ==========")
        logger.info("=========================================================")
        model_st=time.time()
        
        # compute weighted loss
        # # logger.info("======== compute weighted loss =========")
        # # class_weights = compute_class_weight(
        # #                                         class_weight = "balanced",
        # #                                         classes = np.unique(train_data['label']),
        # #                                         y = data_df['label']                                                    
        # #                                     )
        # class_weights= [1.7429, 1.6902, 0.5451]
        # class_weights=torch.tensor(class_weights, dtype=torch.float32)
        # # unsqueeze the tensor along the first dimension to create a tensor of shape 1x3
        # class_weights = class_weights.unsqueeze(0)
        # # class_weights = class_weights.cuda()
        # logger.info("class_weights is \n {}".format(class_weights))

        # loss_fun = CrossEntropyLossFlat(
        #     ignore_index=255,
        #     weight=class_weights, 
        #     reduction='mean') # why choose flat ? https://docs.fast.ai/losses.html


        opt_func = partial(Adam, wd=0.1)
        learner = language_model_learner(dls=data_lm, 
                                         arch=AWD_LSTM, 
                                         opt_func=opt_func,
                                         # loss_func=loss_fun,
                                         metrics=[accuracy, Precision(average='weighted'), Recall(average='weighted'), F1Score(average='weighted')], 
                                         path=model_folder
                                        )
        learner = learner.to_fp16()
        # learner.loss_func=loss_fun

#         # Define early stopping callback
#         early_stop_cb = EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=5)

#         # Define learning rate scheduler
#         lr_sched_cb = ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, factor=0.5, min_lr=1e-6, patience=2)
        
        learner.lr_find()

        learner.fit_one_cycle(1, 4e-3, moms=(0.8,0.7,0.8))

        learner.save(model_folder+'stage1')

        learner.load(model_folder+'stage1')

        learner.unfreeze()
        learner.fit_one_cycle(15, 4e-3, moms=(0.8,0.7,0.8))
        # learner.fit_one_cycle(15, 4e-3, moms=(0.8,0.7,0.8), cbs=[early_stop_cb, lr_sched_cb])
        
        learner.save_encoder(model_folder+'finetuned_ULMFit_LM')
    
        del learner
        
        logger.info("Finetune LM time {} seconds".format(time.time()-model_st))
        
def ClassifierTrain(logger, Classifier, model_folder):        
       
    if Classifier == 1:      
        
        # inputs
        logger.info("Get inputs values")        
        
        
        logger.info("==================== load data_clas ======================")
        data_clas = joblib.load(model_folder+'data_clas.pkl')
        
        logger.info("=========================================================")  
        logger.info("======== Use Finetuned LM to train a classifier =========")
        logger.info("=========================================================")
        model_st=time.time()
        
        opt_func = partial(Adam, wd=0.1)
        classifier = text_classifier_learner(dls=data_clas, 
                                         arch=AWD_LSTM, 
                                         opt_func=opt_func,
                                         # loss_func=loss_fun,
                                         metrics=[accuracy, Precision(average='weighted'), Recall(average='weighted'), F1Score(average='weighted')], 
                                         path=model_folder,
                                         drop_mult=0.5
                                        )
        # classifier.loss_func=loss_fun

        classifier = classifier.load_encoder(model_folder+'finetuned_ULMFit_LM')
        classifier = classifier.to_fp16()

#         # Define early stopping callback
#         early_stop_cb = EarlyStoppingCallback(monitor='accuracy', min_delta=0.01, patience=5)

#         # Define learning rate scheduler
#         lr_sched_cb = ReduceLROnPlateau(monitor='accuracy', min_delta=0.1, factor=0.5, min_lr=1e-6, patience=2)
        
        lr = 1e-1 * 10/128
        classifier.fit_one_cycle(1, lr, moms=(0.8,0.7,0.8), wd=0.1)

        classifier.freeze_to(-2)
        lr /= 2
        classifier.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7,0.8), wd=0.1)

        classifier.freeze_to(-3)
        lr /= 2
        classifier.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7,0.8), wd=0.1)

        classifier.unfreeze()
        lr /= 5
        logger.info("50 epoch")  
        classifier.fit_one_cycle(50, slice(lr/(2.6**4),lr), moms=(0.8,0.7,0.8), wd=0.1)
        # classifier.fit_one_cycle(50, slice(lr/(2.6**4),lr), moms=(0.8,0.7,0.8), wd=0.1, cbs=[early_stop_cb, lr_sched_cb])
        # preds,y,losses = classifier.get_preds(with_loss=True)
        # interp = ClassificationInterpretation.from_learner(classifier) 
        # interp.plot_confusion_matrix()

        # interp1 = Interpretation.from_learner(classifier)
        # interp1.plot_top_losses(2)

        classifier.export(model_folder+'finetuned_ULMFit_classifer')

        del classifier
        
        logger.info("Classifier train time {} seconds".format(time.time()-model_st))

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

import time
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

import pickle
import boto3
import json
import io

# preprocess
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

# train
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# prediction
import torch

# Functions

def Transformers_predict(logger, model_predict, test_data, model_folder):
    ## Load trained model and predict
    if model_predict == 1:
        logger.info("=========================================================")  
        logger.info("================Prediction on Test data==================")
        logger.info("=========================================================")
        predict_st = time.time()
        
        logger.info("=======loading FineTuned model==========")
        # Tokenize the text and return PyTorch tensors:
        tokenizer = AutoTokenizer.from_pretrained(model_folder)
        # load model
        model = AutoModelForSequenceClassification.from_pretrained(model_folder)
        
        y_pred = []
        for text in test_data['article']:
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
            # Pass your inputs to the model and return the logits:
            with torch.no_grad():
                logits = model(**inputs).logits
            # predict
            predicted_class_id = logits.argmax().item()
            y_pred.append(predicted_class_id)
        
        # save y_pred
        logger.info("Saving Predictions along with test_data_yTrue in json format")
        test_data_yTrue_yPred = test_data.copy()
        test_data_yTrue_yPred['y_pred'] = y_pred
        test_data_yTrue_yPred.to_json(model_folder+"/test_data_yTrue_yPred.json", orient='records')
        
        # metrics
        logger.info("==========metrics===========")
        target_names = ['Liberal', 'Conservative', 'Restricted']
        classi_report = classification_report(test_data.label, y_pred, target_names=target_names, digits=4)
        logger.info("classi_report:\n{}".format(classi_report))
        logger.info("Testing f1_weighted score: {}".format(f1_score(test_data.label, y_pred, average='weighted')))
        logger.info("Plot ConfusionMatrix")
        cm = confusion_matrix(test_data.label, y_pred)
        # fig
        fig, ax = plt.subplots(figsize=(3,3))
        display_labels=target_names
        SVM_ConfusionMatrix = sns.heatmap(cm, annot=True, xticklabels=display_labels, yticklabels=display_labels, cmap='Blues', ax=ax, fmt='d')
        plt.yticks(va="center")
        plt.xticks(va="center")
        fig.savefig(model_folder+'/ConfusionMatrix.png', format='png', dpi=1200, bbox_inches='tight')
        
        logger.info("prediction time {} seconds".format(time.time()-predict_st))
        

def Transformers_train(logger, model_train, model_type, AllTrainData, model_folder):

    # and train a model on the all_train_data and save the scores
    # Then test it on the test_data and save the predictions and scores
    if model_train == 1:
        model_st = time.time()
        logger.info("=========================================================")  
        logger.info("================Finetuning on Train data=================")
        logger.info("=========================================================")
        
        logger.info("=======Tokenize train data==========")
        
        # The next step is to load a DistilBERT tokenizer to preprocess the text field:
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        # function to tokenize text and truncate seq to be no longer than maximum input length:
        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)
        # To apply the preprocessing function over the entire dataset, use Datasets map function. You can speed up map by setting batched=True to process multiple elements of the dataset at once:
        tokenized_AllTrainData = AllTrainData.map(preprocess_function, batched=True)
        logger.info("tokenized_AllTrainData is \n {}".format(tokenized_AllTrainData))
        # Now create a batch of examples using DataCollatorWithPadding. It’s more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        logger.info("======== train =========")

        # compute metric
        weighted_f1_metric = evaluate.load("f1")
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return weighted_f1_metric.compute(predictions=predictions, references=labels, average="weighted")

        # Before you start training your model, create a map of the expected ids to their labels with id2label and label2id:
        id2label = {0: "LIBERAL", 1: "CONSERVATIVE", 2: "RESTRICTED"}
        logger.info("id2label is \n {}".format(id2label))
        label2id = {"LIBERAL": 0, "CONSERVATIVE": 1, "RESTRICTED": 2}

        # train

        logger.info("========Training Model=========")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_type, num_labels=3, id2label=id2label, label2id=label2id
        )

        training_args = TrainingArguments(
            output_dir=model_folder,
            seed=seed,
            learning_rate=2e-5,
            per_device_train_batch_size=5,
            per_device_eval_batch_size=5,
            num_train_epochs=1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            # save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_AllTrainData["train"],
            eval_dataset=tokenized_AllTrainData["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )


        trainer.train()
        logger.info("========Saving Model=========")
        trainer.save_model()
        
        logger.info("train time {} seconds".format(time.time()-model_st))
        
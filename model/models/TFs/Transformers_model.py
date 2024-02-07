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
import json
import io
import shutil

# preprocess
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

# preprocess
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
import pandas as pd

# train
import evaluate
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# prediction
import torch
import glob
from natsort import natsorted

# Functions

def dataTruncationTEXT(texts):
    truncated_texts = []
    for text in texts:
        if len(text.split()) > 4000:
            text = ' '.join(text.split()[:4000])
        truncated_texts.append(text)

    return truncated_texts

def Transformers_predict(logger, model_select, model_predict, test_data, model_folder):
    ## Load trained model and predict
    if model_predict == 1:
        logger.info("=========================================================")  
        logger.info("================Prediction on Test data==================")
        logger.info("=========================================================")
        predict_st = time.time()
        
        logger.info("=======loading FineTuned model==========")
        # Tokenize the text and return PyTorch tensors:
        tokenizer = AutoTokenizer.from_pretrained(model_folder)

        if model_select=="OpenAIGPT2": # https://github.com/huggingface/transformers/issues/3859 and https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/ 
            # default to left padding
            tokenizer.padding_side = "left"
            # Define PAD Token = EOS Token = 50256
            tokenizer.pad_token = tokenizer.eos_token

        # load model
        model = AutoModelForSequenceClassification.from_pretrained(model_folder)

        if model_select=="OpenAIGPT2": # https://github.com/huggingface/transformers/issues/3859 and https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/ 
            # resize model embedding to match new tokenizer
            model.resize_token_embeddings(len(tokenizer))

            # fix model padding token id
            model.config.pad_token_id = model.config.eos_token_id
            
        # Define a function to tokenize the text and prepare inputs for the model
        def preprocess_text(article):
            encoding = tokenizer(article, truncation=True, padding=True, return_tensors="pt")
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        logger.info("=======Prediction starts==========")
        # GPU prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        y_pred=[]
        for count, chunk in enumerate(np.array_split(test_data, 1000)):
            print(count, chunk.shape)            
            inputs = preprocess_text(chunk['text'].values.tolist())
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Make predictions on the entire DataFrame
            with torch.no_grad():
                logits = model(**inputs).logits
                # Move logits to CPU and convert to numpy array
                y_pred.append(torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy().tolist())

        y_pred = [item for items in y_pred for item in items]
        # Add the predictions to the DataFrame
        test_data_yTrue_yPred = test_data.copy()
        test_data_yTrue_yPred["y_pred"] = y_pred
        
        # save y_pred
        logger.info("Saving Predictions along with test_data_yTrue in json format")
        test_data_yTrue_yPred = test_data.copy()
        test_data_yTrue_yPred['y_pred'] = y_pred
        test_data_yTrue_yPred.to_json(model_folder+"/test_data_yTrue_yPred.json", orient='records')
        
        # metrics
        logger.info("==========metrics===========")
        target_names = ['0', '1', '2', '3', '4', '5']
        classi_report = classification_report(test_data.label, y_pred, target_names=target_names, digits=4)
        logger.info("classi_report:\n{}".format(classi_report))

        logger.info("confusion_matrix:\n {}".format(confusion_matrix(test_data.label, y_pred)))

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


def dataTruncation(data):
    data['totalwords'] = data['text'].str.split().str.len()
    data_list = []
    for index, row in data.iterrows():
        if row["totalwords"] >= 4000:
            # print(row['text'])
            row['text'] = ' '.join(row['text'].split()[:4000])
            # print(row['text'])
            # break
        data_list.append([row['text'], row['label']])    
    data=pd.DataFrame(data_list, columns=['text','label'])
    
    return data

def BatchTokenize(logger, model_tokenize, model_type, model_select, model_folder, train_data, val_data):

    if model_tokenize==1:
        logger.info("=========================================================")  
        logger.info("===============Tokenize train and val data================")
        logger.info("=========================================================")
        tok_st = time.time()

        # The next step is to load a DistilBERT tokenizer to preprocess the text field:
        tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=True)
        if model_select=="OpenAIGPT2": # https://github.com/huggingface/transformers/issues/3859 and https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/ 
            # default to left padding
            tokenizer.padding_side = "left"
            # Define PAD Token = EOS Token = 50256
            tokenizer.pad_token = tokenizer.eos_token
        # function to tokenize text and truncate seq to be no longer than maximum input length:
        def preprocess_function(batch):
            tokenized_input = tokenizer(batch["text"], truncation=True)
            return tokenized_input
        def tokenize_dataset(AllTrainData):
            tokenized_dataset = AllTrainData.map(preprocess_function, num_proc=1, batched=True, batch_size=1)
            # tokenized_dataset.set_format('numpy', columns=['input_ids', 'attention_mask', 'labels']) # ['text', 'label', 'input_ids', 'attention_mask']
            return tokenized_dataset

        # format
        trainhalves = np.array_split(train_data, 10)
        valhalves = np.array_split(val_data, 10)

        # trainhalves =trainhalves[751:]
        # valhalves =valhalves[751:]

        for idx, (tr, va) in enumerate(zip(trainhalves, valhalves)):
            print(idx)
            # if idx>750:
            # print(751+idx)
            # tr = dataTruncation(tr) # on it data is too large to tokenize
            # va = dataTruncation(va) # on it data is too large to tokenize
            # tr.to_json("tr751.json") # to check
            # va.to_json("va751.json") # to check
            temp = DatasetDict(
                {'train': Dataset.from_dict(tr), 
            'val': Dataset.from_dict(va)})
            tokenized_temp = tokenize_dataset(temp)
            tokenized_temp.save_to_disk(model_folder+'/tokenized/'+str(idx))
            # tokenized_temp.save_to_disk(model_folder+'/tokenized/'+str(751+idx))
            del tokenized_temp
            del idx, tr, va
            time.sleep(1)

        logger.info("Tokenize n time {} seconds".format(time.time()-tok_st))
        time.sleep(30)


def BatchTokenizeCombine(logger, model_folder):
        logger.info("=========================================================") 
        logger.info("========= Combine Tokenize train and val data ==========")
        logger.info("=========================================================") 
        tokag_st = time.time()

        for idxx, nn in enumerate(range(10)):
            print(idxx)
            if nn == 0:
                tokenized_Data = load_from_disk(model_folder+'/tokenized/'+str(nn))
                # tokenized_Data.save_to_disk(model_folder+'/tokenized_AllTrainData')
                # del tokenized_Data
            else:
                # tokenized_Data= load_from_disk(model_folder+'/tokenized_AllTrainData')
                # shutil.rmtree(model_folder+'/tokenized_AllTrainData')
                tokenized_temp = load_from_disk(model_folder+'/tokenized/'+str(nn))

                dataset_tr = concatenate_datasets([tokenized_Data['train'], tokenized_temp['train']])
                dataset_val = concatenate_datasets([tokenized_Data['val'], tokenized_temp['val']])
                del tokenized_temp
                del tokenized_Data

                tokenized_Data = DatasetDict(
                    {'train': dataset_tr, 
                    'val': dataset_val}
                    )
                del dataset_tr
                del dataset_val
                # tokenized_Data.save_to_disk(model_folder+'/tokenized_AllTrainData')
                # del tokenized_Data
            time.sleep(2)

        tokenized_Data.save_to_disk(model_folder+'/tokenized_AllTrainData')
        logger.info("tokenized_AllTrainData is \n {}".format(tokenized_Data))

        logger.info("Combine Tokens n time {} seconds".format(time.time()-tokag_st))


def Transformers_train(logger,  model_select, model_train, model_type, model_folder, AllTrainData):

    # and train a model on the all_train_data and save the scores
    # Then test it on the test_data and save the predictions and scores
    if model_train == 1:
        model_st = time.time()
        yLabel = AllTrainData['label']
        # del AllTrainData # to clear memory

        logger.info("=========================================================")  
        logger.info("================Finetuning on Train data=================")
        logger.info("=========================================================")
        
        logger.info("=======load Tokenize train  and val data==========")
        
        tokenized_AllTrainData = load_from_disk(model_folder+'/tokenized_AllTrainData')
        # tokenized_AllTrainData = load_from_disk(model_folder+'/gpt2-large-tokenized_AllTrainData') # model OOM

        logger.info("tokenized_AllTrainData is \n {}".format(tokenized_AllTrainData))

        # Now create a batch of examples using DataCollatorWithPadding. It’s more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.
        # The next step is to load a DistilBERT tokenizer to preprocess the text field:
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        
        if model_select=="OpenAIGPT2": # https://github.com/huggingface/transformers/issues/3859 and https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/ 
            # default to left padding
            tokenizer.padding_side = "left"
            # Define PAD Token = EOS Token = 50256
            tokenizer.pad_token = tokenizer.eos_token
            
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        logger.info("======== train =========")

        # compute metric
        weighted_f1_metric = evaluate.load("f1")
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return weighted_f1_metric.compute(predictions=predictions, references=labels, average="weighted")

        # Before you start training your model, create a map of the expected ids to their labels with id2label and label2id:
        id2label = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
        logger.info("id2label is \n {}".format(id2label))
        label2id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}

        # compute weighted loss
        logger.info("======== compute weighted loss =========")
        class_weights = compute_class_weight(
                                                class_weight = "balanced",
                                                classes = np.unique(yLabel),
                                                y = yLabel                                                    
                                            )
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        class_weights = class_weights.cuda()
        logger.info("class_weights is \n {}".format(class_weights))
        # logger.info("class_weights.is_cuda is \n {}".format(class_weights.is_cuda))
            
        # train

        logger.info("========Training Model=========")

        # from epoch 0
        # multi class and single label; not problem_type="multi_label_classification"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_type, num_labels=6, id2label=id2label, label2id=label2id
        )
        
        # # path to the model checkpoint from the 36th epoch
        # model_checkpoint = "/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/DataModelsResults/Results/OpenAIGPT2/checkpoint-288000/"
        # # Load the model from the checkpoint
        # model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

        if model_select=="OpenAIGPT2": # https://github.com/huggingface/transformers/issues/3859 and https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/ 
            # resize model embedding to match new tokenizer
            model.resize_token_embeddings(len(tokenizer))

            # fix model padding token id
            model.config.pad_token_id = model.config.eos_token_id

        batch_size = 1
        training_args = TrainingArguments(
            output_dir=model_folder,
            seed=seed,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size, # to avoid OOM
            gradient_accumulation_steps=1, # to avoid OOM
            per_device_eval_batch_size=batch_size, # to avoid OOM
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            save_total_limit=2,
            save_steps=1000,
            eval_steps=1000,
            fp16=True, # to avoid OOM
        )

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                # forward pass
                outputs = model(**inputs)
                logits = outputs.get("logits")
                # compute custom loss (suppose one has 3 labels with different weights)
                loss_fct = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer = CustomTrainer(
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
        
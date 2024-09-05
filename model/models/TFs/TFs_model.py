import warnings

# Suppress FutureWarning and DeprecationWarning
from warnings import simplefilter
simplefilter("ignore", category=FutureWarning)
simplefilter("ignore", category=DeprecationWarning)

# Suppress all other warnings
warnings.filterwarnings("ignore", category=Warning)

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

import torch
# Clear GPU memory
# torch.cuda.empty_cache()

#preTrain
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, DataCollatorForLanguageModeling

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

import glob
from natsort import natsorted

# Functions



def Transformers_preTrain(model_select, model_type, preTrain_dataset, model_folder, mlm=True):

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    # Determine if MLM or Causal Language Modeling
    if model_select=="RoBERTa": # MLM-based models (RoBERTa, Longformer)
        model = AutoModelForMaskedLM.from_pretrained(model_type)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    elif model_select=="OpenAIGPT2": # Causal Language Modeling (GPT-2)
        model = AutoModelForCausalLM.from_pretrained(model_type, num_labels=3)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Save the model and tokenizer to the specified folder
    model.save_pretrained(f"{model_folder}/{model_type}")
    tokenizer.save_pretrained(f"{model_folder}/{model_type}")
    
    print(f"Model {model_type} saved in {model_folder}/{model_type}/")

    # Check if 'text' column is present
    if 'text' not in preTrain_dataset.column_names:
        raise ValueError("Dataset must contain a 'text' column.")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    # Apply tokenization to the dataset
    preTrain_dataset = preTrain_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch (convert to torch tensors)
    preTrain_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Define training arguments
    batch_size = 8  # Adjust batch size based on your system's capabilities
    
    training_args = TrainingArguments(
        output_dir=model_folder,           # Directory to save model checkpoints
        evaluation_strategy="no",          # No evaluation during pretraining
        learning_rate=2e-5,                # Learning rate
        per_device_train_batch_size=batch_size,  # Training batch size
        num_train_epochs=3,                # Number of training epochs
        weight_decay=0.01,                 # Weight decay
        save_strategy="epoch",             # Save model checkpoint every epoch
        logging_dir=f"{model_folder}/logs",# Directory for logs
        report_to="none"                   # No reporting (disable WandB or other tools)
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preTrain_dataset,    # Train on the custom dataset
        data_collator=data_collator        # Use data collator for MLM if applicable
    )
    
    # Start pretraining
    trainer.train()
    
    # Save the final pretrained model
    model.save_pretrained(f"{model_folder}/final_model")
    tokenizer.save_pretrained(f"{model_folder}/final_model")
    
    print(f"Pretraining completed. Model and tokenizer saved to {model_folder}/final_model")



def Transformers_predict(logger, model_select, model_predict, test_data, model_folder, fileName):
    ## Load trained model and predict
    if model_predict == 1:
        logger.info("=========================================================")  
        logger.info("================Prediction on Test data==================")
        logger.info("=========================================================")
        predict_st = time.time()
        
        logger.info("=======loading FineTuned model==========")
        # Tokenize the text and return PyTorch tensors:
        # model_folder = model_folder + "checkpoint-7664/"
        tokenizer = AutoTokenizer.from_pretrained(model_folder)
        logger.info("model_folder saved is \n {}".format(model_folder))

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
        
        labeledFile = fileName.split('.')[0] + "_yPred_" + model_select +  ".json"

        logger.info("test_data_yPred saved in \n {}".format(labeledFile))


        test_data_yTrue_yPred.to_json(labeledFile, orient='records')

        logger.info("test_data_yPred is \n {}".format(test_data_yTrue_yPred.shape))
        
        logger.info("prediction time {} seconds".format(time.time()-predict_st))



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
            tokenized_dataset = AllTrainData.map(preprocess_function, num_proc=1, batched=True, batch_size=5)
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


def Transformers_train(logger,  model_select, model_train, model_type, model_folder):

    # and train a model on the all_train_data and save the scores
    # Then test it on the test_data and save the predictions and scores
    if model_train == 1:
        model_st = time.time()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("device is {}".format(device))
        
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

        # # Before you start training your model, create a map of the expected ids to their labels with id2label and label2id:
        # id2label = {0: "0", 1: "1", 2: "2"}
        # logger.info("id2label is \n {}".format(id2label))
        # label2id = {'0': 0, '1': 1, '2': 2}
            
        # train

        logger.info("========Training Model=========")

        logger.info("========Select Model=========")
        # from epoch 0
        # multi class and single label; not problem_type="multi_label_classification"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_type, num_labels=3
        )

        # Enable gradient checkpointing to reduce memory usage
        # model.gradient_checkpointing_enable()
        
        # # path to the model checkpoint from the 36th epoch
        # model_checkpoint = "/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/DataModelsResults/Results/OpenAIGPT2/checkpoint-288000/"
        # # Load the model from the checkpoint
        # model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

        if model_select=="OpenAIGPT2": # https://github.com/huggingface/transformers/issues/3859 and https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/ 
            # resize model embedding to match new tokenizer
            model.resize_token_embeddings(len(tokenizer))

            # fix model padding token id
            model.config.pad_token_id = model.config.eos_token_id

        model.to(device)

        logger.info("======== Model args =========")

        batch_size=2

        training_args = TrainingArguments(
            output_dir=model_folder,
            seed=seed,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size, # to avoid OOM
            gradient_accumulation_steps=2, # to avoid OOM
            per_device_eval_batch_size=batch_size, # to avoid OOM
            num_train_epochs=100, #prev runs saturated at less than 50/100
            weight_decay=0.01,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            save_total_limit=2,
            save_steps=16,
            eval_steps=16,
            # fp16=True, # to avoid OOM # remove to run on GTX 1080 CARD
            metric_for_best_model="f1",  # Use the F1 score as the metric
            greater_is_better=True,  # Higher F1 score is better
            disable_tqdm=False,  # Show progress bar
        )

        # Set CUDA_LAUNCH_BLOCKING environment variable
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        logger.info("======== Trainer =========")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_AllTrainData["train"],
            eval_dataset=tokenized_AllTrainData["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )


        # # Resume training from the checkpoint
        # logger.info("======== Resuming trainer.train() from last checkpoint =========")
        # trainer.train(resume_from_checkpoint=checkpoint_dir)

        logger.info("======== trainer.train() =========")

        trainer.train()
        
        logger.info("========Saving Model=========")
        
        trainer.save_model()

        logger.info("========Log the best model details=========")

        # Log the best model
        # Extract and log the best model checkpoint and the step/epoch
        best_model_checkpoint = trainer.state.best_model_checkpoint
        best_metric = trainer.state.best_metric
        # Log the best model details
        logger.info(f"Best model checkpoint: {best_model_checkpoint}")
        logger.info(f"Best model's score: {best_metric}")

        # Extract the step from the checkpoint name
        best_step = None
        if best_model_checkpoint:
            checkpoint_parts = best_model_checkpoint.split('-')
            if len(checkpoint_parts) > 1 and checkpoint_parts[-1].isdigit():
                best_step = int(checkpoint_parts[-1])
        # Log the best model details
        if best_step is not None:
            logger.info(f"Best model was saved at step: {best_step}")
        else:
            logger.info("Best model step could not be determined from checkpoint name.")

        
        logger.info("train time {} seconds".format(time.time()-model_st))
        
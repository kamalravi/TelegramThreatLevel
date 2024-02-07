# import libraries
from sklearn.model_selection import StratifiedKFold
import time
import pandas as pd

# Functions

def split_data(logger, data_prep, root_dir, data_file):
    logger.info("Getting Data and creating train and test splits for each Fold")
    Getting_Data_st = time.time()
    data = pd.read_json(root_dir+'/Data/'+data_file, orient = 'records')
    logger.info("Total num of samples: {}".format(len(data)))
    data5split(logger, data, data_prep, root_dir)
    logger.info("Getting_Data time {} seconds".format(time.time()-Getting_Data_st))

def data5split(logger, data, data_prep, root_dir):
    ## Load data. Get K-Fold data. Save 5 fold indices (80% train, 20% test)
    skf = StratifiedKFold(n_splits=5, shuffle=False) # data is already shuffled with random_state=42
    for idx, (train_index, test_index) in enumerate(skf.split(data["article"].tolist(), data["label"].tolist())):
        if data_prep == 1:
            # logger.info("Fold:", idx+1, "TRAIN:", len(train_index), "TEST:", len(test_index))
            logger.info("Fold: {}, TRAIN: {}, TEST: {}".format(idx+1, len(train_index), len(test_index)))
            # with open('../../data_clean/KFoldData/all_train_index_Fold_' + str(idx+1)+ '.txt', 'w') as filehandle:
            with open(root_dir + '/KFoldData/all_train_index_Fold_' + str(idx+1)+ '.txt', 'w') as filehandle:
                for listitem in train_index:
                    filehandle.write('%s\n' % listitem)
            # with open('../../data_clean/KFoldData/test_index_Fold_' + str(idx+1)+ '.txt', 'w') as filehandle:
            with open(root_dir + '/KFoldData/test_index_Fold_' + str(idx+1)+ '.txt', 'w') as filehandle:
                for listitem in test_index:
                    filehandle.write('%s\n' % listitem)
        # break

def data_module(logger, data_file, KFold, root_dir):
    ## Load  (80% train, 20% test) indices from .txt files
    ## data is already shuffled with random_state=42
    data = pd.read_json(root_dir+'/Data/'+data_file, orient = 'records')
    logger.info("Total num of samples: {}".format(len(data)))
    train_index = pd.read_csv(root_dir + '/KFoldData/all_train_index_Fold_' + str(KFold)+ '.txt', sep=" ", header=None)
    test_index = pd.read_csv(root_dir + '/KFoldData/test_index_Fold_' + str(KFold)+ '.txt', sep=" ", header=None)
    logger.info("train_index.shape {}".format(train_index.shape))
    logger.info("test_index.shape {}".format(test_index.shape))
    # print(test_index[0].head())
    all_train_data = data.iloc[train_index[0]] #data.iloc[train_index]
    test_data = data.iloc[test_index[0]] #data.iloc[test_index]
    logger.info("all_train_data.shape {}".format(all_train_data.shape))
    logger.info("test_data.shape {}".format(test_data.shape))
    
    return all_train_data, test_data

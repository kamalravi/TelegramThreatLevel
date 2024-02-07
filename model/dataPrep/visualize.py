# import libraries
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from yellowbrick.text import TSNEVisualizer

import pickle

# Functions

# def get_DATAtSNE(visuallize, logger, KFold, all_train_data, root_dir):
def get_DATAtSNE(all_train_data, root_dir, visuallize, logger, KFold):
    ## Visualize high-dim data using t-SNE to get an intuition linearly separability
    if visuallize == 1:
        logger.info("Getting features and labels for plot")
        # Create the visualizer
        fig = plt.figure(figsize=(4,3))
        tSNE_ax = fig.add_subplot(111)
        tsne = TSNEVisualizer(
            perplexity=5,
            decompose_by=50, #1000 120GB RAM maxed out 
            random_state=42, 
            alpha=0.3, 
            colors=["blue", "red", "green"], 
            title="t-SNE projection of " + str(len(all_train_data)) +" Articles ",
            ax=tSNE_ax)

        Getting_tSNE_st = time.time()
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
        features = tfidf_vectorizer.fit_transform(all_train_data.article) #.toarray()
        labels = all_train_data.label

        # Open a file and use dump()
        with open(root_dir + "/Results/tSNE/features_KFold_" + str(KFold) + ".pkl", 'wb') as file:
            # A new file will be created
            pickle.dump(features, file)
        with open(root_dir + "/Results/tSNE/labels_KFold_" + str(KFold) + ".pkl", 'wb') as file:
            # A new file will be created
            pickle.dump(labels, file)

        logger.info("tfidf_vectorizer features shape {}".format(features.shape))
        logger.info("labels.shape {}".format(labels.shape))
        logger.info("tfidf_vectorizer time {} seconds".format(time.time()-Getting_tSNE_st))

        logger.info("Getting t-SNE fit")
        Getting_tSNE_fit = time.time()
        # draw the vectors
        tsne.fit(features, labels)
        logger.info("tfidf_vectorizer fit time {} seconds".format(time.time()-Getting_tSNE_fit))
        tsne.show(outpath= root_dir + "/Results/tSNE/tSNE_KFold_" + str(KFold) + ".png")
        logger.info("Getting_tSNE time {} seconds".format(time.time()-Getting_tSNE_st))
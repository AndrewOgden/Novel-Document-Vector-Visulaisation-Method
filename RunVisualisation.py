import gensim
import pandas as pd
import numpy as np
import seaborn as sns
import math
import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from gensim.models import Word2Vec
from scipy import spatial
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import time


categories = ["alt.atheism", "comp.sys.ibm.pc.hardware", 'sci.space']  
#list should contain the categories you want to plot
#must be in the list of valid topics found at https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

no_samples = 10

def euclidean(x, y):
    """Calculates the euclidean distance between two input vectors

    Args:
        x (array_like): The first input vector
        y (array_like): The second input vector
    """
    
    return(spatial.distance.euclidean(x, y))

def cosine(x, y):
    
    """Calculates the cosine distance between two input vectors 
    
    Args:
        x (array_like): The first input vector
        y (array_like): The second input vector
    """

    
    return(spatial.distance.cosine(x, y))

def remove_stopwords(x):
    """Removes the stopwords from the input string, returns the resulting string

    Args:
        x (String): The string to remove stopwords from
    """
    
    filtered_text = [t for t in x if not t in stopwords.words("english")]
    
    return(filtered_text)


def get_label(x):
    """gets the assoicated lable for an index in the categories list,  
    used to create a new coumn in a pandas dataframe populated with text labels  

    Args:
        x (int): an index of the categories array
    """
    
    return(categories[x])

def get_topic_centers(x_values, y_values):
    """Gets the average value of every point within a topic

    Args:
        x_values (array_like): the x values of each point
        y_values (array_like): the y values of each points
    """
    
    topic_totals = [ [0]*2 for _ in range(len(categories))]
    topic_counts = [0] * len(categories)

    #totals each axis for each topic
    for i, label in enumerate(labels):
        topic_totals[label][0] += y_values[i]
        topic_totals[label][1] += x_values[i]
    
        topic_counts[label] += 1
    
    #gets the averge of both axes for each topic
    for i in range(0, len(categories)):
        topic_totals[i][0] = topic_totals[i][0]/ topic_counts[i]
        topic_totals[i][1] = topic_totals[i][1]/ topic_counts[i]
    
    return(topic_totals)
    
    


#load the text to be used as the reference vector, the Bible by default 
with open('bible.txt') as f:
    ref_text = f.readlines()
        
ref_text = ' '.join(ref_text)
ref_text = ref_text.replace('\n', '')

#create a pandas dataframe entry for the vector    
ref_text_table = pd.DataFrame(data = list(zip([ref_text], [1], [gensim.utils.simple_preprocess(ref_text)])), columns = ['text', 'target', 'text_clean'])

#update the text to the preprocessed version
ref_text = gensim.utils.simple_preprocess(ref_text)



#load the training section of the dataset
news_data = fetch_20newsgroups(
    subset="train",
    remove=['headers', 'footers'],
    categories=categories,
)

#load the testing section of the dataset
tnews_data = fetch_20newsgroups(
    subset="test",
    remove=['headers', 'footers'],
    categories=categories,
)

#create pandas datframe with training data
dataset = pd.DataFrame(news_data.data, columns=['text'])
dataset['target'] = pd.Series(data=news_data.target, index=dataset.index)

#preprocess the data and add it as new column in the dataframe
dataset['text_clean'] = dataset['text'].apply(lambda x: gensim.utils.simple_preprocess(x))

#optionaly remove stopwords
#dataset['text_clean'] = dataset['text_clean'].apply(lambda x: remove_stopwords(x))


#add the referece vector to the dataframe
dataset = dataset.append(ref_text_table, ignore_index=True)


#create pandas datframe with testing data
testing_dataset = pd.DataFrame(tnews_data.data, columns=['text'])
testing_dataset['target'] = pd.Series(data=tnews_data.target, index=testing_dataset.index)

#preprocess the data and add it as new column in the dataframe
testing_dataset['text_clean'] = testing_dataset['text'].apply(lambda x: gensim.utils.simple_preprocess(x))

#optionaly remove stopwords
#testing_dataset['text_clean'] = testing_dataset['text_clean'].apply(lambda x: remove_stopwords(x))

#create new column with each etries associated labels
dataset['target_label'] = dataset['target'].apply(lambda x: get_label(x))
testing_dataset['target_label'] = testing_dataset['target'].apply(lambda x: get_label(x))

#optionaly select a subsection of the dataset
#dataset = dataset.sample(n=1000, ignore_index=True)

#combine both datasets and use the result to train the Word2Vec model, this gives the model more   
#reference data and appeared to imprve the qaulity of the vectors produced. Thus making the resulting 
#visualisaions better. The testing data was not plotted, just used to increaese vector quality.

full_data = dataset['text_clean'].append(testing_dataset['text_clean'], ignore_index=True)

#optionaly highlight specific values on the plot by including thier index in the list
#highlight = [219, 147, 159, 1042, 667, 207, 873, 769, 372, 207, 761, 1049, 598, 150, 920, 683]
highlight = []

#print the text and lable of the highlighted points

# for i in lst:xz                  
#     print(i)
#     print(dataset['target'][i])
#     print('\n\n')
#     print(dataset['text'][i])
#     print('\n\n\n\n\n')
    
    


#run the Word2Vec model
w2v_model = gensim.models.Word2Vec(full_data,
                                   vector_size=100,
                                   window=5,
                                   min_count=2)

#get the set of words that have an associated vecotor 
words = set(w2v_model.wv.index_to_key )

#create lists of all word vectors in each document in the testing dataset
doc_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in dataset['text_clean']], dtype=object)

#create lists of all word vectors in each document in the testing dataset
#not used in this implemtaion, but could be used for testing classification techniques or other uses 
testing_doc_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in testing_dataset['text_clean']], dtype=object)

#create array of all word vectors in the reference test
ref_vec = np.array([w2v_model.wv[i] for i in ref_text if i in words])

#empty lists to store the resulting document vectors
doc_vect_avg = [0] * len(doc_vect)
testing_doc_vect_avg = [0] * len(testing_doc_vect)

#create document vectors for all documents with word vectors in testing dataset
for i in range(0, len(doc_vect)):
    if doc_vect[i].size:
        doc_vect_avg[i] = doc_vect[i].mean(axis=0)
    
#create document vectors for all documents with word vectors in testing dataset       
for i in range(0, len(testing_doc_vect)):
    if testing_doc_vect[i].size:
        testing_doc_vect_avg[i] = testing_doc_vect[i].mean(axis=0)
   
#create reference vectors       
ref_vec = ref_vec.mean(axis = 0)
  
#create reference vector of all key words in the corpus        
# ref_vec = list(w2v_model.wv.index_to_key)
# ref_vec = np.array([w2v_model.wv[i] for i in ref_vec if i in words])
# ref_vec = ref_vec.mean(axis=0)

#create refernece vector from the first document in the dataset
#ref_vec = doc_vect_avg[0]

#for any non array_like vectors, replace with array of zeros
#could be changed to remove instead, but the associated entires in the pandas dataframe must als be removed so they match for later use
for i in range(0, len(doc_vect_avg)):
    if isinstance(doc_vect_avg[i], int):
        doc_vect_avg[i] = ([0]*100)


        
#get the list of topic lables 
labels = dataset['target']
labels_t = dataset['target_label']
eucs = []
coss = []

#get the vlaues for the novel metric
start = time.time()       
for v in doc_vect_avg:
    eucs.append(float(euclidean(v, [0] * len(ref_vec))))
    coss.append(float(cosine(v, ref_vec)))  
print("nov - ", time.time() - start)   
    
#normalise both ranges of values
eucs = [(float(i)-min(eucs))/(max(eucs)-min(eucs)) for i in eucs]
coss = [(float(i)-min(coss))/(max(coss)-min(coss)) for i in coss]


topic_centers = get_topic_centers(coss, eucs)

#plot the novel visualisation
fig, ax = plt.subplots(nrows = 1, ncols = 3)
plt.title('Topic Plotting')

sns.scatterplot(x=coss,
                y=eucs,
                hue=labels_t,
                palette="deep",
                legend=['Ath', 'crypt'],
                alpha=0.3,
                ax=ax[0]
                )
ax[0].plot([x for _, x in topic_centers ],
                 [y for y, _ in topic_centers],
                 'k+',
                 markersize=15,
                 )

ax[0].set_title('Cosine Vs Euclidean')
ax[0].set_xlabel('Cosine Disimilarity')
ax[0].set_ylabel('Euclidean Similarity (Normalised)')

#annoate the list of points from the previous list
for i, txt in enumerate(labels):
    if i in highlight:
        ax[0].annotate(i, (coss[i], eucs[i]))

#calculate and then print the intra cluster distence
distence = abs(topic_centers[0][1] - topic_centers[1][1])
print("Cosine differnce - ", distence)




#convert list of docment vecotrs to numpy array
dv = np.array(doc_vect_avg, dtype=object)

#run t-sne
starttsne = time.time()
tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(dv)    
print("t-sne - ", time.time() - starttsne)   

dataset['tsne-2d-one'] = tsne_results[:,0]
dataset['tsne-2d-two'] = tsne_results[:,1]

print(dataset)

topic_centers = get_topic_centers(dataset['tsne-2d-two'], dataset['tsne-2d-one'])

#plot the t-sne visualisation
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="target_label",
    palette="deep",
    data=dataset,
    legend="full",
    alpha=0.3,
    ax=ax[1]
)
ax[1].plot([x for x, _ in topic_centers ],
                 [y for _, y in topic_centers],
                 'k+',
                 markersize=15,
                 )


for i, txt in enumerate(labels):
    if i in highlight:
        ax[1].annotate(i, (dataset['tsne-2d-one'][i], dataset['tsne-2d-two'][i]))

ax[1].set_title('T-sne')

#calcultae PCA
startpca = time.time()   
pca = PCA(n_components=2)
pca_results = pca.fit_transform(dv)
print("pca - ", time.time() - startpca)   

dataset['pca-component-one'] = pca_results[:,0]
dataset['pca-component-two'] = pca_results[:,1]

topic_center = get_topic_centers(dataset['pca-component-two'], dataset['pca-component-one'])

#Plot PCA data
sns.scatterplot(
    x="pca-component-one", y="pca-component-two",
    hue="target_label",
    palette="deep",
    data=dataset,
    legend="full",
    alpha=0.3,
    ax=ax[2]
)
ax[2].plot([x for x, _ in topic_center ],
                 [y for _, y in topic_center],
                 'k+',
                 markersize=15,
                 )

for i, txt in enumerate(labels):
    if i in highlight:
        ax[2].annotate(i, (dataset['pca-component-one'][i], dataset['pca-component-two'][i]))

ax[2].set_title('PCA')

#finaly, show the figures.
plt.show()





        
    

    

        




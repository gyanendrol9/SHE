#-*- coding: utf-8 -*-
import numpy as np
import gc
import pickle
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')



from gensim.models.fasttext import FastText

stopwords = ['rt','amp','url','sir','day','title','shri','crore','time',"a", "about","above", "across", "after", "afterwards", "again", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]


#load your pretrained embedding

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('LOAd_eMbedding.emb', binary=False) 
idx=1
words = []
word2idx = {}
vectors=[]
vect_idx=[]

vectors.append(np.asarray([0]*model.vector_size))
with open('LOAd_eMbedding.emb', 'rb') as f:
    for i,l in enumerate(f):
        if i > 0:
            line = l.decode().split()
            if line[1] != '-nan':
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                vect = line[1:]
                vectors.append(np.asarray(vect))
                vect_idx.append(idx)
                idx += 1
                

print(len(vectors), len(word2idx), len(vectors[0]))

emb=len(vectors[0])
vectors=np.asarray(vectors)
vect_idx=np.array(vect_idx)
tss=int(len(word2idx) * 0.8)
train, test = vectors[:tss][:], vectors[tss:][:]
train_idx, test_idx = vect_idx[:tss][:], vect_idx[tss:][:]
sent=dict()


#load sentiment lexicon
f=open('data/Tweets_sentilexicon_final_auto', 'r')
lines=f.readlines()
for l in lines:
    word=l.lower().strip().split('\t')
    if word[1] not in sent:
        sent[word[1]]=[]
    sent[word[1]].append(word[0])
f.close()

positive=[]
negative=[]
neutral=[]

def gen_index(wlist,word2idx):
    ip=[]
    for i in wlist:
        try:
            idx=word2idx[i]
            ip.append(idx)
        except:
            pass
    return ip

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

folds=dict()

positive = gen_index(sent['positive'],word2idx)
negative = gen_index(sent['negative'],word2idx)
neutral = gen_index(sent['neutral'],word2idx)

from random import shuffle

print("Data preparation for fold ")

fold=[]
for idx in sent['positive']:
    fold.append(idx)
shuffle(fold)
folds['positive']=split_list(fold, wanted_parts=10)

fold=[]
for idx in sent['negative']:
    fold.append(idx)
shuffle(fold)
folds['negative']=split_list(fold, wanted_parts=10)

fold=[]
for idx in sent['neutral']:
    fold.append(idx)
shuffle(fold)
folds['neutral']=split_list(fold, wanted_parts=10)

f_out = open('data/3class_sent_fold.pkl','wb')
pickle.dump(folds,f_out)
f_out.close()


# f_out = open('SHE/SHE_new_fasttext_3class_sent_fold.pkl','rb')
# folds=pickle.load(f_out)
# f_out.close()


from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,SimpleRNN,GRU,CuDNNGRU,Reshape
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation, Flatten, UpSampling1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import plot_model
import tensorflow as tf
from keras.models import model_from_json
import pickle
from sklearn.metrics import accuracy_score
from keras import backend as K

for fi in range(1):
    testing=[]
    for sample in folds['positive'][fi]:
        try:
            idx=word2idx[sample]
            testing.append([idx,[1,0,0]])
        except:
            pass
    
    for sample in folds['negative'][fi]:
        try:
            idx=word2idx[sample]
            testing.append([idx,[0,1,0]])
        except:
            pass
    
    for sample in folds['neutral'][fi]:
        try:
            idx=word2idx[sample]
            testing.append([idx,[0,0,1]])
        except:
            pass
    
    #testing+=folds['neutral'][fi]
    training=[]
    for fj in range(10):
        if fj != fi:
            for sample in folds['positive'][fj]:
                try:
                    idx=word2idx[sample]
                    training.append([idx,[1,0,0]])
                except:
                    pass
            
            for sample in folds['negative'][fj]:
                try:
                    idx=word2idx[sample]
                    training.append([idx,[0,1,0]])
                except:
                    pass
            
            for sample in folds['neutral'][fj]:
                try:
                    idx=word2idx[sample]
                    training.append([idx,[0,0,1]])
                except:
                    pass
            
            #training+=folds['neutral'][fj]
   
    print( "Training fold begin",fi)

    training_labels=[]
    testing_labels=[]

    training_idx=[]
    testing_idx=[]

    training_AE=[]
    testing_AE=[]   # = sent_vec[:TSS][:], sent_vec[TSS:][:]
    for i in training:
        idx=i[0]
        training_labels.append(i[1])
        training_AE.append(vectors[idx])
        training_idx.append(idx)

    training_AE=np.array(training_AE)
    training_labels=np.array(training_labels)
    training_idx=np.array(training_idx)

    for i in testing:
        idx=i[0]
        testing_labels.append(i[1])
        testing_AE.append(vectors[idx])
        testing_idx.append(idx)

    testing_AE=np.array(testing_AE)
    testing_labels=np.array(testing_labels)
    testing_idx=np.array(testing_idx)
    #for i in range(10):
    input_node=Input(shape=(emb,))
    #encode=Embedding(len(word2idx), emb, weights=[vectors], input_length=1,trainable=True)(input_node)
    encode=Reshape((1,emb))(input_node)
    encode=Conv1D(emb,
                     3,
                     padding='same',
                     activation='relu',
                     strides=1)(encode)
    encode=MaxPooling1D(pool_size=1)(encode)
    decoder=Dropout(0.2)(encode)
    decoder=Conv1D(64,
                     3,
                     padding='same',
                     activation='relu',
                     strides=1)(decoder)
    decoder=GlobalMaxPooling1D()(decoder)
    decode=Dense(emb,activation='tanh',name="dense_1")(decoder)
    
    classifier=Conv1D(64,
                     3,
                     padding='same',
                     activation='relu',
                     strides=1)(encode)
    classifier=Dropout(0.2)(classifier)
    classifier=GlobalMaxPooling1D()(classifier)
    senti_class=Dense(3,activation='softmax',name="dense_2")(classifier)


    encoder = Model(input_node, encode)
    
    autoencoder=Model(input_node,decode)
    autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    autoencoder.summary()

    
    combine=Model(input_node,[decode,senti_class])

    losses = {
        "dense_1": "mean_squared_error",
        "dense_2": "categorical_crossentropy",
    }
    lossWeights = {"dense_1": 1.0, "dense_2": 1.0}
    combine.compile(optimizer='adam', loss=losses, loss_weights=lossWeights,
        metrics=["accuracy"])
    combine.summary()
    for learn in range(5):
        autoencoder.fit(train, train,
                    epochs=20,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(test, test))
                    
        combine.fit(training_AE, [training_AE, training_labels],
                    epochs=20,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(testing_AE, [testing_AE, testing_labels]))
    
    model_json = combine.to_json()
    with open("model/senti_autoencoder_combine.json", "w") as json_file:
        json_file.write(model_json)
    combine.save_weights("model/senti_autoencoder_combine.h5")

    model_json = encoder.to_json()
    with open("model/3class_encoder.json", "w") as json_file:
        json_file.write(model_json)
    encoder.save_weights("model/3class_encoder.h5")
    
    
    
    f2=open('model/SHE_encoder_out.emb','w')
    f2.write(str(len(word2idx))+' '+str(emb)+'\n')
    for key in word2idx:
        f2.write(key.replace(' ','')+" ")
        w2v=model[key]
        result=encoder.predict(np.asarray([w2v]),verbose=0)
        for i in list(result[0][0]):
            f2.write(str(i)+" ")
        f2.write("\n")
    f2.close()

    
    del input_node
    del encode
    del classifier
    del senti_class
    del autoencoder
    del combine
    gc.collect()   
    K.clear_session()


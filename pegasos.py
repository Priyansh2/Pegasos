##Implementation of Pegasos (Kernalised and Non-kernalised)
## NOTE: Download everything from below link in directory containing the 'pegasos.py'.
## Link- "https://drive.google.com/open?id=155TgsWyn_yvtsOGCGLRE6Xa54FKucDW-"
import os,sys,re,numpy as np,time,logging,gzip,fastText,string
from pprint import pprint
from io import open
from numpy import linalg
from collections import defaultdict,Counter
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV,ShuffleSplit,cross_val_score,KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support as score, make_scorer
from scipy.special import expit
from sklearn import preprocessing
from fastText import load_model
from pathlib import Path

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class Pegasos():
    ##Non-kernalised Pegasos
    def __init__(self,epoch=10,lamb=1e-3,loss_func="hinge"):
    	##lamb --> regularisation param
        ## loss_func can be "hinge loss" or "logistic loss"
        self.pos_cl = 1
        self.neg_cl = -1
        self.epoch=epoch
        self.lamb = lamb
        self.loss_func_list=("hinge","log")
        if loss_func not in self.loss_func_list:
            raise Exception("Loss function not found!!\n")
        if loss_func==self.loss_func_list[0]:
            self.obj = self.update_hinge
        else:
            self.obj = self.update_log
        #self.init_index = initial_index

    def update_hinge(self,score,yi,xi,eta,w):
        if yi*score < 1:
            return (1 - eta*self.lamb)*w + eta*yi*xi
        else:
            return (1 - eta*self.lamb)*w

    def update_log(self,score,yi,xi,eta,w):
        return (1 - eta*self.lamb)*w + xi*eta*(yi/(1+expit(yi*score)))

    def fit(self,x,y):
        try:
            x=x.toarray()
        except AttributeError:
            pass
        x = np.insert(x,x.shape[1],1,axis=1)
        #print(x[0],x[1])
        m,n = x.shape[0],x.shape[1] ## m --> number of samples , n --> number of features
        self.w = np.zeros(n)
        y = list(y)
        classes = sorted(set(y))
        if len(classes) != 2:
            raise Exception("Not a binary classification!!\n")
        for i in range(self.epoch):
            eta = 1 / (self.lamb*(i+1))
            j = np.random.randint(0,m)
            #j=self.init_index
            xi, yi = x[j], y[j]
            score = xi.dot(self.w)
            self.w = self.obj(score,yi,xi,eta,self.w)
        print("fitting Complete!!\n")
        return self

    def predict(self,x):
        try:
            x=x.toarray()
        except AttributeError:
            pass
        x = np.insert(x,x.shape[1],1,axis=1)
        scores = x.dot(self.w)
        out = np.select([scores>=0.0, scores<0.0], [self.pos_cl, self.neg_cl])
        return out

class KernelPegasos():
    ## Kernelised Pegasos with different kernel support:- "linear","guassian", "polynomial"
    def __init__(self,epoch=10,lamb=1e-3,kernel="gaussian",loss_func="hinge"):
        ##lamb --> regularisation param
        ## loss_func can be "hinge loss" or "logistic loss"
        self.kernel_list=("polynomial","gaussian","linear")
        self.epoch=epoch
        self.pos_cl = 1
        self.neg_cl = -1
        self.lamb = lamb
        #self.init_index = initial_index
        self.loss_func_list=("hinge","log")
        if loss_func not in self.loss_func_list:
            raise Exception("Loss function not found!!\n")
        if loss_func==self.loss_func_list[0]:
            self.obj = self.update_hinge
        else:
            self.obj = self.update_log
        if kernel not in self.kernel_list:
            raise Exception("Kernel not found!!\n")
        if kernel==self.kernel_list[0]:
            self.kernel = self.polynomial_kernel
        elif kernel==self.kernel_list[1]:
            self.kernel = self.gaussian_kernel
        else:
            self.kernel = self.linear_kernel

    def update_hinge(self,score,yi,xi,eta,w):
        if yi*score < 1:
            return (1 - eta*self.lamb)*w + eta*yi*xi
        else:
            return (1 - eta*self.lamb)*w

    def update_log(self,score,yi,xi,eta,w):
        return (1 - eta*self.lamb)*w + xi*eta*yi / (1+expit(yi*score))

    def linear_kernel(self,x,y):
        return np.dot(x,y)

    def polynomial_kernel(self,x, y, p=3):
        return (1 + np.dot(x, y)) ** p

    def gaussian_kernel(self,x, y, sigma=5.0):
        return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

    def fit(self,x,y):
        ## Complexity :- O(#epoch * #train samples)
        try:
            x=x.toarray()
        except AttributeError:
            pass
        x = np.insert(x,x.shape[1],1,axis=1)
        m,n = x.shape[0],x.shape[1] ## m --> number of samples , n --> number of features
        self.alpha = np.zeros((self.epoch+1,m))
        y = list(y)
        classes = sorted(set(y))
        if len(classes) != 2:
            raise Exception("Not a binary classification!!\n")
        for i in range(self.epoch):
            eta = 1. / (self.lamb*(i+1))
            j = np.random.randint(0,m)
            #j = self.init_index
            xi, yi = x[j], y[j]
            for k in range(m):
                if k!=j:
                    self.alpha[i+1,k]=self.alpha[i,k]
            sum_=0.0
            for k in range(m):
                sum_+=self.alpha[i,k]*self.kernel(xi,x[k])*y[k]
            sum_*=yi*eta
            if sum_<1:
                self.alpha[i+1,j] = self.alpha[i,j]+1
            else:
                self.alpha[i+1,j]=self.alpha[i,j]
        self.alpha = self.alpha[self.epoch]
        self.y_train = y
        self.x_train = x
        print("fitting Complete!!\n")
        return self

    def predict(self,x):
        ## complexity O(#support vectors * #test samples)
        try:
            x=x.toarray()
        except AttributeError:
            pass
        x = np.insert(x,x.shape[1],1,axis=1)
        l = x.shape[0]
        m = self.alpha.shape[0]
        scores = np.zeros(l)
        for i in range(l):
            score=0.0
            for k in range(m):
                if self.alpha[k]>0:
                    score+=self.alpha[k]*self.kernel(x[i],self.x_train[k])*self.y_train[k]
            scores[i]=score
        out = np.select([scores>=0.0, scores<0.0], [self.pos_cl, self.neg_cl])
        return out

##### Utility functions defined here ####
def read_corpus(corpus_file,get_tokens=True): ## for sentiment_analysis
    X = []
    Y = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            Y.append(tokens[1])
            if get_tokens:
                X.append(tokens[3:])
            else:
                X.append(" ".join(token for token in tokens[3:]))
    return X, Y

def mapping_label(labels,Y,inverse=False):
    ## Mapping numerical value of label to its actual value (if inverse=False)
    if inverse:
        if isinstance(labels, dict):
            y = np.array([labels[j] for j in Y])
        else:
            print("Data type is not 'dictionary'")
    else:
        y = np.array([labels[j] for j in Y])
    return y

def preprocess_text(text):
	##Custom naive preprocessor based on "space" as delimiter.
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    punctuations_marks = string.punctuation
    punctuations_marks = punctuations_marks + '¿'
    text = text.translate(str.maketrans('','',punctuations_marks))
    text = text.strip()
    return text

def join_tokens(token_list):
	##Detokenizer with "space" delimiter
    return " ".join(token for token in token_list)

def extract_data(file):
	## extract fasttext formatted data into X,Y where X contains sentences and Y
	# contains their labels.
    X,Y=[],[]
    with open(file,"r",encoding="utf-8") as fl:
        for line in fl:
            text = line.strip()
            Y.append(int(text.split("__label__")[1].split()[0]))
            X.append(join_tokens(text.split("__label__")[1].split()[1:]).strip())
    #print(X[0],Y[0])
    return X,Y

def prepare_data_for_fasttext(X,Y,filename):
	## Prepare raw data for fasttext model training
    data=''
    fd = open(filename,"w",encoding='utf-8')
    for i in range(len(X)):
        sent = preprocess_text(join_tokens(X[i]))
        label = Y[i]
        data+="__label__"+str(label)+" "+sent+"\n"
    fd.write(data)
    fd.close()

def train_ft_model(X,Y):
	##Training fasttext model. This is done to obtain word-vectors for word
	#(feature vector) and would be used in training Pegasos.
    #help(fastText.FastText)
    corpus_file = "./sentiment.train"
    exist = Path(corpus_file)
    if not exist.is_file():
        prepare_data_for_fasttext(X,Y)
    model = fastText.train_supervised(corpus_file,epoch=25,wordNgrams=3)
    model.save_model("ft_models/ftmodel-dim-100.bin")
    print("Training completed!!\n")
    print("Model saved in -> ","~/ft_models/ftmodel-dim-100.bin\n")


def standardise(X):
	## Change data distribution into gaussian distribution with zero mean and
	# unit sd.
    return preprocessing.scale(X)

def normalise(X):
    ## Normalise data matrix by making each feature vector of unit norm.
    return preprocessing.normalize(X,norm='l2')


def print_fold_info(fold):
	##Print info of a fold
    Xtrain = fold[0]
    Ytrain = fold[1]
    Xtest = fold[2]
    Ytest = fold[3]
    print(Xtrain.shape,Ytrain.shape,Xtest.shape,Ytest.shape)
    print("Label: ",Ytrain[0],"\n") ## first label
    print("Text Vector: ",Xtrain[0],"\n\n") ## first sentence vector
    print("Training samples: ",len(Xtrain)) ## No. of train sentences in fold
    print("Test samples: ",len(Xtest)) ## No. of test sentences in fold
    print("\n\nDistribution of labels in train data:",Counter(Ytrain))
    print("Distribution of labels in test data:",Counter(Ytest))

def split_joint_data_matrix(data):
	## split [X|Y] into X,Y
    X = data.T
    m,n = data.shape[0],data.shape[1]
    Y = X[n-1]
    x = X[0]
    for j in range(1,n-1):
        x = np.vstack((x,X[j]))
    return x.T,np.squeeze(np.asarray(np.array(Y))).astype(dtype="int64")

def cross_splits(data,folds=10,is_shuffle=True):
	## 'K-folds' of data matrix is generated.
	## Value of 'k' is user-specific
	## Split can be made on after shuffling data by setting "is_shuffle=True"
	# else it will split on without it.
    data_splits=[]
    if is_shuffle:
        np.random.shuffle(data)
    print("Original_Data Shape: \n",data.shape)
    print("Each FOLD Shape: \n")
    (rows,cols) = data.shape
    score=0
    testrows = int(rows/folds)
    ptr=0
    alpha=ptr
    beta=ptr+testrows
    testdata=data[alpha]
    for j in range(alpha+1,beta):
        temp_testdata=data[j]
        testdata=np.vstack((testdata,temp_testdata))
    ptr+=testrows
    alpha=ptr
    beta=rows
    traindata=data[alpha]
    for j in range(alpha+1,beta):
        temp_traindata=data[j]
        traindata=np.vstack((traindata,temp_traindata))
    (row_te,col_te) = testdata.shape
    (row_tr,col_tr) = traindata.shape
    transposed_traindata =  traindata.T
    y_transposed = transposed_traindata[col_tr-1]
    alpha = 0
    beta = col_tr-1
    x_transposed = transposed_traindata[0]
    for j in range(alpha+1,beta):
        temp_x_transposed = transposed_traindata[j]
        x_transposed = np.vstack((x_transposed,temp_x_transposed))
    X_train = x_transposed.T
    Y_train = np.squeeze(np.asarray(np.array(y_transposed)))
    Y_train = Y_train.astype(dtype="int64")
    X_test,Y_test = split_joint_data_matrix(testdata)
    print("X_train     Y_train     X_test     Y_test","\n\n")
    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
    data_splits.append([np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test)])
    for k in range(2,folds+1):
        alpha=ptr
        beta=ptr+testrows
        testdata=data[alpha]
        for j in range(alpha+1,beta):
            temp_testdata=data[j]
            testdata=np.vstack((testdata,temp_testdata))
        ptr+=testrows
        alpha=0
        beta=(k-1)*testrows
        traindata1=data[alpha]
        for j in range(alpha+1,beta):
            temp_traindata1=data[j]
            traindata1=np.vstack((traindata1,temp_traindata1))

        alpha=ptr
        beta=rows
        traindata2=data[alpha]
        for j in range(alpha+1,beta):
            temp_traindata2=data[j]
            traindata2=np.vstack((traindata2,temp_traindata2))
        traindata = np.vstack((traindata1,traindata2))
        transposed_traindata = traindata.T
        y_transposed = transposed_traindata[traindata.shape[1]-1]
        alpha = 0
        beta = traindata.shape[1]-1
        x_transposed = transposed_traindata[0]
        for j in range(alpha+1,beta-alpha):
            temp_x_transposed = transposed_traindata[j]
            x_transposed = np.vstack((x_transposed,temp_x_transposed))
        X_train = x_transposed.T
        Y_train = np.squeeze(np.asarray(np.array(y_transposed)))
        Y_train = Y_train.astype(dtype="int64")
        X_test,Y_test = split_joint_data_matrix(testdata)
        print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
        data_splits.append([np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test)])
    return data_splits

def tfidf(X,Y,is_standardise=True,is_normalise=False,top_f=100):
	##TfIdf vectoriser which convert documents of raw corpus into tfidf feature
	# vectors.
	## Optional functionality is provided to standardise and normalise the tfidf
	# feature matrix.
    feature_gen = Pipeline( [
    ('tf_idf', TfidfVectorizer(preprocessor = lambda x: x,tokenizer = lambda x: x,ngram_range=(1, 3)
                              ,max_features=top_f))
    ])
    feature_vec = feature_gen.fit_transform(X,Y).toarray()
    #return feature_vec
    if is_standardise and not is_normalise:
        feature_vec = standardise(feature_vec) ##standarise due to data sparness
    elif is_normalise and not is_standardise:
        feature_vec = normalise(feature_vec)
    elif is_normalise and is_standardise:
        feature_vec = normalise(standardise(feature_vec))
    return feature_vec

def doc2vec(X,model,de_tokenise=False):
    ## generate sentence vector for given list of sentences.
    for i in range(len(X)):
        if de_tokenise:
            X[i] = join_tokens(X[i])
        sent = preprocess_text(X[i])
        X[i] = model.get_sentence_vector(sent)
    return X

def feature_extraction(X,Y,type_="doc2vec",cv=True):
	##Feature extraction using "tfidf" or "fastext"
	##Final output will be the k-folds (k is given by user and set to 10 by
	# default) of test,train data where each row represents document vector (sent2vec).
	## 'k-folds' or 'k-splits' of data is generated by setting 'cv=True' else
	# split will be made based on 80-20 ratio (80 for train and 20 for test)
    if type_=="tfidf":
        X = tfidf(X,Y)
        if cv:
            data = np.hstack([X,np.matrix(Y).T])
            splits = cross_splits(data)
            return splits
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y,test_size=0.2,random_state=4242)
    else:
        ft_model = load_model("ft_models/ftmodel-dim-100.bin")
        Xtrain,Ytrain = extract_data("./sentiment.train")
        Xtest,Ytest = extract_data("./sentiment.test")
        Ytrain,Ytest = mapping_label({0:-1,1:1},Ytrain,inverse=True),mapping_label({0:-1,1:1},Ytest,inverse=True)
        Xtrain = doc2vec(Xtrain,ft_model)
        Xtest = doc2vec(Xtest,ft_model)
        if cv:
            Xtrain = np.array(Xtrain)
            Xtest = np.array(Xtest)
            Xdata = np.vstack([Xtrain,Xtest])
            Ydata = np.vstack([np.matrix(Ytrain).T,np.matrix(Ytest).T])
            data = np.hstack([Xdata,Ydata])
            splits = cross_splits(data)
            return splits
        else:
            Xtrain,Ytrain,Xtest,Ytest = np.array(Xtrain),np.array(Ytrain),np.array(Xtest),np.array(Ytest)

    return [[Xtrain,Ytrain,Xtest,Ytest]]


X, Y = read_corpus('all_sentiment_shuffled.txt')
print("Total samples: ",len(X)) #Total samples:  11914
distinct_labels = list(set(Y))
print("Label kinds: ",distinct_labels) #Label kinds:  ['pos', 'neg']
labels = {"pos":1,"neg":-1}
Y = mapping_label(labels,Y,inverse=True)

##### Prepare data for fasttext and training it thereafter using this data #####
#prepare_data_for_fasttext(Xtrain,Ytrain,"sentiment.train")
#prepare_data_for_fasttext(Xtest,Ytest,"sentiment.test")
#train_ft_model(Xtrain,Ytrain)

'''ft_model = load_model("ft_models/ftmodel-dim-100.bin")
print("Vocabulary of model: ",len(ft_model.get_words()))
object_methods = [method_name for method_name in dir(ft_model) if callable(getattr(ft_model, method_name))]
print("\n","Model functions and attributes: ",object_methods,"\n\n")
test_data="./sentiment.test"
print("fasttext classification results: \n")
print_results(*ft_model.test(test_data))
##genereate sentence vec for text :- hello world! My name is xyz@omegalul
sample_sent = "hello world! My name is xyz@omegalul"
sent_vec = ft_model.get_sentence_vector(sample_sent)
print("\n",sent_vec.shape)
print(sent_vec)'''

##### Feature Extraction ######
splits = feature_extraction(X,Y,type_="tfidf",cv=True)
##### Printing each fold info of data #####
#for fold in splits:
    #print_fold_info(fold)
    #print("\n\n---------------------------------------\n\n")


##### Setting initials of Pegasos #####
c=1e-3
##### Running Classifiers with different settings #####
print("########## Pegasos (loss - hinge) ##########")
avg_acc=0.0
cnt=1
for fold in splits:
    X_train_mat = fold[0]
    y_train = fold[1]
    X_test_mat = fold[2]
    y_test = fold[3]
    clf = Pegasos(epoch=25,lamb=c,loss_func="hinge")
    t0 = time.time()
    #print(X_train_mat[0],y_train[0],X_test_mat[0],y_test[0])
    clf.fit(X_train_mat, y_train)
    t1 = time.time()
    #print('Training time:', t1-t0, 'seconds.\n')
    t0 = time.time()
    y_pred = clf.predict(X_test_mat)
    t1 = time.time()
    #print('Prediction time:', t1-t0, 'seconds.\n')
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy_score (in %): ",acc*100,"\n")
    #print(classification_report(y_test, y_pred))
    p,r,f,_ = score(y_test,y_pred,average=None,labels=[1,-1])
    print('P_pos={:.6f}, P_neg={:.6f}\n'.format(p[0],p[1]))
    print('R_pos={:.6f}, R_neg={:.6f}\n'.format(r[0],r[1]))
    print('F_pos={:.6f}, F_neg={:.6f}\n'.format(f[0],f[1]))
    print("\n\n---------END OF FOLD",str(cnt),"-----------\n\n")
    cnt+=1
    avg_acc+=acc
print("\n\nAverage Accuracy across all folds is : ",avg_acc*100/len(splits))

print("########## Pegasos (loss - logistic) ##########")
avg_acc=0.0
cnt=1
for fold in splits:
    X_train_mat = fold[0]
    y_train = fold[1]
    X_test_mat = fold[2]
    y_test = fold[3]
    clf = Pegasos(epoch=25,lamb=c,loss_func="log")
    #clf = KernelPegasos(epoch=20,lamb=c,initial_index=init_index,kernel="linear")
    t0 = time.time()
    clf.fit(X_train_mat, y_train)
    t1 = time.time()
    #print('Training time:', t1-t0, 'seconds.\n')
    t0 = time.time()
    y_pred = clf.predict(X_test_mat)
    t1 = time.time()
    #print('Prediction time:', t1-t0, 'seconds.\n')
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy_score (in %): ",acc*100,"\n")
    #print(classification_report(y_test, y_pred))
    p,r,f,_ = score(y_test,y_pred,average=None,labels=[1,-1])
    print('P_pos={:.6f}, P_neg={:.6f}\n'.format(p[0],p[1]))
    print('R_pos={:.6f}, R_neg={:.6f}\n'.format(r[0],r[1]))
    print('F_pos={:.6f}, F_neg={:.6f}\n'.format(f[0],f[1]))
    print("\n\n---------END OF FOLD",str(cnt),"-----------\n\n")
    cnt+=1
    avg_acc+=acc
print("\n\nAverage Accuracy across all folds is : ",avg_acc*100/len(splits))

print("########## Kernel Pegasos (kernel - linear) ##########")
avg_acc=0.0
cnt=1
for fold in splits:
    X_train_mat = fold[0]
    y_train = fold[1]
    X_test_mat = fold[2]
    y_test = fold[3]
    clf = KernelPegasos(epoch=20,lamb=c,kernel="linear")
    t0 = time.time()
    clf.fit(X_train_mat, y_train)
    t1 = time.time()
    #print('Training time:', t1-t0, 'seconds.\n')
    t0 = time.time()
    y_pred = clf.predict(X_test_mat)
    t1 = time.time()
    #print('Prediction time:', t1-t0, 'seconds.\n')
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy_score (in %): ",acc*100,"\n")
    #print(classification_report(y_test, y_pred))
    p,r,f,_ = score(y_test,y_pred,average=None,labels=[1,-1])
    print('P_pos={:.6f}, P_neg={:.6f}\n'.format(p[0],p[1]))
    print('R_pos={:.6f}, R_neg={:.6f}\n'.format(r[0],r[1]))
    print('F_pos={:.6f}, F_neg={:.6f}\n'.format(f[0],f[1]))
    print("\n\n---------END OF FOLD",str(cnt),"-----------\n\n")
    cnt+=1
    avg_acc+=acc
print("\n\nAverage Accuracy across all folds is : ",avg_acc*100/len(splits))

print("########## Kernel Pegasos (kernel - polynomial) ##########")
avg_acc=0.0
cnt=1
for fold in splits:
    X_train_mat = fold[0]
    y_train = fold[1]
    X_test_mat = fold[2]
    y_test = fold[3]
    clf = KernelPegasos(epoch=20,lamb=c,kernel="polynomial")
    t0 = time.time()
    clf.fit(X_train_mat, y_train)
    t1 = time.time()
    #print('Training time:', t1-t0, 'seconds.\n')
    t0 = time.time()
    y_pred = clf.predict(X_test_mat)
    t1 = time.time()
    #print('Prediction time:', t1-t0, 'seconds.\n')
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy_score (in %): ",acc*100,"\n")
    #print(classification_report(y_test, y_pred))
    p,r,f,_ = score(y_test,y_pred,average=None,labels=[1,-1])
    print('P_pos={:.6f}, P_neg={:.6f}\n'.format(p[0],p[1]))
    print('R_pos={:.6f}, R_neg={:.6f}\n'.format(r[0],r[1]))
    print('F_pos={:.6f}, F_neg={:.6f}\n'.format(f[0],f[1]))
    print("\n\n---------END OF FOLD",str(cnt),"-----------\n\n")
    cnt+=1
    avg_acc+=acc
print("\n\nAverage Accuracy across all folds is : ",avg_acc*100/len(splits))

print("########## Kernel Pegasos (kernel - gaussian) ##########")
avg_acc=0.0
cnt=1
for fold in splits:
    X_train_mat = fold[0]
    y_train = fold[1]
    X_test_mat = fold[2]
    y_test = fold[3]
    clf = KernelPegasos(epoch=20,lamb=c,kernel="gaussian")
    t0 = time.time()
    clf.fit(X_train_mat, y_train)
    t1 = time.time()
    #print('Training time:', t1-t0, 'seconds.\n')
    t0 = time.time()
    y_pred = clf.predict(X_test_mat)
    t1 = time.time()
    #print('Prediction time:', t1-t0, 'seconds.\n')
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy_score (in %): ",acc*100,"\n")
    #print(classification_report(y_test, y_pred))
    p,r,f,_ = score(y_test,y_pred,average=None,labels=[1,-1])
    print('P_pos={:.6f}, P_neg={:.6f}\n'.format(p[0],p[1]))
    print('R_pos={:.6f}, R_neg={:.6f}\n'.format(r[0],r[1]))
    print('F_pos={:.6f}, F_neg={:.6f}\n'.format(f[0],f[1]))
    print("\n\n---------END OF FOLD",str(cnt),"-----------\n\n")
    cnt+=1
    avg_acc+=acc
print("\n\nAverage Accuracy across all folds is : ",avg_acc*100/len(splits))

print("########## Scikit-learn SVM (kernel -> linear) ##########")
avg_acc=0.0
cnt=1
for fold in splits:
    X_train_mat = fold[0]
    y_train = fold[1]
    X_test_mat = fold[2]
    y_test = fold[3]
    clf = SVC(C=1e-3,kernel="linear",gamma=0.02)
    t0 = time.time()
    clf.fit(X_train_mat,y_train)
    t1 = time.time()
    #print('Training time:', t1-t0, 'seconds.\n')
    t0 = time.time()
    y_pred = clf.predict(X_test_mat)
    t1 = time.time()
    #print('Prediction time:', t1-t0, 'seconds.\n')
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy_score (in %): ",acc*100,"\n")
    #print(classification_report(y_test, y_pred))
    p,r,f,_ = score(y_test,y_pred,average=None,labels=[1,-1])
    print('P_pos={:.6f}, P_neg={:.6f}\n'.format(p[0],p[1]))
    print('R_pos={:.6f}, R_neg={:.6f}\n'.format(r[0],r[1]))
    print('F_pos={:.6f}, F_neg={:.6f}\n'.format(f[0],f[1]))
    print("\n\n---------END OF FOLD",str(cnt),"-----------\n\n")
    cnt+=1
    avg_acc+=acc
print("\n\nAverage Accuracy across all folds is : ",avg_acc*100/len(splits))

print("########## Scikit-learn SVM (kernel -> polynomial) ##########")
avg_acc=0.0
cnt=1
for fold in splits:
    X_train_mat = fold[0]
    y_train = fold[1]
    X_test_mat = fold[2]
    y_test = fold[3]
    clf = SVC(C=1e-3,kernel="poly",gamma=0.02)
    t0 = time.time()
    clf.fit(X_train_mat,y_train)
    t1 = time.time()
    #print('Training time:', t1-t0, 'seconds.\n')
    t0 = time.time()
    y_pred = clf.predict(X_test_mat)
    t1 = time.time()
    #print('Prediction time:', t1-t0, 'seconds.\n')
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy_score (in %): ",acc*100,"\n")
    #print(classification_report(y_test, y_pred))
    p,r,f,_ = score(y_test,y_pred,average=None,labels=[1,-1])
    print('P_pos={:.6f}, P_neg={:.6f}\n'.format(p[0],p[1]))
    print('R_pos={:.6f}, R_neg={:.6f}\n'.format(r[0],r[1]))
    print('F_pos={:.6f}, F_neg={:.6f}\n'.format(f[0],f[1]))
    print("\n\n---------END OF FOLD",str(cnt),"-----------\n\n")
    cnt+=1
    avg_acc+=acc
print("\n\nAverage Accuracy across all folds is : ",avg_acc*100/len(splits))

print("########## Scikit-learn SVM (kernel -> gaussian) ##########")
avg_acc=0.0
cnt=1
for fold in splits:
    X_train_mat = fold[0]
    y_train = fold[1]
    X_test_mat = fold[2]
    y_test = fold[3]
    clf = SVC(C=1e-3,kernel="rbf",gamma=0.02)
    t0 = time.time()
    clf.fit(X_train_mat,y_train)
    t1 = time.time()
    #print('Training time:', t1-t0, 'seconds.\n')
    t0 = time.time()
    y_pred = clf.predict(X_test_mat)
    t1 = time.time()
    #print('Prediction time:', t1-t0, 'seconds.\n')
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy_score (in %): ",acc*100,"\n")
    #print(classification_report(y_test, y_pred))
    p,r,f,_ = score(y_test,y_pred,average=None,labels=[1,-1])
    print('P_pos={:.6f}, P_neg={:.6f}\n'.format(p[0],p[1]))
    print('R_pos={:.6f}, R_neg={:.6f}\n'.format(r[0],r[1]))
    print('F_pos={:.6f}, F_neg={:.6f}\n'.format(f[0],f[1]))
    print("\n\n---------END OF FOLD",str(cnt),"-----------\n\n")
    cnt+=1
    avg_acc+=acc
print("\n\nAverage Accuracy across all folds is : ",avg_acc*100/len(splits))

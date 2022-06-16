from utils.MLutils import feature_Extractor
from sklearn.metrics import accuracy_score,precision_score, recall_score
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

NUM_ITER=1

def IterModel(model,train,test):
    acc_sum, pre_sum, rec_sum = 0, 0, 0
    for i in range(NUM_ITER):
        acc, pre, rec = TrainModel(model=model, train=train,test=test, random_state=i)
        acc_sum += acc
        pre_sum += pre
        rec_sum += rec

    mean_acc = acc_sum / NUM_ITER
    mean_pre = pre_sum / NUM_ITER
    mean_rec = rec_sum / NUM_ITER

    print(f'{model} acc:{round(mean_acc,4)} precision: {round(mean_pre,4)} recall: {round(mean_rec,4)}')
    return mean_acc,mean_pre,mean_rec

def TrainModel(model,train,test,random_state=1):
    if model =='SVM':
        classifier = SVC(gamma=0.3,C=0.5)
        # classifier = LinearSVC(random_state=0, C=0.3,tol=1e-5, max_iter=100000)
    elif model =='DecisionTree':
        classifier = DecisionTreeClassifier(random_state=1004, max_depth=4)
    elif model =='RandomForest':
        classifier = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=0)

    else:
        print('No specific model')
        return 0,0,0

    X_train = train[0]
    y_train = train[1]

    X_test = test[0]
    y_test = test[1]
    classifier.fit(X_train, y_train)
  
    y_predicted = classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_predicted)
    precisions = precision_score(y_test, y_predicted, average='weighted', zero_division=True)
    recalls = recall_score(y_test, y_predicted, average='weighted')

    return acc_score,precisions,recalls

    
def feature_base():
    
    train=open('data/train_r.txt','r').read().split('\n')[:-1]
    train_x = []
    train_y = []
    for num,i in enumerate(train):
        complexity,code=i.split('\t')
        train_x.append(np.array(feature_Extractor(source=code, version=1)))
        train_y.append(int(complexity))
    train=([np.array(train_x),train_y])
    print(f'{train[0].shape[0]} train data loaded')

    test=open('data/test_r.txt','r').read().split('\n')[:-1]
    test_x = []
    test_y = []
    for num,i in enumerate(test):
        complexity,code=i.split('\t')
        test_x.append(np.array(feature_Extractor(source=code, version=1)))
        test_y.append(int(complexity))
    test=([np.array(test_x),test_y])
    print(f'{test[0].shape[0]} test data loaded')
    #model training phase


    IterModel(model='SVM',train=train,test=test)
    IterModel(model='DecisionTree', train=train,test=test)
    IterModel(model='RandomForest', train=train,test=test)

feature_base()
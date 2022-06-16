
from utils.MLutils import feature_Extractor
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


NUM_ITER=1

def IterModel(model,train,test):
    result_dict={}
    for i in range(NUM_ITER):
        result=TrainModel(model=model, train=train,test=test, random_state=i)
        for r in result.keys():
            if r in result_dict.keys():
                result_dict[r]+=result[r]
            else:
                result_dict[r]=result[r]
    for r in result_dict.keys():
        result_dict[r]/=NUM_ITER
    return result_dict

def TrainModel(model,train,test,random_state=1):
    if model =='SVM':
        classifier = SVC(gamma=0.3,C=0.5)
    elif model =='DecisionTree':
        classifier = DecisionTreeClassifier(random_state=random_state, max_depth=4)
    elif model =='RandomForest':
        classifier = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=random_state)

    else:
        print('No specific model')
        return 0,0,0

    X_train = train[0]
    y_train = train[1]
    classifier.fit(X_train, y_train)
    acc_result={}

    for i in test.keys():
        acc_score=evaluate(test[i],classifier)
        acc_result[i]=acc_score

    return acc_result
def evaluate(test,classifier):
    X_test = test[0]
    y_test = test[1]
    y_predicted = classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_predicted)
    return acc_score

def feature_base(results,fold):
    
    train=open(f'data/train_{fold}_fold.txt','r').read().split('\n')[:-1]
    train_x = []
    train_y = []
    for num,i in enumerate(train):
        complexity,code=i.split('\t')
        train_x.append(np.array(feature_Extractor(source=code, version=1)))
        train_y.append(int(complexity))
    train=([np.array(train_x),train_y])


    test=open(f'data/test_{fold}_fold.txt','r').read().split('\n')[:-1]
    test_x = []
    test_y = []
    for num,i in enumerate(test):
        complexity,code=i.split('\t')
        test_x.append(np.array(feature_Extractor(source=code, version=1)))
        test_y.append(int(complexity))
    test=([np.array(test_x),test_y])

    length_items=['256','512','1024','over']
    complexity_items=['constant','linear','quadratic','cubic','logn','nlogn','np']

    test_items={'origin':test}
    for item in length_items:
        test=open(f'data/length_split/{item}_test_{fold}_fold.txt','r').read().split('\n')[:-1]
        test_x = []
        test_y = []
        for num,i in enumerate(test):
            complexity,code=i.split('\t')
            test_x.append(np.array(feature_Extractor(source=code, version=1)))
            test_y.append(int(complexity))
        test=([np.array(test_x),test_y])
        test_items[item]=test
        
    for item in complexity_items:
        test=open(f'data/complexity_split/{item}_test_{fold}_fold.txt','r').read().split('\n')[:-1]
        test_x = []
        test_y = []
        for num,i in enumerate(test):
            complexity,code=i.split('\t')
            test_x.append(np.array(feature_Extractor(source=code, version=1)))
            test_y.append(int(complexity))
        test=([np.array(test_x),test_y])
        test_items[item]=test

    #model training phase
    tmp_result=IterModel(model='RandomForest', train=train,test=test_items)

    for t in tmp_result.keys():
        if t in results.keys():
            results[t].append(tmp_result[t])
        else:
            results[t]=[tmp_result[t]]
results={}

for i in range(5):
    feature_base(results,fold=i)

for i in results.keys():
    print(f'{i} : {np.mean(results[i])} std : {np.std(results[i])}')
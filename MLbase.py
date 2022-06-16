from utils.MLutils import feature_Extractor
from sklearn.metrics import accuracy_score,precision_score, recall_score
import numpy as np

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
import torch

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
    elif model =='KNN':
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif model =='Kmeans':
        classifier = KMeans(n_clusters=5, random_state=0)
    elif model =='DecisionTree':
        classifier = DecisionTreeClassifier(random_state=1004, max_depth=4)
    elif model =='LogisticRegression':
        classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    elif model =='NaiveBayse':
        classifier = BernoulliNB()
    elif model =='RandomForest':
        classifier = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=0)
    elif model =='MLP':
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)
    elif model == 'torch':
        classifier = torch_model()
    else:
        print('No specific model')
        return 0,0,0

    # DataDict=generate_data(data,random_state=random_state)
    X_train = train[0]
    y_train = train[1]

    X_test = test[0]
    y_test = test[1]

    if model =='Kmeans':
        classifier.fit(X_train)
    elif model == 'torch':
        torch_train(classifier,X_train,y_train)
    else:
        classifier.fit(X_train, y_train)
    if model =='torch':
        classifier.eval()
        y_predicted =classifier.predict(torch.FloatTensor(X_test))
        y_predicted=y_predicted.numpy()
    else:
        y_predicted = classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_predicted)
    precisions = precision_score(y_test, y_predicted, average='weighted', zero_division=True)
    recalls = recall_score(y_test, y_predicted, average='weighted')

    return acc_score,precisions,recalls
def torch_train(model,X,Y):
    
    criterion = nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters())
    train_set = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.LongTensor(Y))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

    for _ in range(1000):
        for (x,y) in train_dataloader:
            
            model.zero_grad()
            pred_y=model(x)
            
            loss=criterion(pred_y,y)
            loss.backward()
            # print(loss)
            optimizer.step()
    
def feature_base():
    
    train=open('data/train_r.txt','r').read().split('\n')[:-1]
    #feature(13) + compleixity(1) - mask(3)
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
    # KNN_acc, KNN_pre, KNN_rec, KNN_acc2, KNN_pre2, KNN_rec2 = IterModel(model='KNN', data=data,data2=data2)
    # Kmeans_acc, Kmeans_pre, Kmeans_rec, Kmeans_acc2, Kmeans_pre2, Kmeans_rec2 = IterModel(model='Kmeans', data=data,data2=data2)
    IterModel(model='DecisionTree', train=train,test=test)
    # LR_acc, LR_pre, LR_rec, LR_acc2, LR_pre2, LR_rec2 = IterModel(model='LogisticRegression', data=data,data2=data2)
    # Naibe_acc, Naibe_pre, Naibe_rec, Naibe_acc2, Naibe_pre2, Naibe_rec2 = IterModel(model='NaiveBayse', data=data,data2=data2)
    IterModel(model='RandomForest', train=train,test=test)
    # MLP_acc, MLP_pre, MLP_rec, MLP_acc2, MLP_pre2, MLP_rec2 = IterModel(model='MLP', data=data,data2=data2)
    # IterModel(model='torch',train=train,test=test)

    # print(f'data1 acc:{torch_acc} precision: {torch_pre} recall: {torch_rec}')
    # print(f'data1 acc:{torch_acc2} precision: {torch_pre2} recall: {torch_rec2}')
    

feature_base()
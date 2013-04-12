import scipy
import numpy as np
import operator
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.svm import SVC

# loading csv data into numpy array
def read_data(f, header=True, test=False, rows=0):
    data = []
    labels = []

    csv_reader = csv.reader(open(f, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index = index + 1
        if rows > 0 & index > rows:
            break
        if header and index == 1:
            continue

        if not test:
            labels.append(int(row[0]))
            row = row[1:]

        data.append(np.array(np.int64(row)))
    return (data, labels)

def predictKNN(train,labels,test):
    print 'start knn'
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train, labels) 
    probabilities = knn.predict_proba(test)
    predictions = knn.predict(test)
    bestScores = probabilities.max(axis=1)
    print 'done with knn'
    return predictions, bestScores

def predictSVC(train, labels, test):
    print 'start SVC'
    clf = SVC(probability=True)
    clf.fit(train, labels)
    svc_predictions = clf.predict(test)
    svc_probs = clf.predict_proba(test)
    svc_bestProbs = svc_probs.max(axis=1)
    print 'svc done!'
    return svc_predictions, svc_bestProbs

def predictRF(train, labels, test, tmpl):
    print 'predicting...'
    rf = RandomForestClassifier(n_estimators=200, n_jobs=2)
    rf.fit(train, labels)
    print 'done fitting...'
    rf_predictions = rf.predict(test)
    rf_probs = rf.predict_proba(test)
    rf_BestProbs = rf_probs.max(axis=1)
    print('done with random forest.  Save text!')
    return rf_predictions, rf_BestProbs

class PredScore:
    def __init__(self,prediction,score):
        self.prediction = prediction
        self.score = score
    prediction = -1
    score = 0

if __name__ == '__main__':
    print 'read data!'
    #only use the below for initial creation of npy files
#    train, labels = read_data("train.csv", rows=1000)
#    np.save('train_small.npy', train)
#    np.save('labels_small.npy', labels)

#    train = np.load('train_small.npy')
#    labels = np.load('labels_small.npy')

    train = np.load('train.npy')
    labels = np.load('labels.npy')
#    
    print 'done reading train'
    
    #only use the below for the initial creation of npy files.
#    test, tmpl = read_data("test.csv", test=True, rows=1000)
#    np.save('test_small.npy', test)
#    np.save('tmpl_small.npy', tmpl)

#    test = np.load('test_small.npy')
#    tmpl = np.load('tmpl_small.npy')
    test = np.load('test.npy')
    tmpl = np.load('tmpl.npy')
    print 'done reading test!'
    rfPredictions, rfScore = predictRF(train, labels, test, tmpl)
    knnPredictions, knnScore = predictKNN(train,labels, test)
    svcPredictions, svcScore = predictSVC(train, labels, test)
    retArray = []
    index = 0
    for rf in rfScore:
        rfPredScore = PredScore(rfPredictions[index],rfScore[index])
        knnPredScore = PredScore(knnPredictions[index],knnScore[index])
        svcPredScore = PredScore(svcPredictions[index],svcScore[index])
        
        options = []
        options.append(rfPredScore)
        options.append(knnPredScore)
        options.append(svcPredScore)
        
        maxObj = max(options,key=operator.attrgetter('score'))
        retArray.append(maxObj.prediction)
        index = index + 1
    np.savetxt('submission.csv', retArray, delimiter=',',fmt='%i')
    print 'done!!!'
    
    
    
from imports.imports import *

class ClassifiersAcc:

    # Testing Decision Tree accuracy using Gridsearch
    def decisionTree(self, X_train, x_test,  y_train, y_test, cv=5):
        parameters = {'max_depth':range(3,20)}

        scores = ['precision', 'recall']

        print("\nDecision Tree\n")

        for score in scores:
            clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=cv)
            clf.fit(X_train, y_train)
            decisionTree = clf.best_estimator_
            print ("Best score: ", clf.best_score_, ', Max depth: ',  clf.best_params_['max_depth']) 

            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%f (%f) with: %r" % (mean, std, params))
            print()
            
        decisionTree = DecisionTreeClassifier(max_depth = clf.best_params_['max_depth']).fit(X_train, y_train) 
        decisionTreePredictions = decisionTree.predict(x_test) 
        
        # creating a confusion matrix 
        cm = confusion_matrix(y_test, decisionTreePredictions) 
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Decision Tree\n Accuracy:{0:2f}".format(accuracy_score(y_test, decisionTreePredictions)))
        plt.ylabel("correct value")
        plt.xlabel("predicted value")
        plt.show()

    def svcParamSelection(self, X_train, x_test, y_train, y_test, cv=5):
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = ['precision', 'recall']

        for score in scores:
            
            clf = GridSearchCV(SVC(), tuned_parameters, cv=cv, scoring='%s_macro' % score)
            clf.fit(X_train, y_train)
            print("\nSVC\n")
            print("Best: %f using %s" % (clf.best_score_, clf.best_params_))
           
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%f (%f) with: %r" % (mean, std, params))
            print()
        clf = SVC(kernel='rbf', C=10, gamma=0.001)
        clf.fit(X_train, y_train)
        svcPrediction = clf.predict(x_test)

         # creating a confusion matrix 
        cm = confusion_matrix(y_test, svcPrediction) 
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("SVC\n Accuracy:{0:2f}".format(accuracy_score(y_test, svcPrediction)))
        plt.ylabel("correct value")
        plt.xlabel("predicted value")
        plt.show()

    
    # Testing Random Forest accuracy using Gridsearch
    def randomForest(self, X_train, x_test,  y_train, y_test):
        parameters = {'max_depth':range(3,20)}

        scores = ['precision', 'recall']

        for score in scores:
            clf = GridSearchCV(RandomForestClassifier(), parameters)
            clf.fit(X_train, y_train)
            reandomForest = clf.best_estimator_
            print("\nRandom Forest\n")
            print ("Best score: ", clf.best_score_, ', Max depth: ',  clf.best_params_['max_depth']) 

            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%f (%f) with: %r" % (mean, std, params))
            print()
            
        reandomForestM = RandomForestClassifier(max_depth = clf.best_params_['max_depth']).fit(X_train, y_train) 
        reandomForestPred = reandomForestM.predict(x_test) 
        
        # creating a confusion matrix 
        cm = confusion_matrix(y_test, reandomForestPred) 
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Random Forest Confusion Matrix\n Accuracy:{0:3f}".format(accuracy_score(y_test, reandomForestPred)))
        plt.ylabel("correct value")
        plt.xlabel("predicted value")
        plt.show()


        # parameters = {'max_depth':range(3,20)}
        # clf = GridSearchCV(RandomForestClassifier(), parameters)
        # clf.fit(X_train, y_train)
        # forest_model = clf.best_estimator_
        # print ("Best score: ", clf.best_score_, ', Max depth: ',  clf.best_params_['max_depth']) 
        # rforest_model = RandomForestClassifier(max_depth = 2).fit(X_train, y_train) 
        # rforest_predictions = rforest_model.predict(x_test) 
        
        # # creating a confusion matrix 
        # cm = confusion_matrix(y_test, rforest_predictions) 
        # sns.heatmap(cm, annot=True, fmt="d")
        # plt.title("Rforest_Confusion_Matrix \nAccuracy:{0:.3f}".format(accuracy_score(y_test, rforest_predictions)))
        # plt.ylabel("Actual value")
        # plt.xlabel("Predicted value")
        
        # plt.show()

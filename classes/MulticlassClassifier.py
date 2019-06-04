from imports.imports import *

class MulticlassClassifier:
    
    def dataSplit(self, mode, hyperparameter='default'):
        dataframe = pd.read_csv("data\wine.csv", header=None)
        dataset = dataframe.values

        Y = dataset[:,0:1]

        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y[:, 0])

        X = dataframe.iloc[:,1:14].values
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        if mode=='validation' and hyperparameter=='default':
            print(mode, ', ' , hyperparameter)
            self.crossValidation(x_train, y_train)
        elif mode=='accuracy' and hyperparameter!='default':
            self.calculateAccuracy(x_train, x_test, y_train, y_test, hyperparameter)
        else:
            print('Wrong combination of parameters')
        

    def buildNeuralNetwork(self, neurons=5, activation='relu', optimizer = 'adam', dropout_rate=0.2):
        classifier = Sequential()
        
        classifier.add(Dense(10, input_dim=13, activation='relu'))
        classifier.add(Dropout(dropout_rate))
        classifier.add(Dense(5, activation='relu'))
        classifier.add(Dropout(dropout_rate))
        classifier.add(Dense(3, activation='sigmoid'))

        classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                        metrics=['accuracy'])
        return classifier

    def calculateAccuracy(self, X_train, x_test, y_train, y_test, hyperparameter):

        classifier = KerasClassifier(build_fn=self.buildNeuralNetwork, dropout_rate=0.2)

        if hyperparameter == 'batchSizeAndEpochs':
            #Batch Size and Epochs accuracy
            hyperAcc = HyperparametersAccuracy()
            hyperAcc.batchSizeAndEpochsAcc(X_train, y_train, classifier, [10, 33, 50], [10, 20, 100, 500, 700, 1000])
        if hyperparameter == 'optimizationAlgorithm':
            #Batch optimization algorithm accuracy
            hyperAcc = HyperparametersAccuracy()
            hyperAcc.optimizationAlgorithmAcc(X_train, y_train, classifier, 50, 500, ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])
        if hyperparameter == 'activationFunction':
            #Neuron activation function accuracy
            hyperAcc = HyperparametersAccuracy()
            hyperAcc.neuronActivationFunctionAcc(X_train, y_train, classifier, 50, 500, ['softmax', 'relu', 'tanh', 'sigmoid', 'linear'])
        if hyperparameter == 'dropoutRate':
            #Dropout rate
            hyperAcc = HyperparametersAccuracy()
            hyperAcc.dropoutRegularizationAcc(X_train, x_test, y_train, y_test, classifier, 50, 500, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        if hyperparameter == 'neurons':
            #Neurons in hidden layers
            hyperAcc = HyperparametersAccuracy()
            hyperAcc.numberOfNeuronsInHiddenLayer(X_train, y_train, classifier, 50, 500, [1, 5, 10, 15, 20, 25, 30])
        if hyperparameter == 'decisionTree':
            #Decision Tree
            classifier = ClassifiersAcc()
            classifier.decisionTree(X_train, x_test, y_train, y_test)
        if hyperparameter == 'svc':
            classifier = ClassifiersAcc()
            classifier.svcParamSelection(X_train, x_test, y_train, y_test)
        if hyperparameter == 'randomForest':
            classifier = ClassifiersAcc()
            classifier.randomForest(X_train, x_test, y_train, y_test)
    

    def crossValidation(self, x_train, y_train):                
        classifier = KerasClassifier(build_fn=self.buildNeuralNetwork,
                                     batch_size=50, epochs=100)
        cv_accuracy = cross_val_score(estimator=classifier,
                                    X=x_train, y=y_train, cv=5, n_jobs=None)
        mean = cv_accuracy.mean()
        variance = cv_accuracy.std()
        print('Accuracy: ', cv_accuracy)
        print('Mean: ', mean)
        print('Variance: ', variance)




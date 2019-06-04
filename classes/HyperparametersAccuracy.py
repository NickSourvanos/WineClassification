from imports.imports import *

# Tunning Hyperparameters
class HyperparametersAccuracy:

    # Tuning batch size and number of epochs using GridSearch
    def batchSizeAndEpochsAcc(self, X_train, y_train, classifier, batchSize, epochs, cv=5):
        parameters = {'batch_size': batchSize,
                    'epochs': epochs}
        grid_search = GridSearchCV(estimator=classifier,
                                param_grid=parameters,
                                scoring='accuracy',
                                cv=cv)
        grid_search = grid_search.fit(X_train, y_train)

        #summarize results
        print("\nBatch Size and Epochs\n")
        print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        params = grid_search.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    
    # Tuning the training otpimization algorithm using GridSearch
    def optimizationAlgorithmAcc(self, X_train, y_train, classifier, batchSize, epochs, optimizers, cv=5):
        parameters = {'batch_size': batchSize,
                    'epochs': epochs}
        grid_search = GridSearchCV(estimator=classifier,
                                param_grid=dict(optimizer=optimizers),
                                cv=cv)
        grid_search = grid_search.fit(X_train, y_train)

        #summarize results
        print("\nOptimization Algorithm\n")
        print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        params = grid_search.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    # Tuning the neuron activation function
    def neuronActivationFunctionAcc(self, X_train, y_train, classifier, batchSize, epochs, activation, cv=5):
        parameters = {'batch_size': batchSize,
                    'epochs': epochs}
        grid_search = GridSearchCV(estimator=classifier,
                                param_grid=dict(activation=activation),
                                cv=cv)
        grid_search = grid_search.fit(X_train, y_train)

        #summarize results
        print("\nNeuron Activation Function\n")
        print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        params = grid_search.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    
    # Tuning dropout regularization
    def dropoutRegularizationAcc(self, X_train, y_train, classifier, batchSize, epochs, dropoutRates, cv=5):
        parameters = {'batch_size': batchSize,
                    'epochs': epochs}
        grid_search = GridSearchCV(estimator=classifier,
                                param_grid=dict(dropout_rate=dropoutRates),
                                cv=cv)
        grid_search = grid_search.fit(X_train, y_train)

        #summarize results
        print("\nDropout Regularization Accuracy\n")
        print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        params = grid_search.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    
    def numberOfNeuronsInHiddenLayer(self, X_train, y_train, classifier, batchSize, epochs, neurons, cv=5):
        parameters = {'batch_size': batchSize,
                    'epochs': epochs}
        grid_search = GridSearchCV(estimator=classifier,
                                param_grid=dict(neurons=neurons),
                                cv=cv)
        grid_search = grid_search.fit(X_train, y_train)

        #summarize results
        print("\nNumber of neurons in Hidden Layer\n")
        print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        params = grid_search.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    
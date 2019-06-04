from imports.imports import *

#------------------Change Parameter for different Plots------------------#
# histogram', 'corMatrix', 'scatter', 'skew', 'box'

plottingClass = PlottingClass()
dataset = plottingClass.getDataset()

plottingClass.getAccuracyFromClassifiersUsed()

# plottingClass.getPlot(dataset, 'corMatrix')




#------------------Change dataSplit parameter------------------#

#validation: cross validation
#accuracy: calculates best accuracy




#-----------------------Select parameter for Hyperparameter Testing(NN)------------------------#

#Batch Size and Epochs accuracy parameter -> batchSizeAndEpochs
#Batch optimization algorithm accuracy parameter -> optimizationAlgorithm
#Neuron activation function accuracy parameter -> activationFunction
#Dropout rate accuracy parameter -> dropoutRate
#Neurons in hidden layers accuracy parameter -> neurons


# multiclassClassifier.dataSplit('accuracy', 'batchSizeAndEpochs')




#-----------------------Select parameter for Hyperparameter Testing(Decision Tree)------------------------#

# Get decision Tree accuracy
# multiclassClassifier = MulticlassClassifier()
# multiclassClassifier.dataSplit('validation')
# multiclassClassifier.dataSplit('accuracy', 'batchSizeAndEpochs')
# multiclassClassifier.dataSplit('accuracy', 'optimizationAlgorithm')
# multiclassClassifier.dataSplit('accuracy', 'activationFunction')
# multiclassClassifier.dataSplit('accuracy', 'dropoutRate')
# multiclassClassifier.dataSplit('accuracy', 'neurons')
# multiclassClassifier.dataSplit('accuracy', 'decisionTree')
# multiclassClassifier.dataSplit('accuracy', 'svc')
# multiclassClassifier.dataSplit('accuracy', 'randomForest')
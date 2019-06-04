from imports.imports import *

class PlottingClass:

    namesList = ['Category', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
                        'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                        'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    
    # Get dataset
    def getDataset(self):
        dataframe = pd.read_csv("data\wine.csv", names = self.namesList)
        return dataframe

    # Choose type of plot
    def getPlot(self, dataset, plotType):
        if plotType == 'histogram':
            self.getHistogram(dataset)
        elif plotType == 'corMatrix':
            self.getCorrelationMatrix(dataset)
        elif plotType == 'scatter':
            self.getScatterPlot(dataset)
        elif plotType == 'skew':
            self.calculateSkew(dataset)
        elif plotType == 'box':
            self.getBoxPLot(dataset)
        else:
            print("Wrong argument")

    # checking our dataset's shape, type, missing values, null values
    def datasetInfo(self, dataset):
        print('\n----------------------DATA INFO----------------------\n')
        print(dataset.info(), '\n')
        print('\n----------------------DATA DESCRIPTION----------------------\n')
        print(dataset.describe())
        print('\n----------------------TOP 10 ROWS----------------------\n')
        print(dataset.head(10))    

    # a brief check of our labels and its values
    def getHistogram(self, dataset):
        fig = plt.figure(figsize=(20,15))
        cols = 5
        rows = math.ceil(float(dataset.shape[1]) / cols)
        for i, column in enumerate(dataset.columns):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title(column)
            if dataset.dtypes[column] == np.object:
                data[column].value_counts().plot(kind="bar", axes=ax)
            else:
                dataset[column].hist(axes=ax)
                plt.xticks(rotation="vertical")
            
        plt.subplots_adjust(hspace=0.7, wspace=0.2)
        plt.savefig('plots\Histograms.png')
        plt.show()


    # check correlation between features
    def getCorrelationMatrix(self, dataset):
        #Plot a correlation matrix with a heat map
        fig = plt.figure(figsize=(12,7))
        sns.heatmap(dataset.corr(),cmap="BrBG_r",annot=True,linecolor='white')
        plt.title('Correlation between Features ')
        plt.savefig('plots\Heatmap.png')
        plt.show()
   
    # check relations of values through the scatter plot    
    def getScatterPlot(self, dataset):
        sns.pairplot(dataset, hue='Category')
        plt.savefig('plots\Scatter.png')
        plt.show()

    # measure distribution symmetry (negetive(mode>mean), normal, positive(mode<mean))
    def calculateSkew(self, dataset):
        kurtosis = dataset.kurtosis()
        skew = dataset.skew()
        
        print( 'Excess kurtosis of distribution (should be 0):', ' \n', (kurtosis), '\n')
        print( 'Skewness of distribution (should be 0):', '\n', (skew), '\n')

    # median, qurtiles (first / third), minimum - maximum values
    def getBoxPLot(self, dataset):
        fig = plt.figure(figsize=(40,30))

        for i in range (1, 14):
            ax1 =fig.add_subplot(3,6,i+1)
            sns.boxplot(x = "Category", y = self.namesList[i], data = dataset, palette = "GnBu_d")
        plt.savefig('plots\BoxPlot.png')
        plt.show()

    def getAccuracyFromClassifiersUsed(self):
        with open('data\classification_scores.json', 'r')as file:
            data = json.load(file)
        data = pd.DataFrame(data['scores'])
        sns.factorplot('classifier', 'accuracy', data=data, kind="bar", palette="muted", legend=False)
        plt.savefig('plots\classificationScores.png')
        plt.show()




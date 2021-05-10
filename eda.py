
# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno # plot missing value counts

# ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Common imports
import numpy as np
import pandas as pd
import math
import os


class Eda:
    '''This class creates some essential plots required to get 
    a priliminary understanding of data.
    input: 
    csv_file: It takes in tabular data in form of a CSV file format
    
    output: it generates plots in the .png file format
    '''
    
    def __init__(self, DATA_PATH, cols=5, width=20, height=15):
        self.data_path = DATA_PATH
        self.cols = cols
        self.width = width
        self.height = height
        
    def load_data(self):
        csv_path = os.path.join(self.data_path, "housing.csv")
        dataset = pd.read_csv(csv_path)
        return dataset
    
    # Let’s plot the distribution of each feature
    def plot_distribution(self, hspace=0.2, wspace=0.5):
        """This function draws standard distributions for 
        numerical and categorical variable"""
        dataset=self.load_data()
        print(dataset.shape)
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(self.width,self.height))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
        rows = math.ceil((dataset.shape[1]) / self.cols)
        for i, column in enumerate(dataset.columns):
            ax = fig.add_subplot(rows, self.cols, i + 1)
            ax.set_title(column)
            if dataset.dtypes[column] == np.object:
                g = sns.countplot(y=column, data=dataset)
                substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
                g.set(yticklabels=substrings)
                plt.xticks(rotation=25)
            else:
                g = sns.distplot(dataset[column])
                plt.xticks(rotation=25)
        # save_fig("feature_distribution")
        
    def plot_missingvalues(self):
        dataset=self.load_data()
        missingno.matrix(dataset, figsize = (12,4))
        
    def get_correlation(self):
        dataset=self.load_data()
        corr_matrix = dataset.corr()
        return corr_matrix
        
        
    def plot_correlation(self):
        corr_matrix = self.get_correlation()
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(
            corr_matrix, 
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        ax.set_title('Correlation Plot', fontsize=14)
        
    def get_scatterplot_matrix(self):
        dataset=self.load_data()
        sns.set_theme(style="ticks")
        sns.pairplot(dataset, diag_kind='kde')


#     def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#         path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#         print("Saving figure", fig_id)
#         if tight_layout:
#             plt.tight_layout()
#         plt.savefig(path, format=fig_extension, dpi=resolution)

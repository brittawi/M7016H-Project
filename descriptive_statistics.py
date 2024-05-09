import pandas as pd
import matplotlib.pyplot as plt


class DiabetesDataBase:
    def __init__(self, csv_path):
        self.diabetes_df = self._load_data(csv_path)
        self.header_list = self.diabetes_df.columns.tolist()
        
    
    def _load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        #Which values can't have 0?, How to handle missing values?
        no_zero_list = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Age"]
        cleansed_df = df.loc[(df[no_zero_list] != 0).all(axis=1)]
        return cleansed_df
    
    
    def describe_data(self):
        for col_name, col in self.diabetes_df.items():
            print(col_name)
            print(col.describe())
            print("\n––––––––––––––––––––––––––––––––––\n")
            
    def plot_histogram_summary(self):
        fig, ax = plt.subplots(2, 4, figsize=(8,4))
        self.diabetes_df.iloc[:, :8].hist(ax=ax, bins = 8)
        plt.tight_layout()
        plt.show()
        
                    
    def plot_boxplot_summary(self):
        fig, ax = plt.subplots(2, 4, figsize=(8, 4))
        for idx in range(2):
            for idy in range(4):
                ax[idx, idy].boxplot(self.diabetes_df.iloc[:,idx+idy], 
                                     vert=False,
                                     )
                ax[idx, idy].set_title(self.header_list[idx+idy])
                ax[idx, idy].set(yticklabels=[])
        plt.tight_layout()
        plt.show()
        
    def plot_histogram_individual(self):
        for i in range(8):
            plt.hist(self.diabetes_df.iloc[:,i], bins = 8)
            plt.title(self.header_list[i])
            plt.show()
            
    def plot_boxplot_individual(self):
        for i in range(8):
            plt.boxplot(self.diabetes_df.iloc[:,i], vert=False)
            plt.title(self.header_list[i])
            plt.show()
        
    
csv_path = "diabetes.csv"
ddb = DiabetesDataBase(csv_path)
ddb.describe_data()
ddb.plot_histogram_summary()
ddb.plot_boxplot_summary()
ddb.plot_histogram_individual()
ddb.plot_boxplot_individual()
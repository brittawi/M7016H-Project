import pandas as pd
import matplotlib.pyplot as plt


class DiabetesDataBase:
    def __init__(self, csv_path):
        self.diabetes_df = self._load_data(csv_path)
        self.header_list = self.diabetes_df.columns.tolist()
        
    
    def _load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        print(f'Shape before removing values that are 0: {df.shape}')
        #Which values can't have 0?, How to handle missing values?
        no_zero_list = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Age"]
        cleansed_df = df.loc[(df[no_zero_list] != 0).all(axis=1)]
        print(f'Shape after removing values that are 0: {cleansed_df.shape}')
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
        for idy in range(4):
            ax[0, idy].boxplot(self.diabetes_df.iloc[:,idy], 
                                    vert=False,
                                    )
            ax[0, idy].set_title(self.header_list[idy])
            ax[0, idy].set(yticklabels=[])
        for idy in range(4):
            ax[1, idy].boxplot(self.diabetes_df.iloc[:,idy+4], 
                                    vert=False,
                                    )
            ax[1, idy].set_title(self.header_list[idy+4])
            ax[1, idy].set(yticklabels=[])
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
        
if __name__ == '__main__':
    csv_path = "diabetes.csv"
    ddb = DiabetesDataBase(csv_path)
    ddb.describe_data()
    ddb.plot_histogram_summary()
    ddb.plot_boxplot_summary()
    #ddb.plot_histogram_individual()
    #ddb.plot_boxplot_individual()
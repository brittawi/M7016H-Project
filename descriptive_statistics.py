import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
        counter = 0
        for idx in range(2):
            for idy in range(4):
                ax[idx, idy].boxplot(self.diabetes_df.iloc[:,counter], 
                                        vert=False,
                                        )
                ax[idx, idy].set_title(self.header_list[counter])
                ax[idx, idy].set(yticklabels=[])
                counter += 1
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
            
    def show_label_balance(self):
        label_0 = len(self.diabetes_df[(self.diabetes_df["Outcome"]==0)])
        label_1 = len(self.diabetes_df[(self.diabetes_df["Outcome"]==1)])
        
        # plot distribution of ones and zeros
        labels = ["0", "1"]
        labels_num = [0,1]

        plt.bar(labels_num, [label_0,label_1])
        plt.xticks(labels_num, labels)
        plt.title("Distribution of labels in dataset")
        plt.show()
        
    def plot_connection_to_outcome(self, col_to_compare, reindexing=None):
        grouped_data = self.diabetes_df[[col_to_compare,"Outcome","DiabetesPedigreeFunction"]].groupby([col_to_compare,"Outcome"]).count().unstack("Outcome")
        
        if reindexing != None:
            grouped_data = grouped_data.reindex(reindexing, level=col_to_compare)

        labels = grouped_data.columns.get_level_values(1)

        grouped_data.plot.bar(figsize=(10,8))
        plt.xlabel(col_to_compare.capitalize()) 
        plt.ylabel("Number of entries") 
        plt.legend(labels, title="Diabetes label")
        plt.title(f"Number of entries with/without diabetes compared to {col_to_compare.lower()}") 
        plt.show()
        
    # TODO normalizing data??
        
    def splitData(self, train=0.8, val=0.1, test=0.1):
        
        # TODO not sure if this is the right way to do
        if train + val + test != 1:
            return ValueError("The percentage of train, val and test percentage has to add up to 1")
        X = self.diabetes_df.iloc[:, :-1].to_numpy()
        y = self.diabetes_df["Outcome"].to_numpy()
        
        # splitting into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=1)
        # splitting train set into train and validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val/train, random_state=1) # TODO not sure about this val/train
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
        
        
if __name__ == '__main__':
    csv_path = "diabetes.csv"
    ddb = DiabetesDataBase(csv_path)
    #ddb.describe_data()
    #ddb.plot_histogram_summary()
    #ddb.plot_boxplot_summary()
    #ddb.plot_histogram_individual()
    #ddb.plot_boxplot_individual()
    #ddb.show_label_balance()
    # TODO plots
    # ddb.plot_connection_to_outcome("Age")
    # ddb.plot_connection_to_outcome("BloodPressure")
    # ddb.plot_connection_to_outcome("Glucose")
    # ddb.plot_connection_to_outcome("Insulin")
    # ddb.plot_connection_to_outcome("BMI")
    X_train, X_val, X_test, y_train, y_val, y_test = ddb.splitData()
    print(len(X_train))
    print(len(X_val))
    print(len(X_test))
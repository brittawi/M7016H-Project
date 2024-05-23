import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


class DiabetesDataBase:
    def __init__(self, csv_path, train_split = 0.8, val_split = 0.1, test_split = 0.1, random_state = 42):
        if train_split + val_split + test_split != 1:
            return ValueError("The percentage of train, val and test percentage has to add up to 1")
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        # read in data
        self.diabetes_df = pd.read_csv(csv_path)
        self.header_list = self.diabetes_df.columns.tolist()
        
        #preprocess data
        self.train_val_set, self.test_set = self._preprocess_data()
        
    def _preprocess_data(self):
        
        # shuffle the dataset
        df = self.diabetes_df.sample(frac=1, random_state=self.random_state) 
        
        # split data into training/val and test set as training and val set have the same preprocessing
        train_val_size = int(df.shape[0]*(self.train_split+self.val_split))
        train_val = df[0:train_val_size]
        test = df[train_val_size:]
        
        # handle missing values
        no_zero_list = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"]
        test = test.loc[(test[no_zero_list] != 0).all(axis=1)] # delete rows with missing values in test set
        # TODO giving a warning but found no way to fix, inplace is not working for me
        train_val[no_zero_list] = train_val[no_zero_list].replace(0, np.NaN) # convert zeros to NaN as this is easier to work with
        train_val = train_val.fillna(train_val.mean(skipna=True)) # replace NaNs with mean from column
        
        return train_val, test
        
    
    def describe_data(self, df):
        # for col_name, col in df.items():
        #     print(col_name)
        #     print(col.describe())
        #     print("\n––––––––––––––––––––––––––––––––––\n")
        print(df.describe())
        print(f'Null values: {df.isnull().values.any()}')
        print(f'Number of zeros for the column Insulin: {(df["Insulin"]== 0).astype(int).sum()}')
            
    def plot_histogram_summary(self, df):
        fig, ax = plt.subplots(2, 4, figsize=(8,4))
        df.iloc[:, :8].hist(ax=ax, bins = 8)
        plt.tight_layout()
        plt.show()
        
                    
    def plot_boxplot_summary(self, df):
        header_list = df.columns.tolist()
        fig, ax = plt.subplots(2, 4, figsize=(8, 4))
        counter = 0
        for idx in range(2):
            for idy in range(4):
                ax[idx, idy].boxplot(df.iloc[:,counter], 
                                        vert=False,
                                        )
                ax[idx, idy].set_title(header_list[counter])
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
            
    def show_label_balance(self, df):
        label_0 = len(df[(df["Outcome"]==0)])
        label_1 = len(df[(df["Outcome"]==1)])
        
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
        
    def splitData(self):
        
        # test data
        X_test = self.test_set.iloc[:, :-1].to_numpy()
        y_test = self.test_set["Outcome"].to_numpy()
        
        # train_val data
        X = self.train_val_set.iloc[:, :-1].to_numpy()
        y = self.train_val_set["Outcome"].to_numpy()
        
        # splitting train_val set into train and validation set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_split/self.train_split, random_state=self.random_state) # TODO not sure about this val/train
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
        
        
if __name__ == '__main__':
    csv_path = "diabetes.csv"
    ddb = DiabetesDataBase(csv_path)
    # analyse and describe Data before the split
    ddb.describe_data(ddb.diabetes_df)
    ddb.plot_histogram_summary(ddb.diabetes_df)
    ddb.plot_boxplot_summary(ddb.diabetes_df)
    #ddb.plot_histogram_individual()
    #ddb.plot_boxplot_individual()
    # TODO plots
    ddb.plot_connection_to_outcome("Age")
    ddb.plot_connection_to_outcome("BloodPressure")
    # ddb.plot_connection_to_outcome("Glucose")
    # ddb.plot_connection_to_outcome("Insulin")
    # ddb.plot_connection_to_outcome("BMI")
    ddb.show_label_balance(ddb.diabetes_df)
    
    # analyse and describe data after the split for train_val
    ddb.describe_data(ddb.train_val_set)
    ddb.plot_histogram_summary(ddb.train_val_set)
    ddb.plot_boxplot_summary(ddb.train_val_set)
    ddb.show_label_balance(ddb.train_val_set)
    
    # analyse and describe data after the split for test
    ddb.describe_data(ddb.test_set)
    ddb.plot_histogram_summary(ddb.test_set)
    ddb.plot_boxplot_summary(ddb.test_set)
    ddb.show_label_balance(ddb.test_set)
    
    # split the data
    X_train, X_val, X_test, y_train, y_val, y_test = ddb.splitData()
    print(len(X_train))
    print(len(X_val))
    print(len(X_test))
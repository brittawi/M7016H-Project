import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


class DiabetesDataBase:
    def __init__(self, csv_path, train_split = 0.8, val_split = 0.1, test_split = 0.1, random_state = 42, augment = False):
        if train_split + val_split + test_split != 1:
            return ValueError("The percentage of train, val and test percentage has to add up to 1")
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        self.augment = augment
        
        # read in data
        self.diabetes_df = pd.read_csv(csv_path)
        self.header_list = self.diabetes_df.columns.tolist()
        
        #preprocess data
        self.train_set,self.val_set, self.test_set = self._preprocess_data()
        
    def _preprocess_data(self):
        
        df = self.diabetes_df.copy()
        
        # replace missing values with NaN so that they are not used for any calculations
        no_zero_list = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"]
        df[no_zero_list] = df[no_zero_list].replace(0, np.NaN) # TODO this gives a warning!
        
        # compute IQR
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3-Q1
        
        # handle outliers, we decided to only handle upper outliers for some columns
        # Skinthickness
        #df = df.drop(df[df.SkinThickness > 80].index)
        df = df.drop(df[df.SkinThickness > (Q3["SkinThickness"] + IQR["SkinThickness"] * 1.5)].index)
        # Insulin
        #df = df.drop(df[df.Insulin > 600].index)
        df = df.drop(df[df.Insulin > (Q3["Insulin"] + IQR["Insulin"] * 1.5)].index)
        # Bloodpressure
        df = df.drop(df[df.BloodPressure > (Q3["BloodPressure"] + IQR["BloodPressure"] * 1.5)].index)
        
        # shuffle the dataset
        df = df.sample(frac=1, random_state=self.random_state) 
        
        # split data into training, val and test set as training and val set have the same preprocessing
        train_size = int(df.shape[0]*(self.train_split))
        val_size = int(df.shape[0]*(self.val_split)) + train_size
        train_set = df[0:train_size]
        val_set = df[train_size:val_size]
        test_set = df[val_size:]
        
        # handle missing values, impute with median from train set
        median_train = train_set.median(skipna=True)
        #mean_train = train_set.mean(skipna=True)
        #std_train = train_set.std(skipna=True)
        train_set = train_set.fillna(median_train)
        val_set = val_set.fillna(median_train)
        test_set = test_set.fillna(median_train)
        
        if self.augment:
            # data augmentation https://stackoverflow.com/questions/46093073/adding-gaussian-noise-to-a-dataset-of-floating-points-and-save-it-python
            train_augmented = train_set.copy()
            #train_augmented = train_augmented.dropna()
            np.random.seed(self.random_state)
            noise = np.random.normal(0, 0.1, [len(train_augmented), len(self.header_list)-1])
            
            train_augmented.iloc[:, :-1] = train_augmented.iloc[:, :-1] + noise
            int_values = ['Age', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']
            train_augmented[int_values] = train_augmented[int_values].astype(int)
            
            # check for negatives and remove
            if (train_augmented < 0).any().any():
                train_augmented = train_augmented[train_augmented.min(axis=1) >= 0]
            
            # concat train set and augmented data
            train_set = pd.concat([train_set, train_augmented])
        
        return train_set, val_set, test_set
        
    
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
        fig, ax = plt.subplots(2, 4, figsize=(8, 4))
        counter = 0
        for idx in range(2):
            for idy in range(4):
                ax[idx, idy].boxplot(df.iloc[:,counter], 
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
        
    def show_split_labels(self):
        x = ["train", "val", "test"]
        healthy_labels = [len(self.train_set[(self.train_set["Outcome"]==0)]), len(self.val_set[(self.val_set["Outcome"]==0)]), len(self.test_set[(self.test_set["Outcome"]==0)])]
        diabetic_labels = [len(self.train_set[(self.train_set["Outcome"]==1)]), len(self.val_set[(self.val_set["Outcome"]==1)]), len(self.test_set[(self.test_set["Outcome"]==1)])]
        plt.bar(x, healthy_labels, color='r')
        plt.bar(x, diabetic_labels, bottom=healthy_labels, color='b')
        plt.xlabel("Splits")
        plt.ylabel("Number of Labels")
        plt.legend(["Healthy","Diabetes"])
        plt.title("Size of splits and number of labels in each split")
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
        
    def get_splits(self):
        
        # train data
        X_train = self.train_set.iloc[:, :-1].to_numpy()
        y_train = self.train_set["Outcome"].to_numpy()
        
        # val data
        X_val = self.val_set.iloc[:, :-1].to_numpy()
        y_val = self.val_set["Outcome"].to_numpy()
        
        # test data
        X_test = self.test_set.iloc[:, :-1].to_numpy()
        y_test = self.test_set["Outcome"].to_numpy()
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
        
        
if __name__ == '__main__':
    csv_path = "diabetes.csv"
    ddb = DiabetesDataBase(csv_path)
    # analyse and describe Data before the split
    #ddb.describe_data(ddb.diabetes_df)
    #ddb.plot_histogram_summary(ddb.diabetes_df)
    #ddb.plot_boxplot_summary(ddb.diabetes_df)
    #ddb.plot_histogram_individual()
    #ddb.plot_boxplot_individual()
    # TODO plots
    #ddb.plot_connection_to_outcome("Age")
    #ddb.plot_connection_to_outcome("BloodPressure")
    # ddb.plot_connection_to_outcome("Glucose")
    # ddb.plot_connection_to_outcome("Insulin")
    # ddb.plot_connection_to_outcome("BMI")
    #ddb.show_label_balance(ddb.diabetes_df)
    
    # analyse and describe data after the split for train_val
    # ddb.describe_data(ddb.train_val_set)
    # ddb.plot_histogram_summary(ddb.train_val_set)
    # ddb.plot_boxplot_summary(ddb.train_val_set)
    # ddb.show_label_balance(ddb.train_val_set)
    
    # analyse and describe data after the split for test
    # ddb.describe_data(ddb.test_set)
    # ddb.plot_histogram_summary(ddb.test_set)
    # ddb.plot_boxplot_summary(ddb.test_set)
    # ddb.show_label_balance(ddb.test_set)
    
    # split the data
    # X_train, X_val, X_test, y_train, y_val, y_test = ddb.get_splits()
    # print(len(X_train))
    # print(len(X_val))
    # print(len(X_test))
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler


#Load the data
data = pd.read_csv('cancer.csv')
data.columns

#check missing value
data.isnull().sum() 

#drop missing value

data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)  
data.head()

#Checking outliers
#Standardize the data
scaler = StandardScaler()
y = data.diagnosis
x = data.drop(["diagnosis"],axis=1)
x_scaler = scaler.fit_transform(x)
columns = x.columns.tolist()
x_scaler_df = pd.DataFrame(x_scaler, columns = columns)
x_scaler_df["diagnosis"] = y

data_melted = pd.melt(x_scaler_df, id_vars = "diagnosis",
                      var_name = "features",
                      value_name = "value")

mpl.style.use(['ggplot']) 
plt.figure(figsize=(12,8))
sns.boxplot(x = "features", y = "value", hue = "diagnosis", data = data_melted)
plt.xticks(rotation = 90)
plt.show()

#Checking for Imbalance in data
#Printing the number of counts for the values of the labels in the diagnosis column 
counts = data["diagnosis"].value_counts()
B, M = counts 
diag_cols = ["B", "M"]
diag_counts = [counts[0], counts[1]]

benign = (diag_counts[0] / sum(diag_counts))*100
malignant = (diag_counts[1] / sum(diag_counts)) * 100

print(f"Benign: {round(benign, 2)}%"," and number of Benign: {}".format(B))
print(f"Malignant: {round(malignant, 2)}%"," and number of Malignan: {}".format(M))

# Plotting a pie chart of the imbalanced dataset 
counts.plot(kind = "pie", figsize=(10, 5))
plt.title("A Pie Chart showing the count of Benign and Malignant Labels")
plt.show() 

##swarm plot
sns.set(style="whitegrid", palette="muted")

data_plot = pd.concat([y,x_scaler_df.iloc[:,0:10]],axis=1)
data_plot = pd.melt(data_plot,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_plot)
plt.xticks(rotation=90);

# Plotting the heatmap of correlation between features
corr = data.corr()
plt.figure(figsize=(18,18))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':12}, cmap='Reds')
plt.title("Correlation Map", fontweight = "bold", fontsize=20)
plt.show()

#postive correlation
mpl.style.use(['ggplot'])
fig,ax=plt.subplots(2,2,figsize=(12,12))
sns.scatterplot(x='perimeter_mean',y='radius_worst',data=data,hue='diagnosis',ax=ax[0][0])
sns.scatterplot(x='area_mean',y='radius_worst',data=data,hue='diagnosis',ax=ax[1][0])
sns.scatterplot(x='texture_mean',y='texture_worst',data=data,hue='diagnosis',ax=ax[0][1])
sns.scatterplot(x='area_worst',y='radius_worst',data=data,hue='diagnosis',ax=ax[1][1])
plt.show()

#no correlation
fig,ax=plt.subplots(2,2,figsize=(12,12))
sns.scatterplot(x='smoothness_mean',y='texture_mean',data=data,hue='diagnosis',ax=ax[0][0])
sns.scatterplot(x='radius_mean',y='fractal_dimension_worst',data=data,hue='diagnosis',ax=ax[1][0])
sns.scatterplot(x='texture_worst',y='symmetry_mean',data=data,hue='diagnosis',ax=ax[0][1])
sns.scatterplot(x='texture_worst',y='symmetry_se',data=data,hue='diagnosis',ax=ax[1][1]);

#negative correlation
fig,ax=plt.subplots(2,2,figsize=(12,12))
sns.scatterplot(x='area_mean',y='fractal_dimension_mean',data=data,hue='diagnosis',ax=ax[0][0])
sns.scatterplot(x='radius_mean',y='smoothness_se',data=data,hue='diagnosis',ax=ax[1][0])
sns.scatterplot(x='smoothness_se',y='perimeter_mean',data=data,hue='diagnosis',ax=ax[0][1])
sns.scatterplot(x='area_mean',y='smoothness_se',data=data,hue='diagnosis',ax=ax[1][1]);

#Remove outliers
def mod_outlier(data):
        data1 = data.copy()
        data = data._get_numeric_data()


        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 -(1.5 * iqr) 
        upper_bound = q3 +(1.5 * iqr)


        for col in data.columns:
            for i in range(0,len(data[col])):
                if data[col][i] < lower_bound[col]:            
                    data[col][i] = lower_bound[col]

                if data[col][i] > upper_bound[col]:            
                    data[col][i] = upper_bound[col]    


        for col in data.columns:
            data1[col] = data[col]

        return(data1)

data = mod_outlier(data)



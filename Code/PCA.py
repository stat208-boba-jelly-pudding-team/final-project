import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns
from collections import Counter
from sklearn.utils import resample
import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score,accuracy_score, precision_score, f1_score,plot_roc_curve,confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


X = data.drop(columns=["diagnosis"]) #original data without outliers
y = data["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pca = PCA()

sc_training = StandardScaler()
X_train = sc_training.fit_transform(X_train)
X_test = sc_training.transform(X_test)

X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.fit_transform(X_test)

print(pca.explained_variance_)# Eigenvalues
Eigen_Values = pca.explained_variance_
ell = pca.explained_variance_ratio_


# Scree plot
plt.subplots(1, 2, figsize = (20, 7))

ax1 = plt.subplot(1, 2, 1)
x = np.arange(len(Eigen_Values)) + 1
ax1.plot(x, Eigen_Values / Eigen_Values.sum(), 'ro-', lw=2)
ax1.set_xticks(x, ["" + str(i) for i in x])
ax1.set_xlabel('Number of components')
ax1.set_ylabel('Explained variance')
ax1.set_title('Scree Plot')

# Pareto plot
ax2 = plt.subplot(1, 2, 2)
ind = np.arange(1, len(ell) + 1)
ax2.bar(ind, ell, align='center', alpha=0.5)
ax2.plot(np.cumsum(ell))
ax2.set_xlabel('Number of components')
ax2.set_ylabel('Cumulative explained variance')
ax1.set_title('Pareto Plot')

for x, y in zip(ind, np.cumsum(ell)):
    label = "{:.2f}".format(y)
    if float(label) >= 0.9:
        plt.annotate("cumulative explained variance: " + label + "\n" +
                     "Number of PC: " + str(x),  # this is the text
                     (x, y),  # this is the point to label
                     textcoords='figure fraction',  # how to position the text
                     xytext=(.8, 0.5),  # distance from text to points (x,y)
                     arrowprops=dict(facecolor='black', shrink=0.1),
                     horizontalalignment='left',
                     verticalalignment='bottom',
                     ha='center')  # horizontal alignment can be left, right or center
        NUMBER_OF_PCs = x # for further use
        break
        
threshold = 0.85
for_test = 0
order = 0
for index,ratio in  enumerate (pca.explained_variance_ratio_):
    if threshold>for_test:
        for_test+= ratio
    else:
        order = index + 1
        break

print( 'The First %d Features Could Represent 90 Percents of the iarance' % order )
print( pca.explained_variance_ratio_[:order].sum() )
com_col = [ 'COM'+str(i+1) for i in range(order) ]
com_col.append('Others')
com_value = [ i for i in pca.explained_variance_ratio_[:order] ]
com_value.append( 1-pca.explained_variance_ratio_[:order].sum() )
com_colors = ['thistle', 'lightblue', 'peachpuff','lightyellow', 'lightgrey', 'wheat','lavenderblush']
plt.figure( figsize=[7,7] )
plt.pie( x=com_value,labels=com_col,colors=com_colors,autopct='%.2f' )
plt.title( 'the first 6 components' )
plt.show()

#Match the labels of components
pc = pca.components_[:6]
pc = pd.DataFrame(pc)
pc.columns = data.drop(['diagnosis'],axis = 1).columns.values
for i in range(len(pc)):
    pc1 = pc.iloc[i]
    #print('component {}:\n {}\n'.format(i+1,pc1[abs(pc1)>=0.2]))
    

pc.index = ['PC_1', 'PC_2', 'PC_3', 'PC_4','PC_5','PC_6']
pc

# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X_train ,y_train)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_train,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("COMP 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("COMP 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("COMP 3")
ax.w_zaxis.set_ticklabels([])

plt.show()

pca = PCA(n_components=6)
X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.fit_transform(X_test)

#Logistics regression
y_test.replace('M',1,inplace=True)
y_test.replace('B',0,inplace=True)
y_train.replace('M',1,inplace=True)
y_train.replace('B',0,inplace=True)


LR = LogisticRegression(max_iter = 200)
LR.fit(X_train_PCA,y_train)
scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
scores = cross_validate(LR, X_train_PCA, y_train, scoring=scoring, cv=10)

sorted(scores.keys())
LR_fit_time = scores['fit_time'].mean()
LR_score_time = scores['score_time'].mean()
LR_accuracy = scores['test_accuracy'].mean()
LR_precision = scores['test_precision_macro'].mean()
LR_recall = scores['test_recall_macro'].mean()
LR_f1 = scores['test_f1_weighted'].mean()

print(f"Accuracy by using KfoldCrossVal score of Logistic Regression is: {round(cross_val_score(LR,X_train_PCA,y_train,cv=10).mean(), 4)}")

y_pred = LR.predict(X_test_PCA) #Predict data for eveluating 
cmLR = confusion_matrix(y_test,y_pred)
f,ax = plt.subplots(figsize = (6,4))
plt.title('The Confusion Matrix of Logistic Regression')
sns.heatmap(cmLR,annot = True,linewidth = 1,fmt =".0f",ax = ax,cmap='Blues');total_LR=sum(sum(cmLR))


#####from confusion matrix calculate accuracy
accuracy_LR=(cmLR[0,0]+cmLR[1,1])/total_LR
print(f"Accuracy: {round(accuracy_LR, 4)}")

sensitivity_LR = cmLR[0,0]/(cmLR[0,0]+cmLR[1,0])
print(f"Sensitivity: {round(sensitivity_LR, 4)}")

specificity_LR = cmLR[1,1]/(cmLR[0,1]+cmLR[1,1])
print(f"Specificity: {round(specificity_LR, 4)}")

balanced_accuracy_LR=(sensitivity_LR+specificity_LR )*0.5
print(f"Balanced accuracy of Logistic Regression : {round(balanced_accuracy_LR, 4)}")

plot_roc_curve(LR, X_test_PCA, y_test);
plt.plot([0, 1], ls="--")
plt.title('Receiver Operating Characteristic - Logistic regression');

#KNN
accuracy=[]
for i in range(1, 6):
    KNN = KNeighborsClassifier(n_neighbors = i)
    KNN = KNN.fit(X_train_PCA, y_train)
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
    scores = cross_validate(KNN, X_train_PCA, y_train, scoring=scoring, cv=10)
    sorted(scores.keys())
    KNN_fit_time = scores['fit_time'].mean()
    KNN_score_time = scores['score_time'].mean()
    KNN_accuracy = scores['test_accuracy'].mean()
    KNN_precision = scores['test_precision_macro'].mean()
    KNN_recall = scores['test_recall_macro'].mean()
    KNN_f1 = scores['test_f1_weighted'].mean()
   
    accuracy.append(KNN.score(X_test_PCA,y_test))

print(f"Accuracy by using KfoldCrossVal score of KNN is: {round(max(accuracy), 4)}"," at K = {}".format(accuracy.index(max(accuracy))+1))

plt.figure(figsize=(8,6))
plt.plot(range(1,6),accuracy, 
         marker='o',markerfacecolor='black', markersize=5)
plt.title('Accuracy vs. K')
plt.xlabel('K')
plt.ylabel('Accuracy');

KNN = KNeighborsClassifier(n_neighbors = 5)
KNN = KNN.fit(X_train_PCA, y_train)
y_pred = KNN.predict(X_test_PCA)
cmKNN = confusion_matrix(y_test,y_pred)
f,ax = plt.subplots(figsize = (6,4))
plt.title('The Confusion Matrix of KNN')
sns.heatmap(cmKNN,annot = True,linewidth = 1,fmt =".0f",ax = ax,cmap='Blues');

total_KNN=sum(sum(cmKNN))
#####from confusion matrix calculate accuracy
accuracy_KNN=(cmKNN[0,0]+cmKNN[1,1])/total_KNN
print(f"Accuracy: {round(accuracy_KNN, 4)}")

sensitivity_KNN = cmKNN[0,0]/(cmKNN[0,0]+cmKNN[1,0])
print(f"Sensitivity: {round(sensitivity_KNN, 4)}")

specificity_KNN = cmKNN[1,1]/(cmKNN[0,1]+cmKNN[1,1])
print(f"Specificity: {round(specificity_KNN, 4)}")

balanced_accuracy_KNN=(sensitivity_KNN+specificity_KNN )*0.5
print(f"Balanced accuracy of KNN: {round(balanced_accuracy_KNN, 4)}")

plot_roc_curve(KNN, X_test_PCA, y_test)
plt.plot([0, 1], ls="--")
plt.title('Receiver Operating Characteristic - KNN');

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB = NB.fit(X_train_PCA, y_train)
scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
scores = cross_validate(NB, X_train_PCA, y_train, scoring=scoring, cv=10)

sorted(scores.keys())
NB_fit_time = scores['fit_time'].mean()
NB_score_time = scores['score_time'].mean()
NB_accuracy = scores['test_accuracy'].mean()
NB_precision = scores['test_precision_macro'].mean()
NB_recall = scores['test_recall_macro'].mean()
NB_f1 = scores['test_f1_weighted'].mean()


print(f"Accuracy by using KfoldCrossVal score of Naive Bayes is: {round(cross_val_score(NB,X_train_PCA,y_train,cv=10).mean(), 4)}")

y_pred = NB.predict(X_test_PCA)
cmNB = confusion_matrix(y_test,y_pred)
f,ax = plt.subplots(figsize = (6,4))
plt.title('The Confusion Matrix of Naive Bayes')
sns.heatmap(cmNB,annot = True,linewidth = 1,fmt =".0f",ax = ax,cmap='Blues');

total_NB=sum(sum(cmNB))
#####from confusion matrix calculate accuracy
accuracy_NB=(cmNB[0,0]+cmNB[1,1])/total_NB
print(f"Accuracy: {round(accuracy_NB, 4)}")

sensitivity_NB = cmNB[0,0]/(cmNB[0,0]+cmNB[1,0])
print(f"Sensitivity: {round(sensitivity_NB, 4)}")

specificity_NB = cmNB[1,1]/(cmNB[0,1]+cmNB[1,1])
print(f"Specificity: {round(specificity_NB, 4)}")

balanced_accuracy_NB=(sensitivity_NB+specificity_NB )*0.5
print(f"Balanced accuracy of Naive Bayes: {round(balanced_accuracy_NB, 4)}")

plot_roc_curve(NB, X_test_PCA, y_test)
plt.plot([0, 1], ls="--")
plt.title('Receiver Operating Characteristic - Naive Bayes');

# Random Forest
accuracy=[]
for i in range(1, 6):
    RF = RandomForestClassifier(n_estimators = i,random_state=0)
    RF = RF.fit(X_train_PCA, y_train)
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
    scores = cross_validate(RF, X_train_PCA, y_train, scoring=scoring, cv=10)
    sorted(scores.keys())
    RF_fit_time = scores['fit_time'].mean()
    RF_score_time = scores['score_time'].mean()
    RF_accuracy = scores['test_accuracy'].mean()
    RF_precision = scores['test_precision_macro'].mean()
    RF_recall = scores['test_recall_macro'].mean()
    RF_f1 = scores['test_f1_weighted'].mean()
    accuracy.append(RF.score(X_test_PCA,y_test))

print(f"Accuracy by using KfoldCrossVal score of Random Forest is: {round(max(accuracy), 4)}"," at N = {}".format(accuracy.index(max(accuracy))+1))

plt.figure(figsize=(8,6))
plt.plot(range(1,6),accuracy, 
         marker='o',markerfacecolor='black', markersize=5)
plt.title('Accuracy vs. N')
plt.xlabel('N')
plt.ylabel('Accuracy');

RF = RandomForestClassifier(n_estimators =3,random_state=0)
RF = RF.fit(X_train_PCA, y_train)
y_pred = RF.predict(X_test_PCA)
cmRF = confusion_matrix(y_test,y_pred)
f,ax = plt.subplots(figsize = (6,4))
plt.title('The Confusion Matrix of Random Forest')
sns.heatmap(cmRF,annot = True,linewidth = 1,fmt =".0f",ax = ax,cmap='Blues');

total_RF=sum(sum(cmRF))
#####from confusion matrix calculate accuracy
accuracy_RF=(cmRF[0,0]+cmRF[1,1])/total_RF
print(f"Accuracy: {round(accuracy_RF, 4)}")

sensitivity_RF = cmRF[0,0]/(cmRF[0,0]+cmRF[1,0])
print(f"Sensitivity: {round(sensitivity_RF, 4)}")

specificity_RF = cmRF[1,1]/(cmRF[0,1]+cmRF[1,1])
print(f"Specificity: {round(specificity_RF, 4)}")

balanced_accuracy_RF=(sensitivity_RF+specificity_RF )*0.5
print(f"Balanced accuracy of Random Forest : {round(balanced_accuracy_RF, 4)}")

plot_roc_curve(RF, X_test_PCA, y_test)
plt.plot([0, 1], ls="--")
plt.title('Receiver Operating Characteristic - Random Forest');

# Feature Importance
X_train_PCA = pd.DataFrame(X_train_PCA)
X_train_PCA.columns=['COMP 1','COMP 2','COMP 3','COMP 4','COMP 5','COMP 6']

def FeatureImportance (_model):
    fi = pd.DataFrame({'feature': list(X_train_PCA.columns),
                       'importance': _model.feature_importances_}).sort_values('importance', ascending = False)
    return fi
FeatureImportance(RF)


# Plot the feature importances of the forest

importances = RF.feature_importances_
std = np.std([tree.feature_importances_ for tree in RF.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

plt.figure(1, figsize=(10,8))
plt.title("Feature importances")
plt.bar(range(X_train_PCA.shape[1]), importances[indices],
       color="#875FDB", align="center")
plt.xticks(range(X_train_PCA.shape[1]), X_train_PCA.columns[indices],rotation=90)
plt.xlim([-1, X_train_PCA.shape[1]])
plt.show()

#SVM
# Create a Support Vector Classifier

svc = svm.SVC()

# Hyperparameter Optimization
parameters = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']}
]

# Run the grid search
grid_obj = GridSearchCV(svc, parameters)
grid_obj = grid_obj.fit(X_train_PCA, y_train)

# Set the svc to the best combination of parameters
svc = grid_obj.best_estimator_
svc

SVM_linear = SVC(kernel='linear',probability = True,C=10)
SVM_linear=SVM_linear.fit(X_train_PCA, y_train)
scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
scores = cross_validate(SVM_linear, X_train_PCA, y_train, scoring=scoring, cv=10)
sorted(scores.keys())
SVM_linear_fit_time = scores['fit_time'].mean()
SVM_linear_score_time = scores['score_time'].mean()
SVM_linear_accuracy = scores['test_accuracy'].mean()
SVM_linear_precision = scores['test_precision_macro'].mean()
SVM_linear_recall = scores['test_recall_macro'].mean()
SVM_linear_f1 = scores['test_f1_weighted'].mean()


print(f"Accuracy by using KfoldCrossVal score of SVM (linear) is: {round(cross_val_score(SVM_linear,X_train_PCA,y_train,cv=10).mean(), 4)}")

y_pred = SVM_linear.predict(X_test_PCA)
cm_linear = confusion_matrix(y_test,y_pred)
f,ax = plt.subplots(figsize = (6,4))
plt.title('The Confusion Matrix of SVM (linear)')
sns.heatmap(cm_linear,annot = True,linewidth = 1,fmt =".0f",ax = ax,cmap='Blues');

total_SVM_linear=sum(sum(cm_linear))
#####from confusion matrix calculate accuracy
accuracy_SVM_linear=(cm_linear[0,0]+cm_linear[1,1])/total_SVM_linear
print(f"Accuracy: {round(accuracy_SVM_linear, 4)}")

sensitivity_SVM_linear = cm_linear[0,0]/(cm_linear[0,0]+cm_linear[1,0])
print(f"Sensitivity: {round(sensitivity_SVM_linear, 4)}")

specificity_SVM_linear = cm_linear[1,1]/(cm_linear[0,1]+cm_linear[1,1])
print(f"Specificity: {round(specificity_SVM_linear, 4)}")

balanced_accuracy_SVM_linear=(sensitivity_SVM_linear+specificity_SVM_linear )*0.5
print(f"Balanced accuracy of SVM(linear): {round(balanced_accuracy_SVM_linear, 4)}")

plot_roc_curve(SVM_linear, X_test_PCA, y_test)
plt.plot([0, 1], ls="--")
plt.title('Receiver Operating Characteristic - SVM (Linear)');

#SVM(rbf)
# Hyperparameter Optimization
parameters = [
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# Run the grid search
grid_obj = GridSearchCV(svc, parameters)
grid_obj = grid_obj.fit(X_train_PCA, y_train)

# Set the svc to the best combination of parameters
svc = grid_obj.best_estimator_
svc

SVM_rbf = SVC(kernel='rbf',probability = True,C=100, gamma=0.001)
SVM_rbf=SVM_rbf.fit(X_train_PCA, y_train)
scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted']
scores = cross_validate(SVM_rbf, X_train_PCA, y_train, scoring=scoring, cv=10)
sorted(scores.keys())
SVM_rbf_fit_time = scores['fit_time'].mean()
SVM_rbf_score_time = scores['score_time'].mean()
SVM_rbf_accuracy = scores['test_accuracy'].mean()
SVM_rbf_precision = scores['test_precision_macro'].mean()
SVM_rbf_recall = scores['test_recall_macro'].mean()
SVM_rbf_f1 = scores['test_f1_weighted'].mean()


print(f"Accuracy by using KfoldCrossVal score of SVM (rbf) is: {round(cross_val_score(SVM_rbf,X_train_PCA,y_train,cv=10).mean(), 4)}")

y_pred=SVM_rbf.predict(X_test_PCA)
cm_rbf = confusion_matrix(y_test, y_pred)

f,ax = plt.subplots(figsize = (6,4))
plt.title('The Confusion Matrix of SVM with Kernal RBF')
sns.heatmap(cm_rbf,annot = True,linewidth = 1,fmt =".0f",ax = ax,cmap='Blues');

total_SVM_rbf=sum(sum(cm_rbf))
#####from confusion matrix calculate accuracy
accuracy_SVM_rbf=(cm_rbf[0,0]+cm_rbf[1,1])/total_SVM_rbf
print(f"Accuracy: {round(accuracy_SVM_rbf, 4)}")

sensitivity_SVM_rbf = cm_rbf[0,0]/(cm_rbf[0,0]+cm_rbf[1,0])
print(f"Sensitivity: {round(sensitivity_SVM_rbf, 4)}")

specificity_SVM_rbf = cm_rbf[1,1]/(cm_rbf[0,1]+cm_rbf[1,1])
print(f"Specificity: {round(specificity_SVM_rbf, 4)}")

balanced_accuracy_SVM_rbf=(sensitivity_SVM_rbf+specificity_SVM_rbf )*0.5
print(f"Balanced accuracy of SVM(rbf) : {round(balanced_accuracy_SVM_rbf, 4)}")

plot_roc_curve(SVM_rbf, X_test_PCA, y_test);
plt.plot([0, 1], ls="--")
plt.title('Receiver Operating Characteristic - SVM (Radial)');

models_correlation = pd.DataFrame({
    'Model'       : ['Logistic Regression','Random forests','Naive Bayes','K-Nearest Neighbors', 'Support Vector Machine_linear','Support Vector Machine_radial'],
    'Fitting time': [LR_fit_time, RF_fit_time,NB_fit_time, KNN_fit_time,SVM_linear_fit_time,SVM_rbf_fit_time],
    'Scoring time': [LR_score_time,RF_score_time,NB_score_time,  KNN_score_time,SVM_linear_score_time,SVM_rbf_score_time  ],
    'Accuracy'    : [LR_accuracy, RF_accuracy, NB_accuracy, KNN_accuracy,SVM_linear_accuracy,SVM_rbf_accuracy ],
    'Balanced accuracy' :[balanced_accuracy_LR, balanced_accuracy_RF, balanced_accuracy_NB,balanced_accuracy_KNN,balanced_accuracy_SVM_linear, balanced_accuracy_SVM_rbf],
    'Precision'   : [LR_precision, RF_precision, NB_precision, KNN_precision, SVM_linear_precision,SVM_rbf_precision],
    'Recall'      : [LR_recall,RF_recall,NB_recall,KNN_recall,SVM_linear_recall,SVM_rbf_recall],
    'F1_score'    : [LR_f1, RF_f1, NB_f1, KNN_f1, SVM_linear_f1,SVM_rbf_f1],
    }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Balanced accuracy', 'Precision',  'Recall','F1_score'])

models_correlation.sort_values(by='Balanced accuracy', ascending=False)



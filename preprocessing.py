#importing libraries
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy
#setting options
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

#----------------------------------------------------------------------------------------------------
# Loading  dataset
url = "adult.csv"
df = pandas.read_csv(url)

#data exploration
df.shape
df.head()
df.info()
df.describe()
df.isna().sum()
df.income.value_counts() #data highly skewed so we will go for upsampling

##Upsampling
## Importing resample from *sklearn.utils* package.
from sklearn.utils import resample
# Separate the case of less  and greater cases
bank_less_equal =  df[df.income == ' <=50K']
bank_greater =    df[df.income == ' >50K']
bank_less_equal.shape
bank_greater.shape

##Upsample the greater cases.
# sample with replacement
df_minority_upsampled = resample(bank_greater,replace=True, n_samples=22000)
# Combine majority class with upsampled minority class
new_bank_df = pandas.concat([bank_less_equal, df_minority_upsampled])
new_bank_df.income.value_counts()

#Rearranging 
from sklearn.utils import shuffle
df = shuffle(new_bank_df)
df.income.value_counts()

# filling missing values
col_names = df.columns
for c in col_names:
    df[c] = df[c].replace("?", numpy.NaN)
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))


#discretisation
df.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace = True)


#label Encoder
category_col =['workclass', 'race', 'education','marital-status', 'occupation','relationship', 'gender', 'native-country', 'income'] 
labelEncoder = preprocessing.LabelEncoder()


# creating a map of all the numerical values of each categorical labels.
mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
print(mapping_dict)


#droping redundant columns
df=df.drop(['fnlwgt','educational-num'], axis=1)


#preparing training and test data
X = df.values[:, 0:12]
Y = df.values[:,12]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


#utility
#function for plotting roc_auc curve
def draw_roc_curve( model, test_X, test_y ):
    ## Creating and initializing a results DataFrame with actual labels
    test_results_df = pd.DataFrame( { 'actual': test_y } )
    test_results_df = test_results_df.reset_index()
    # predict the probabilities on the test set
    predict_proba_df = pd.DataFrame( model.predict_proba( test_X ) )
    ## selecting the probabilities that the test example belongs to class 1
    test_results_df['chd_1'] = predict_proba_df.iloc[:,1:2]
    ## Invoke roc_curve() to return the fpr, tpr and threshold values.
    ## threshold values contain values from 0.0 to 1.0
    fpr, tpr, thresholds = metrics.roc_curve( test_results_df.actual,
    test_results_df.chd_1,
    drop_intermediate = False )
    ## Getting the roc auc score by invoking metrics.roc_auc_score method
    auc_score = metrics.roc_auc_score( test_results_df.actual, test_results_df.chd_1 )
    ## Setting the size of the plot
    plt.figure(figsize=(8, 6))
    ## plotting the actual fpr and tpr values
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    ## plotting th diagnoal line from (0,1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    ## Setting labels and titles
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return auc_score, fpr, tpr, thresholds



#building a simple model
#KNN Algorithm
## Importing the KNN classifier algorithm
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
## Initializing the classifier
knn_clf = KNeighborsClassifier()
## Fitting the model with the training set
knn_clf.fit( X_train, y_train )



## Invoking draw_roc_curve with the KNN model
##_, _, _, _ = draw_roc_curve( knn_clf, X_test, y_test )


## Predicting on test set
y_pred = knn_clf.predict(X_test)
metrics.classification_report( y_test, y_pred ) 

## seraching optimal parameters
## Importing GridSearchCV
from sklearn.model_selection import GridSearchCV
## Creating a dictionary with hyperparameters and possible values for searching
tuned_parameters = [{'n_neighbors': range(5,10),
'metric': ['canberra', 'euclidean', 'minkowski']}]
## Configuring grid search
clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=10,scoring='roc_auc')
## fit the search with training set
clf.fit(X_train, y_train )


print('Best Parameters: '+str(clf.best_params_))
print('Best Score: '+str(clf.best_score_))


##Building classifier with optimal parametes
## Initializing the classifier

knn_clf_best = KNeighborsClassifier(n_neighbors=5,metric='canberra')
## Fitting the model with the training set
knn_clf_best.fit( X_train, y_train )


y_pred = knn_clf_best.predict(X_train)
print( metrics.classification_report( y_train, y_pred ) )

## Invoking draw_roc_curve with the KNN model
_, _, _, _ = draw_roc_curve( knn_clf_best, X_test, y_test )


## Predicting on test set
y_pred = knn_clf_best.predict(X_test)
print( metrics.classification_report( y_test, y_pred ) )

## Invoking draw_roc_curve with the KNN model
_, _, _, _ = draw_roc_curve( knn_clf_best, X_test, y_test )

#creating and training a model
#serializing our model to a file called model.pkl
import pickle
pickle.dump(knn_clf_best, open("model.pkl","wb"))


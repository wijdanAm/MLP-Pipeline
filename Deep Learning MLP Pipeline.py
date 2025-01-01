"""#Step 0 : import the necessary libraries"""
#import laspy
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#import time
from numpy import asarray
from numpy import savetxt
from sklearn.metrics import f1_score
"""step 1 : is to load the data using the function np.loadtext from numpy library"""

text = np.loadtxt('Your_PoinCloud.txt')

"""#then extract each variable and feature from our point cloud """

x = text[:, 0]
y = text[:, 1]
z = text[:, 2]
class_label = text[:, 4]
Eigenvalues_Sum = text[:, 5]
Omnivariance = text[:, 6]
Eigenentropy = text[:, 7]
Anisotropy = text[:, 8]
Planarity = text[:, 9]
Linearity = text[:, 10]
PCA1 = text[:, 11]
PCA2 = text[:, 12]
SurfaceVariation = text[:, 13]
Sphericity = text[:, 14]
Verticality = text[:, 15]

#group all the features except the variable that contains labels

#All_coords = np.vstack((x,y,z,Eigenvalues_Sum,Omnivariance,Eigenentropy,Anisotropy,Planarity,Linearity,PCA1,PCA2,SurfaceVariation, Sphericity, Verticality)).transpose()

#use just the selected features

All_coords = np.vstack((x,y,z,Omnivariance,Eigenentropy,Anisotropy,SurfaceVariation, Sphericity, Verticality)).transpose()

"""
step 2 : split the dataset into training set and testing set, and separating the labels of each set in a separate 
variable, so that the lables of the test set will be considered as a refence model to compare it with the obatined labels 
in order to get the accuracy of the algorithm. to split the data we use the function -train_test_split- from sklearn labrary.
that take as parameters the grouped features, the labels, the size of test set in our case we use 0.2==20% of the original data 
and the last parameter is  random_state controls the shuffling process. this function returns 4 variables; X_train contain 
training set, y_train : the labels of the training set, X_test : test set and y_test the labels of the test set that
will be considered as reference model to calculate the score
"""

X_train, X_test, y_train, y_test = train_test_split(All_coords, class_label, test_size=0.2, random_state=0)

"""
step 3 : Create the object ( model) of the MLP machine learning algorithm by calling the function MLPClassifier. the used 
parameters are: numbre of hidden layers and activation function, for alpha value and solver we keep them as default 
alpha = 0.0001 solver = 'adam'
"""
model = MLPClassifier(
    hidden_layer_sizes=(50,) * 10,  # 10 couches cachées avec 50 neurones chacune
    activation="logistic",            # Fonction d'activation sigmoïde
    max_iter=1000,                    # Nombre maximal d'itérations pour l'entraînement
    solver="adam",                    # Optimiseur pour l'entraînement
    random_state=42                   # Pour la reproductibilité des résultats
)

#model.fit(X_train[:,3:], y_train)
"""
step 4 : before training the algorithm we create the pipeline object using the function -make_pipeline- from sklearn labrary
which takes as parameters the created model of ML algorithm + steps : list of Estimator objects, memory : str 
or object with the joblib.Memory interface default=None used to cache the fitted transformers of the pipeline, verbose : bool, 
default=False. If True, the time elapsed while fitting each step will be printed as it is completed.
we keep all params by defaults and use StandardScaler() removes the mean and scales each feature/variable to unit variance + 
MPL object
"""

pipeline = make_pipeline(StandardScaler(), model)

"""
step 5 : train the model using .fit function from sklearn library that takes as params training set and its labels
"""

pipeline.fit(X_train, y_train)


"""
step 6 : once the algorithm is trained it'll be tested using the test set by calling the function -predict- takes as params
the test set
"""

prediction_test = pipeline.predict(X_test)

"""
Step 7 : compare the results obtained from predictions of the labels of the test set with the reference model y_test using the 
function -accuracy_score- that take as params the y_test reference model and the predicted values and return an numbre that 
represent the score
"""
print("Test set accuracy score: ", accuracy_score(y_test,prediction_test))


"""
Step 8 : compare the results obtained using f1 score to calculate the score
"""

print("Test set f1 score: ", f1_score(y_test, prediction_test, average=None))
print("Test set f1-weighted score: ", f1_score(y_test, prediction_test, average='weighted'))
print("Test set f1-macro score: ", f1_score(y_test, prediction_test, average='macro'))
print("Test set f1-micro score: ", f1_score(y_test, prediction_test, average='micro'))

"""# export results for test set"""

prediction_test_=np.resize(prediction_test,(len(prediction_test),1))
y_test_=np.resize(y_test,(len(y_test),1))

Data_Pred = np.hstack((X_test,prediction_test_))
#Data_init = np.hstack((X_test,y_test_))

#return just labeled as 3 noise class
mask_3 = (Data_Pred[:, -1] == 3)
Other = Data_Pred[mask_3, :]

#return just labeled as 2 building class
mask_2 = (Data_Pred[:, -1] == 2)
Building = Data_Pred[mask_2, :]

#return just labeled as 1 vegetation class
mask_1 = (Data_Pred[:, -1] == 1)
Vegetation = Data_Pred[mask_1, :]

#return just labeled as 0 terrain class
mask_0 = (Data_Pred[:, -1] == 0)
non_Vegetation = Data_Pred[mask_0, :]

# save to txt file

savetxt('All_veget_Pred3.txt', Vegetation, delimiter=' ')
savetxt('All_Building_Pred3.txt', Building, delimiter=' ')
savetxt('All_Terrian_Pred3.txt', non_Vegetation, delimiter=' ')

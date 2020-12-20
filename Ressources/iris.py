import numpy as np # linear algebra
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    df = pd.read_csv('./iris.csv')
    df.head()
    print(df.head())
    X_train, X_test, y_train, y_test = train_test_split(df[['sepal.length', 'sepal.width', 
                                                            'petal.length', 'petal.width']],
                                                        df['variety'], test_size=0.7,random_state=4)
    print("X_train shape: {}\ny_train shape: {}".format(X_train.shape, y_train.shape))
    print("X_test shape: {}\ny_test shape: {}".format(X_test.shape, y_test.shape))
    remote_server_uri = "https://manip.herokuapp.com/"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name='IRIS')
    with mlflow.start_run(run_name="KNN"):
        knn = KNeighborsClassifier(n_neighbors=8)
        mlflow.log_param('neighbors',8)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score =knn.score(X_test, y_test)
        print(score)
        mlflow.log_metric('score',score)


     
    
  
                                    
    
            

    
    

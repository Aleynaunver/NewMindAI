#### Classification of Comments ####
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import joblib

copy_data = pd.read_csv("datasets/copy_data.csv")

class tfidf_class:


    ##Feature Extraction
    #TF-IDF Function. This function used to calculate term weights in the text.
    def tf_idf(data, column):
        
        #Since TF-IDF is a text-based algorithm, it cannot work in list format. The column that became a list due to previous operations was converted back to a string.
        data[column] = data[column].apply(lambda x: " ".join(x) if isinstance(x, list) else x)  

        #Limiting number of features to reduce the size of the matrix
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)      #keep only the top 5000 features
        tfidf_result = tfidf.fit_transform(data[column])                                          #get tf-df values

        tfidf_feature_names = tfidf.get_feature_names_out()                                       #take features name
        tfidf_df = pd.DataFrame(tfidf_result.toarray(), columns=tfidf_feature_names)  #convert matrix to DataFrame for better understanding

        return tfidf_df, tfidf      #return result and tfidf object 



    def train_model(data):

        #Classification of categories in the label column.

        #Data was converted to numerical values ​​with label encoder. This conversion was done so that machine learning algorithms could work.
        label_encoder = LabelEncoder()
        data["encoded_label"] = label_encoder.fit_transform(data["label"])

        print(label_encoder.classes_)  #According to these,the number of label is as follows: Negative=0, Positive=1

        #Call the function and get the results
        tfidf_result, tfidf = tfidf_class.tf_idf(data, "cleaned_review")
        print(tfidf_result)


        #Analyze TF-IDF results
        print(tfidf_result.head(10))


        #Most used words in the text
        top_words = tfidf_result.sum(axis=0).sort_values(ascending=False).head(15)
        print(f"Most used words in the text \n{top_words}")

        #Least used words in the text
        least_words = tfidf_result.sum(axis=0).sort_values().head(15)
        print(f"Least used words in the text \n{least_words}")



        #List of classification models
        models = {

            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }

        #Seaching best parameters
        param_grids = {                 #All the following combinations were tried to find the best parameters. Since the data set was large, the best parameter was not searched again before each train.
                                        #The tried parameter types were selected from the codes in the comment line.
            "Logistic Regression": {
                'C': 1.0,                       #[0.1, 1, 10]
                'solver': "liblinear",          #['liblinear', 'saga']
                'max_iter': 100                 #[100, 200, 300]
            },

            "Random Forest": {
                'n_estimators': 100,            #[100, 200, 300]
                'max_depth': None,              #[None, 10, 20]
                'min_samples_split': 2,         #[2, 5]
                'min_samples_leaf': 2           #[1, 2]
            },

            "SVM": {
                'C': 1,                         #[0.1, 1, 10]
                'kernel': 'rbf',                #['linear', 'rbf']
            },

            "Decision Tree": {
                'max_depth': 20,                #[None, 10, 20]
                'min_samples_split': 2,         #[2, 5]
                'min_samples_leaf': 1           #[1, 2]
            },

            "Gradient Boosting": {
                'n_estimators': 100,            #[100, 200]
                'learning_rate': 0.1,           #[0.01, 0.1, 0.2]
                'max_depth': 3                  #[0.01, 0.1, 0.2]
            },

            "K-Nearest Neighbors": {
                'n_neighbors': 7,               #[3, 5, 7]
                'weights': 'uniform',           #['uniform', 'distance']
                'metric': 'euclidean'           #['euclidean', 'manhattan']
            },

            "Naive Bayes": {
                'var_smoothing': 1e-9           #[1e-9, 1e-8, 1e-7]
            }
        }


        #Split train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_result, data['encoded_label'], test_size=0.2, random_state=42
        )


        #Call the function and get the results
        results_df, trained_models = tfidf_class.train_and_evaluate_models(X_train, X_test, y_train, y_test, models, param_grids)
        print(results_df)

        
        best_model = trained_models['Logistic Regression']           #choose model in trained_models
        tfidf_class.save_model(best_model, "Logistic_Regression")    #save the model
        model = tfidf_class.load_model("Logistic_Regression")    

        joblib.dump(tfidf, "tfidf_vectorizer.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")   


    #Function to save the selected model
    def save_model(model, model_name):

        os.makedirs("models", exist_ok=True)      #create folder for models
        file_path = f"models/{model_name}.pkl"    #path to save model file
        joblib.dump(model, file_path)             #save the model to the specified file path


    #Function to load the selected model
    def load_model(model_name):

        file_path = f"models/{model_name}.pkl"    #file path of the saved model

        if os.path.exists(file_path):             #load model if file exists
            return joblib.load(file_path)

        else:                                     #return error message if file not found
            raise FileNotFoundError(f"Model file not found: {file_path}")



    #Model training and evaluation function
    def train_and_evaluate_models(X_train, X_test, y_train, y_test, models, param_grids):
        
        results = []            #empty list for results
        trained_models = {}     #empty dictionary for trained models
        

        for model_name, model in models.items():
            
            model.set_params(**param_grids[model_name])     #adjusting the parameters of the model with the best parameters
            
            model.fit(X_train, y_train)                     #train the model
            
            y_pred = model.predict(X_test)                  #prediction on test set

            trained_models[model_name] = model              #save the model  


            #Calculating performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)


            #Adding results to list
            results.append({  
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })
        
        
        return pd.DataFrame(results), trained_models     #return result as DataFrame

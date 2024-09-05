import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sklearn

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss,roc_auc_score, confusion_matrix,ConfusionMatrixDisplay, roc_curve, classification_report

data=pd.read_csv(r"C:\Users\Kola Karthikeya\Downloads\archive (5)\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df=pd.DataFrame(data)
df.drop(["customerID"],axis = 1,inplace=True)
df["TotalCharges"] = np.where(df["TotalCharges"].str.strip() == '', np.nan, df["TotalCharges"])
df["TotalCharges"] = df["TotalCharges"].astype(float)

df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})


y = df['Churn'].astype(int)
X = df.drop(["Churn"],axis = 1)

cont_cols = X.select_dtypes(exclude = "object").columns
cat_cols = X.select_dtypes(include = "object").columns

cont_pl = Pipeline(steps = [('Imputation',SimpleImputer(strategy="median")),
                    ('Scaling',RobustScaler()),
                    ('PowerTransformation',PowerTransformer())])

cat_nominal_pl = Pipeline(steps = [('Imputation',SimpleImputer(strategy="most_frequent")),
                    ('Onehot encoding',OneHotEncoder(sparse_output=False,drop='first'))])

cat_ordinal_pl = Pipeline(steps = [('Imputation',SimpleImputer(strategy = "most_frequent")),
                            ('Ordinal encoding',OrdinalEncoder())])

ct = ColumnTransformer(transformers= [('Continuous Cols',cont_pl,cont_cols),
                                ('Cat Nominal Cols',cat_nominal_pl,cat_cols[cat_cols != "Contract"]),
                                ('Cat Ordinal Cols',cat_ordinal_pl,['Contract'])])

final_pl = Pipeline([('Final',ct)])


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=10,stratify=y)

X_train_df = pd.DataFrame(final_pl.fit_transform(X_train),columns=final_pl.get_feature_names_out())
X_test_df = pd.DataFrame(final_pl.transform(X_test),columns = final_pl.get_feature_names_out())

classification_models = ["KNeighborsClassifier","GaussianNB", "LogisticRegression", "SVC", "DecisionTreeClassifier", "RandomForestClassifier"]

st.sidebar.header("Select the model:")
selected_classification_models = []
for clf in classification_models:
    if st.sidebar.checkbox(clf, key=f"class_{clf}"):
        selected_classification_models.append(clf)

results = []

def Random_Search(model,param_grid, score):

    Random_Search = RandomizedSearchCV(model, param_grid, scoring=score, cv=5)

    start_time = time.time()
    Random_Search.fit(X_train_df, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    best_model = Random_Search.best_estimator_

    start_time = time.time()
    y_pred = best_model.predict(X_test_df)
    end_time = time.time()
    testing_time = end_time - start_time

    clf_report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(clf_report).transpose()

    proba_predict=best_model.predict_proba(X_test_df)
    fpr,tpr,__=roc_curve(y_test,proba_predict[:,1])

    plt.plot(fpr,tpr,label="Random roc")
    plt.title("Roc_auc Curve")

    results.append({"Classifier": model,
                "Best parameters found": Random_Search.best_params_,
                "Best cross-validation score": Random_Search.best_score_,
                "Training time":training_time,
                "Testing time": testing_time,
                "Roc-Auc score": roc_auc_score(y_test,y_pred),
                "Log-loss": log_loss(y_test,proba_predict),
                "Classification Report": report_df,
                "Confusion Matrix": confusion_matrix(y_test,y_pred),
                "FPR": fpr,
                "TPR":tpr,
                "Best model":best_model
                })
    
    
def KNN_Classifier():
    knn_clf = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [i for i in range(1, 20, 2)],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    Random_Search(knn_clf, param_grid, "roc_auc")

def Gaussian_NB():
    GNB_clf = GaussianNB()
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
    Random_Search(GNB_clf, param_grid, "roc_auc")

def Logistic_Regression():
    LR_clf = LogisticRegression(max_iter=10000)
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'solver': ['liblinear', 'saga']},
        {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
        {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': np.linspace(0, 1, 5)},
        {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['none'], 'solver': ['newton-cg', 'lbfgs', 'sag']}
    ]
    Random_Search(LR_clf, param_grid, "accuracy")

def SVM_Classifier():
    SVC_clf = SVC(probability=True)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
    Random_Search(SVC_clf, param_grid, 'f1')

def Decision_Tree():
    DT_clf = DecisionTreeClassifier()
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    }
    Random_Search(DT_clf, param_grid, 'accuracy')

def Randon_Forest():
    RFC_clf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    Random_Search(RFC_clf, param_grid, 'accuracy')


def main():
    if st.sidebar.button("Submit"):

        for i in selected_classification_models:
            if i == "KNeighborsClassifier":
                KNN_Classifier()
            elif i == "GaussianNB":
                Gaussian_NB()
            elif i == "LogisticRegression":
                Logistic_Regression()
            elif i == "SVC":
                SVM_Classifier()
            elif i == "DecisionTreeClassifier":
                Decision_Tree()
            elif i == "RandomForestClassifier":
                Randon_Forest()
            
    
        for result in results:
            st.subheader(f"**{result['Classifier']}**")
            st.write(f"Best parameters found: {result['Best parameters found']}")
            st.write(f"Best cross-validation score: {result['Best cross-validation score']:.2f}")
            st.write(f"Training time: {result['Training time']:.4f} seconds")
            st.write(f"Testing time: {result['Testing time']:.4f} seconds")
            st.write(f"Roc-Auc score: {result['Roc-Auc score']:.2f}")
            st.write(f"Log-loss: {result['Log-loss']:.2f}")
            st.write("---")
            st.write("Classification Report:")
            st.dataframe(result["Classification Report"])
            st.write("---")
            st.write("### ROC Curve")
            fig, ax = plt.subplots()
            ax.plot(result["FPR"], result["TPR"], label=f"ROC curve (area = {result['Roc-Auc score']:.2f})")
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            st.pyplot(fig)
            st.write("---")
            st.write("### Confusion Matrix")
            cm=ConfusionMatrixDisplay(result['Confusion Matrix'])
            cm.plot()   
            plt.show()
            st.pyplot(plt)
            st.write("---")
            st.write("---")
        
            user_input={}
            for col in cont_cols:
                min_val = int(df[col].min())
                max_val = int(df[col].max())
                user_input[col] = st.slider(f"{col}", min_val, max_val, int((min_val + max_val) / 2), key=f"slider_{col}")

            for col in cat_cols:
                unique_values = df[col].unique()
                user_input[col] = st.selectbox(f"{col}", unique_values, key=f"selectbox_{col}")

            input_df = pd.DataFrame([user_input])

            input_df_processed = final_pl.transform(input_df)

            user_pred_proba = result["Best model"].predict_proba(input_df_processed)[:, 1]

            st.write("### Prediction Result")
            st.write(f"The probability of churn for the given input is: {user_pred_proba[0]:.2f}")

    else:

        st.title("Bank Customer Churn")
        st.header("Problem Statement:", divider="green")
        st.markdown(
            """
            It is advantageous for banks to know what leads a client towards the decision to leave the company. 
            Churn prevention allows companies to develop loyalty programs and retention campaigns to keep as many customers as possible.
            """
        )
        st.subheader("Data Columns")
        st.markdown(
            """
            - **RowNumber**: Corresponds to the record (row) number and has no effect on the output.
            - **CustomerId**: Contains random values and has no effect on customer leaving the bank.
            - **Surname**: The surname of a customer has no impact on their decision to leave the bank.
            - **CreditScore**: Can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
            - **Geography**: A customer's location can affect their decision to leave the bank.
            - **Gender**: It's interesting to explore whether gender plays a role in a customer leaving the bank.
            - **Age**: This is certainly relevant, since older customers are less likely to leave their bank than younger ones.
            - **Tenure**: Refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
            - **Balance**: Also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
            - **NumOfProducts**: Refers to the number of products that a customer has purchased through the bank.
            """
        )

        st.subheader("Data Frame")
        st.dataframe(df)

if __name__ == "__main__":
    main()
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from pycebox.ice import ice, ice_plot
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tpot import TPOTClassifier
import time
from sklearn.metrics import accuracy_score
import joblib

# read in csv file
df = pd.read_csv("c:/Users/kyraa/OneDrive/Documents/GitHub/A04-vwe25002/train_loan_imbalanced.csv")

# examine data
print(df.head())
df.drop('Loan_ID', axis=1, inplace=True)
print(df.shape)

# check for missing values
print(df.isnull().sum())

# drop rows with missing values and check shape again
df.dropna(inplace=True, axis=0)
print(df.shape)

# check class distribution of target variable
print(df['Loan_Status'].value_counts())

# convert target variable to binary values
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# split data into features and target variable
y = df['Loan_Status']
X = df.drop('Loan_Status', axis=1)
print(X.shape, y.shape)

# create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

if __name__ == '__main__':

# create random forest classifier model and fit the model
    RFC = RandomForestClassifier()
    RFC = RFC.fit(X_train, y_train)

    # model predictions
    train_preds_RFC = RFC.predict(X_train)
    test_preds_RFC = RFC.predict(X_test)

    # evaluate RFC model performance
    print(classification_report(y_test, test_preds_RFC))
    print(confusion_matrix(y_test, test_preds_RFC))

    # create figures folder if it doesn't exist
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # clear out old figures in the figures folder before saving new ones
    for f in os.listdir('figures'):
        os.remove(os.path.join('figures', f))

    # Permutation importance for Random Forest Classifier
    clf = RFC

    result = permutation_importance(clf, X_test, y_test, n_repeats=20,
                                    random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()
    perm_sorted_idx_top5 = perm_sorted_idx[-5:]
    top5_features = X.columns[perm_sorted_idx_top5]

    tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
    tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                labels=X.columns[perm_sorted_idx])
    fig.suptitle('RFC Feature Importance', y=1.05)
    fig.tight_layout()
    plt.savefig('figures/RFC_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.boxplot(result.importances[perm_sorted_idx_top5].T, vert=False,
                labels=top5_features)
    fig.suptitle('RFC Feature Importance - Top 5', y=1.05)
    fig.tight_layout()
    plt.savefig('figures/RFC_feature_importance_top5.png', dpi=300, bbox_inches='tight')
    plt.show()

    # create data frame with imputed data
    train_X_df = pd.DataFrame(X_train, columns=X.columns)
    print(train_X_df.head())
    print(train_X_df.shape)

    # print top 5 features
    print(top5_features)

    # create data frames for ICE plots for top 5 features based on RFC
    for feature in top5_features:
        tmpdf = ice(data=train_X_df, # ice needs a dataframe
                    column=feature, # the column name
                           predict=RFC.predict) # the predict statement from the model

        # plot the ICE plot with the PDP line based on RFC
        ice_plot(tmpdf, c='dimgray', linewidth=0.3,
                          plot_pdp=True,
                 pdp_kwargs={'linewidth': 2, 'color':'red'})
        plt.title(f'PDP: {feature}')
        plt.ylabel('Predicted Loan_Status')
        plt.xlabel(feature);
        plt.savefig(f'figures/RFC_ICE_{feature}.png', dpi=300, bbox_inches='tight')
        plt.show()

## Try TPOT classifier on the same data to find the best pipeline
    
    # =========================================================
    # 1) Construct and fit TPOT classifier
    # =========================================================
    start_time = time.time()
    tpot = TPOTClassifier(
        generations=3,      # increased number of generations to improve accuracy
        cv=3,
        random_state=42,
        verbose=2,
        n_jobs=1
    )
    tpot.fit(X_train, y_train)
    end_time = time.time()

    print("TPOT classifier finished in %.2f seconds" % (end_time - start_time))

    # =========================================================
    # 2) Evaluate best pipeline on test set
    # =========================================================
    y_pred = tpot.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Best pipeline test accuracy: %.3f" % acc)

    # =========================================================
    # 3) Inspect the best pipeline
    # =========================================================
    print("\nBest pipeline found by TPOT:\n")
    print(tpot.fitted_pipeline_)

    print("\nPipeline steps (one by one):")
    for name, step in tpot.fitted_pipeline_.steps:
        print(f"\nStep: {name}\n{step}\n")

    # =========================================================
    # 4) Save best pipeline to disk
    # =========================================================
    model_path = "tpot_houseval_pipeline.joblib"
    joblib.dump(tpot.fitted_pipeline_, model_path)
    print(f"Saved best pipeline to {model_path}")
    
    # =========================================================
    # 5) Reload and sanity-check
    # =========================================================
    best_pipe = joblib.load(model_path)
    y_pred_loaded = best_pipe.predict(X_test)
    acc_loaded = accuracy_score(y_test, y_pred_loaded)
    print("Reloaded pipeline test accuracy: %.3f" % acc_loaded)
      
    print(confusion_matrix(y_test, y_pred_loaded))
    print('----------------- ')
    print(classification_report(y_test, y_pred_loaded))

    # =========================================================
    # 6) Permutation importance for TPOT best pipeline
    # =========================================================
    clf2 = best_pipe

    result = permutation_importance(clf2, X_test, y_test, n_repeats=20,
                                    random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()
    perm_sorted_idx_top5 = perm_sorted_idx[-5:]
    top5_features = X.columns[perm_sorted_idx_top5]

    # comment out because TPOT pipeline may not have feature_importances_ attribute and they aren't referenced below
    # tree_importance_sorted_idx = np.argsort(clf2.feature_importances_)
    # tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                labels=X.columns[perm_sorted_idx])
    fig.suptitle('TPOT Feature Importance', y=1.05)
    fig.tight_layout()
    plt.savefig('figures/TPOT_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.boxplot(result.importances[perm_sorted_idx_top5].T, vert=False,
                labels=top5_features)
    fig.suptitle('TPOT Feature Importance - Top 5', y=1.05)
    fig.tight_layout()
    plt.savefig('figures/TPOT_feature_importance_top5.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =========================================================
    # 7) create data frame with imputed data
    # =========================================================
    train_X_df = pd.DataFrame(X_train, columns=X.columns)
    print(train_X_df.head())
    print(train_X_df.shape)

    # print top 5 features
    print(top5_features)

    # =========================================================
    # 8) create data frames for ICE plots for top 5 features
    # =========================================================
    for feature in top5_features:
        tmpdf = ice(data=train_X_df, # ice needs a dataframe
                    column=feature, # the column name
                           predict=best_pipe.predict) # the predict statement from the model

    # =========================================================
    # 9) plot the ICE plot with the PDP line
    # =========================================================
        ice_plot(tmpdf, c='dimgray', linewidth=0.3,
                          plot_pdp=True,
                 pdp_kwargs={'linewidth': 2, 'color':'red'})
        plt.title(f'PDP: {feature}')
        plt.ylabel('Predicted Loan_Status')
        plt.xlabel(feature);
        plt.savefig(f'figures/TPOT_ICE_{feature}.png', dpi=300, bbox_inches='tight')
        plt.show()
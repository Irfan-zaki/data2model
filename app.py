import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.metrics import mean_squared_error, accuracy_score,r2_score,f1_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import streamlit.components.v1 as components
import xgboost as xgb
from scipy import stats
import time
import numpy as np 
import pandas as pd
import streamlit as st
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix # For classification report
from sklearn.cluster import KMeans # Example for unsupervised
from sklearn.decomposition import PCA # Example for unsupervised
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px # For interactive plots
import matplotlib.pyplot as plt # For dendrogram
from scipy.cluster.hierarchy import dendrogram # For dendrogram
import pandas as pd # If not already imported for DataFrames
import numpy as np # If not already imported for numerical operations
from sklearn.decomposition import PCA # Import PCA if not already imported

st.set_page_config(layout="wide")
import streamlit as st

st.markdown("<h1 style='text-align: center; color: yellow;'>DATA2MODEL</h1>", unsafe_allow_html=True)
import streamlit as st
with st.container(border=False):
    st.markdown("<h4 style='text-align: center; color: white;'>Transform Raw Data into Powerful Insights & AI Models—Effortlessly!😉😎</h4>", unsafe_allow_html=True)
import streamlit as st
# Initialize session state for cleaned data
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None

tab1, tab2 = st.tabs(['Data Refinery', 'Data Visualization'])

@st.cache_data
def handle_missing_values(df):
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({"Column": df.columns, "Missing (%)": missing_percent})
    missing_data = missing_data[missing_data["Missing (%)"] > 0]

    if not missing_data.empty:
        st.write("🔍 **Missing Values Overview**:")
        st.dataframe(missing_data.style.format({"Missing (%)": "{:.2f}"}))

        for column in missing_data["Column"]:
            if df[column].dtype in ['int64', 'float64']:
                # Check skewness for best fill method
                skewness = df[column].skew()
                if abs(skewness) < 1:  # Normal distribution
                    df[column] = df[column].fillna(df[column].mean())
                    time.sleep(1.5)  # Add a delay of 1 second before showing the success message
                    st.success(f"✅ Filled missing values in **{column}** using **Mean**.")
                else:  # Skewed data
                    df[column] = df[column].fillna(df[column].median())
                    time.sleep(1.5)  # Add a delay of 1 second before showing the success message
                    st.success(f"✅ Filled missing values in **{column}** using **Median**.")
            else:
                df[column] = df[column].fillna(df[column].mode().iloc[0])
                time.sleep(1.5)  # Add a delay of 1 second before showing the success message
                st.success(f"✅ Filled missing values in **{column}** using **Mode**.")
    return df

# Function to clean the dataset
@st.cache_data
def clean_data(df, columns_to_drop=None):
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    return df

@st.cache_data
def scale_data(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create a copy of the dataframe to store the scaled data
    scaled_df = df.copy()
    scaling_info = {}
    scalers = {} # Dictionary to store scalers

    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    scaler_robust = RobustScaler()

    for col in numeric_cols:
        # Perform Shapiro-Wilk test for normality
        stat, p_value = shapiro(df[col].dropna())  # Drop NaN for the test
        skewness = df[col].skew()  # Check skewness level

        # Apply appropriate scaling method
        if p_value > 0.05:
            scaled_df[col] = scaler_standard.fit_transform(df[[col]])
            scaling_info[col] = "Standardization (Z-score)"
            scalers[col] = scaler_standard # Store scaler
        elif abs(skewness) > 1.5:
            scaled_df[col] = scaler_robust.fit_transform(df[[col]])
            scaling_info[col] = "Robust Scaling (Median & IQR)"
            scalers[col] = scaler_robust # Store scaler
        else:
            scaled_df[col] = scaler_minmax.fit_transform(df[[col]])
            scaling_info[col] = "Min-Max Scaling (0-1 Range)"
            scalers[col] = scaler_minmax # Store scaler

    # Display Scaling Information
    time.sleep(1.5)
    st.subheader("✅ **Scaling Applied Successfully**")
    for col, method in scaling_info.items():
        time.sleep(0.5)
        st.write(f"📌 **{col}**: {method}")

    return scaled_df,scalers

import pandas as pd
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce

import pandas as pd
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce

import streamlit as st
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce

@st.cache_data
def encode_categorical_columns(df, ordinal_encoding_config): # Add ordinal_encoding_config as argument
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoding_info = {}

    if categorical_cols.empty:
        st.warning("⚠️ No categorical columns found for encoding.")
        return df

    for col in categorical_cols:
        unique_values = df[col].nunique()
        total_rows = len(df)
        cardinality_ratio = unique_values / total_rows

        # Check if ordinal encoding is configured for this column
        if col in ordinal_encoding_config and ordinal_encoding_config[col]['apply_ordinal']:
            order = ordinal_encoding_config[col]['order']
            if order:
                ordered_categories = [cat.strip() for cat in order.split(',')]
                encoder = OrdinalEncoder(categories=[ordered_categories])
                df[col] = encoder.fit_transform(df[[col]])
                encoding_info[col] = f"Ordinal Encoding (Ordered categories) applied to {col}"

        # One-Hot Encoding (Low Cardinality, Nominal Data)
        elif unique_values <= 10:
            encoded_df = pd.get_dummies(df[[col]], prefix=col, drop_first=True)
            df = df.drop(columns=[col]).join(encoded_df)
            encoding_info[col] = f"One-Hot Encoding (Low cardinality, unordered categories) applied to {col}"

        # Hash Encoding (High Cardinality)
        elif cardinality_ratio > 0.05:
            encoder = ce.HashingEncoder(cols=[col], n_components=8)
            encoded_df = encoder.fit_transform(df[[col]])
            encoded_df.columns = [f"{col}_hash_{i}" for i in range(encoded_df.shape[1])]
            df = df.drop(columns=[col]).join(encoded_df)
            encoding_info[col] = f"Hash Encoding (High cardinality feature) applied to {col}"

        # Frequency Encoding (Category Frequencies Affecting Target)
        else:
            freq_map = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_map)
            encoding_info[col] = f"Frequency Encoding (Based on category occurrence frequency) applied to {col}"

    # Display Encoding Information
    if encoding_info:
        st.subheader("✅ **Encoding Applied Successfully!**")
        for col, method in encoding_info.items():
            st.write(f"📌 **{col}**: {method}")

    return df

algorithm_options = {
    "Classification": {
        # ... (Classification algorithms - already defined) ...
        "Logistic Regression": {
            "description": "A linear model for binary or multiclass classification. Good for baseline models and interpretable results.",
            "parameters": {
                "penalty": {
                    "type": "selectbox",
                    "label": "Penalty",
                    "options": ["l1", "l2", "elasticnet", "none"],
                    "default": "l2",
                    "help": "Regularization penalty norm ('l1', 'l2', 'elasticnet', 'none').",
                },
                "C": {
                    "type": "slider",
                    "label": "C (Regularization strength)",
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.1,
                    "default": 1.0,
                    "help": "Inverse of regularization strength; smaller values specify stronger regularization.",
                },
                "solver": {
                    "type": "selectbox",
                    "label": "Solver",
                    "options": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                    "default": "lbfgs",
                    "help": "Algorithm to use in the optimization problem.",
                },
            },
            "model": "LogisticRegression",
            "library": "sklearn.linear_model"
        },
        "Decision Tree": {
            "description": "A tree-based model that makes decisions based on feature values. Easy to interpret, but can overfit.",
            "parameters": {
                "criterion": {
                    "type": "selectbox",
                    "label": "Criterion",
                    "options": ["gini", "entropy"],
                    "default": "gini",
                    "help": "The function to measure the quality of a split.",
                },
                "max_depth": {
                    "type": "slider",
                    "label": "Max Depth",
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "default": None, # Important to handle None correctly in model instantiation
                    "help": "The maximum depth of the tree. Controls complexity and overfitting.",
                },
                "min_samples_split": {
                    "type": "slider",
                    "label": "Min Samples Split",
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "default": 2,
                    "help": "The minimum number of samples required to split an internal node.",
                },
            },
            "model": "DecisionTreeClassifier",
            "library": "sklearn.tree"
        },
        "KNN": {
            "description": "K-Nearest Neighbors classifier. Instance-based learning, classifies based on majority vote of neighbors.",
            "parameters": {
                "n_neighbors": {
                    "type": "slider",
                    "label": "Number of Neighbors (k)",
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "default": 5,
                    "help": "Number of neighbors to consider when classifying a new point.",
                },
                "weights": {
                    "type": "selectbox",
                    "label": "Weights",
                    "options": ["uniform", "distance"],
                    "default": "uniform",
                    "help": "'uniform' - all neighbors weighted equally, 'distance' - closer neighbors have more weight.",
                },
                "algorithm": {
                    "type": "selectbox",
                    "label": "Algorithm",
                    "options": ["auto", "ball_tree", "kd_tree", "brute"],
                    "default": "auto",
                    "help": "Algorithm used to compute the nearest neighbors: 'ball_tree', 'kd_tree', 'brute', 'auto'.",
                },
            },
            "model": "KNeighborsClassifier",
            "library": "sklearn.neighbors"
        },
        "SVM": {
            "description": "Support Vector Machine classifier. Effective in high dimensional spaces, versatile with different kernel functions.",
            "parameters": {
                "C": {
                    "type": "slider",
                    "label": "C (Regularization)",
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.1,
                    "default": 1.0,
                    "help": "Regularization parameter. Smaller values mean stronger regularization.",
                },
                "kernel": {
                    "type": "selectbox",
                    "label": "Kernel",
                    "options": ["linear", "poly", "rbf", "sigmoid"],
                    "default": "rbf",
                    "help": "Kernel type to be used in the algorithm: 'linear', 'poly', 'rbf', 'sigmoid'.",
                },
                "gamma": {
                    "type": "selectbox",
                    "label": "Gamma (Kernel coefficient)",
                    "options": ["scale", "auto"],
                    "default": "scale",
                    "help": "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 'scale' (default) uses 1 / (n_features * X.var()).",
                },
                "probability": {
                    "type": "checkbox",
                    "label": "Probability",
                    "default": False,
                    "help": "Whether to enable probability estimates. Slows down training.",
                },
            },
            "model": "SVC", # SVC for classification, SVR for regression
            "library": "sklearn.svm"
        },
        "Random Forest": {
            "description": "Random Forest classifier. Ensemble of decision trees, reduces overfitting and improves generalization.",
            "parameters": {
                "n_estimators": {
                    "type": "slider",
                    "label": "Number of Estimators",
                    "min": 10,
                    "max": 200,
                    "step": 10,
                    "default": 100,
                    "help": "The number of trees in the forest.",
                },
                "criterion": {
                    "type": "selectbox",
                    "label": "Criterion",
                    "options": ["gini", "entropy"],
                    "default": "gini",
                    "help": "The function to measure the quality of a split.",
                },
                "max_depth": {
                    "type": "slider",
                    "label": "Max Depth",
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "default": None,
                    "help": "The maximum depth of the tree. Controls complexity and overfitting.",
                },
                "min_samples_split": {
                    "type": "slider",
                    "label": "Min Samples Split",
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "default": 2,
                    "help": "The minimum number of samples required to split an internal node.",
                },
                "bootstrap": {
                    "type": "checkbox",
                    "label": "Bootstrap",
                    "default": True,
                    "help": "Whether bootstrap samples are used when building trees.",
                },
            },
            "model": "RandomForestClassifier",
            "library": "sklearn.ensemble"
        },
        "CatBoost": {
            "description": "CatBoost classifier. Gradient boosting algorithm, handles categorical features natively, robust and accurate.",
            "parameters": {
                "iterations": {
                    "type": "slider",
                    "label": "Iterations",
                    "min": 50,
                    "max": 500,
                    "step": 50,
                    "default": 100,
                    "help": "Maximum number of iterations for training.",
                },
                "learning_rate": {
                    "type": "slider",
                    "label": "Learning Rate",
                    "min": 0.01,
                    "max": 0.3,
                    "step": 0.01,
                    "default": 0.1,
                    "help": "Learning rate. Decreases the contribution of each tree.",
                },
                "depth": {
                    "type": "slider",
                    "label": "Depth",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "default": 6,
                    "help": "Depth of the trees.",
                },
                "l2_leaf_reg": {
                    "type": "slider",
                    "label": "L2 Leaf Regularization",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "default": 3,
                    "help": "L2 regularization coefficient. Controls overfitting.",
                },
                "verbose": {
                    "type": "checkbox",
                    "label": "Verbose",
                    "default": False,
                    "help": "Set to True to display training progress.",
                },
            },
            "model": "CatBoostClassifier",
            "library": "catboost" # Note: Requires 'pip install catboost'
        },
        "Naive Bayes": {
            "description": "Gaussian Naive Bayes classifier. Simple probabilistic classifier, efficient and works well with high-dimensional data.",
            "parameters": {
                "var_smoothing": {
                    "type": "slider",
                    "label": "Variance Smoothing",
                    "min": 1e-9,
                    "max": 1e-6,
                    "step": 1e-7,
                    "default": 1e-9,
                    "format": "%e", # Use scientific notation for display
                    "help": "Portion of the largest variance of all features that is added to variances for calculation stability.",
                },
            },
            "model": "GaussianNB",
            "library": "sklearn.naive_bayes"
        },
    },
    "Regression": {
        # ... (Regression algorithms - already defined) ...
        "Linear Regression": {
            "description": "A linear model that finds the best-fitting line to predict continuous values.",
            "parameters": {
                "fit_intercept": {
                    "type": "checkbox",
                    "label": "Fit Intercept",
                    "default": True,
                    "help": "Whether to calculate the intercept for this model. If set to False, no intercept will be used.",
                }
            },
             "model": "LinearRegression",
            "library": "sklearn.linear_model"
        },
        "Polynomial Regression": {
            "description": "Polynomial regression extends linear regression by adding polynomial features to the input variables.",
            "parameters": {
                "degree": {
                    "type": "slider",
                    "label": "Polynomial Degree",
                    "min": 2,
                    "max": 5,
                    "step": 1,
                    "default": 2,
                    "help": "Degree of the polynomial features.",
                },
                "include_bias": {
                    "type": "checkbox",
                    "label": "Include Bias (Intercept)",
                    "default": True,
                    "help": "Whether to include a bias column (intercept) in polynomial features.",
                },
                "interaction_only": {
                    "type": "checkbox",
                    "label": "Interaction Only",
                    "default": False,
                    "help": "If True, only interaction features are produced: features that are products of at most degree distinct input features (so no powers of single features).",
                },
            },
            "model": "PolynomialFeatures",
            "library": "sklearn.preprocessing" # Using preprocessing library for PolynomialFeatures
        },
        "Decision Tree": {
            "description": "Decision tree regressor. Non-linear model, can capture complex relationships, but prone to overfitting.",
            "parameters": {
                "criterion": {
                    "type": "selectbox",
                    "label": "Criterion",
                    "options": ["squared_error", "friedman_mse", "absolute_error", "poisson"], # Regression criteria
                    "default": "squared_error",
                    "help": "The function to measure the quality of a split.",
                },
                "max_depth": {
                    "type": "slider",
                    "label": "Max Depth",
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "default": None,
                    "help": "The maximum depth of the tree. Controls complexity and overfitting.",
                },
                "min_samples_split": {
                    "type": "slider",
                    "label": "Min Samples Split",
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "default": 2,
                    "help": "The minimum number of samples required to split an internal node.",
                },
                "max_features": {
                    "type": "selectbox",
                    "label": "Max Features",
                    "options": ["auto", "sqrt", "log2", None],
                    "default": None,
                    "help": "The number of features to consider when looking for the best split.",
                },
            },
            "model": "DecisionTreeRegressor",
            "library": "sklearn.tree"
        },
        "KNN": {
            "description": "K-Nearest Neighbors regressor. Predicts based on the average of the target values of its nearest neighbors.",
            "parameters": {
                "n_neighbors": {
                    "type": "slider",
                    "label": "Number of Neighbors (k)",
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "default": 5,
                    "help": "Number of neighbors to consider when predicting a new point.",
                },
                "weights": {
                    "type": "selectbox",
                    "label": "Weights",
                    "options": ["uniform", "distance"],
                    "default": "uniform",
                    "help": "'uniform' - all neighbors weighted equally, 'distance' - closer neighbors have more weight.",
                },
                "algorithm": {
                    "type": "selectbox",
                    "label": "Algorithm",
                    "options": ["auto", "ball_tree", "kd_tree", "brute"],
                    "default": "auto",
                    "help": "Algorithm used to compute the nearest neighbors: 'ball_tree', 'kd_tree', 'brute', 'auto'.",
                },
            },
            "model": "KNeighborsRegressor",
            "library": "sklearn.neighbors"
        },
        "Random Forest": {
            "description": "Random Forest regressor. Ensemble of decision trees for regression, robust and often provides good performance.",
            "parameters": {
                "n_estimators": {
                    "type": "slider",
                    "label": "Number of Estimators",
                    "min": 10,
                    "max": 200,
                    "step": 10,
                    "default": 100,
                    "help": "The number of trees in the forest.",
                },
                "criterion": {
                    "type": "selectbox",
                    "label": "Criterion",
                    "options": ["squared_error", "absolute_error", "poisson"], # Regression criteria
                    "default": "squared_error",
                    "help": "The function to measure the quality of a split.",
                },
                "max_depth": {
                    "type": "slider",
                    "label": "Max Depth",
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "default": None,
                    "help": "The maximum depth of the tree. Controls complexity and overfitting.",
                },
                "min_samples_split": {
                    "type": "slider",
                    "label": "Min Samples Split",
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "default": 2,
                    "help": "The minimum number of samples required to split an internal node.",
                },
                "bootstrap": {
                    "type": "checkbox",
                    "label": "Bootstrap",
                    "default": True,
                    "help": "Whether bootstrap samples are used when building trees.",
                },
            },
            "model": "RandomForestRegressor",
            "library": "sklearn.ensemble"
        },
        "Gradient Boosting": {
            "description": "Gradient Boosting regressor. Ensemble method, builds trees sequentially, correcting errors of previous trees.",
            "parameters": {
                "n_estimators": {
                    "type": "slider",
                    "label": "Number of Estimators",
                    "min": 50,
                    "max": 200,
                    "step": 10,
                    "default": 100,
                    "help": "The number of boosting stages to perform.",
                },
                "learning_rate": {
                    "type": "slider",
                    "label": "Learning Rate",
                    "min": 0.01,
                    "max": 0.3,
                    "step": 0.01,
                    "default": 0.1,
                    "help": "Learning rate shrinks the contribution of each tree.",
                },
                "max_depth": {
                    "type": "slider",
                    "label": "Max Depth",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "default": 3,
                    "help": "Maximum depth of the individual regression estimators.",
                },
                "loss": {
                    "type": "selectbox",
                    "label": "Loss Function",
                    "options": ["squared_error", "absolute_error", "huber", "quantile"], # Regression loss functions
                    "default": "squared_error",
                    "help": "Loss function to be optimized. 'squared_error' for ordinary least squares, 'absolute_error' for L1 loss.",
                },
            },
            "model": "GradientBoostingRegressor",
            "library": "sklearn.ensemble"
        },
        "SVR": {
            "description": "Support Vector Regressor. Effective in high dimensional spaces, versatile with different kernel functions for regression.",
            "parameters": {
                "C": {
                    "type": "slider",
                    "label": "C (Regularization)",
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.1,
                    "default": 1.0,
                    "help": "Regularization parameter. Smaller values mean stronger regularization.",
                },
                "kernel": {
                    "type": "selectbox",
                    "label": "Kernel",
                    "options": ["linear", "poly", "rbf", "sigmoid"],
                    "default": "rbf",
                    "help": "Kernel type to be used in the algorithm: 'linear', 'poly', 'rbf', 'sigmoid'.",
                },
                "gamma": {
                    "type": "selectbox",
                    "label": "Gamma (Kernel coefficient)",
                    "options": ["scale", "auto"],
                    "default": "scale",
                    "help": "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 'scale' (default) uses 1 / (n_features * X.var()).",
                },
                "epsilon": {
                    "type": "slider",
                    "label": "Epsilon",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "default": 0.1,
                    "help": "Epsilon parameter in the epsilon-insensitive loss function. Specifies the epsilon-tube within which no penalty is associated in the training loss function.",
                },
            },
            "model": "SVR",
            "library": "sklearn.svm"
        },
    },
    "Unsupervised Learning": {
        "K-Means": {
            "description": "Partitions data into k clusters. Simple and effective for clustering, requires specifying the number of clusters.",
            "parameters": {
                "n_clusters": {
                    "type": "slider",
                    "label": "Number of Clusters (k)",
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "default": 3,
                    "help": "The number of clusters to form as well as the number of centroids to generate.",
                },
                "init": {
                    "type": "selectbox",
                    "label": "Initialization method",
                    "options": ["k-means++", "random"],
                    "default": "k-means++",
                    "help": "Method for initialization: 'k-means++' for smart initialization, 'random' for random initialization.",
                },
                "max_iter": {
                    "type": "slider",
                    "label": "Max Iterations",
                    "min": 100,
                    "max": 500,
                    "step": 50,
                    "default": 300,
                    "help": "Maximum number of iterations of the k-means algorithm for a single run.",
                },
                "random_state": { # Added random_state for reproducibility
                    "type": "slider",
                    "label": "Random State",
                    "min": 0,
                    "max": 42,
                    "step": 1,
                    "default": 0,
                    "help": "Random state for reproducibility.",
                },
            },
            "model": "KMeans",
            "library": "sklearn.cluster"
        },
        "Hierarchical Clustering": {
            "description": "Hierarchical clustering builds a tree of clusters. Useful for visualizing hierarchical relationships.",
            "parameters": {
                "n_clusters": { # Optional n_clusters for AgglomerativeClustering
                    "type": "slider",
                    "label": "Number of Clusters (optional)",
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "default": 2,
                    "help": "Number of clusters to find. If None, the algorithm will produce a hierarchy.",
                    "optional": True # Mark as optional
                },
                "linkage": {
                    "type": "selectbox",
                    "label": "Linkage Method",
                    "options": ["ward", "complete", "average", "single"],
                    "default": "ward",
                    "help": "Which linkage criterion to use. 'ward' minimizes the variance of all clusters.",
                },
                "affinity": {
                    "type": "selectbox",
                    "label": "Affinity Metric",
                    "options": ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
                    "default": "euclidean",
                    "help": "Metric used to compute the linkage. 'euclidean' is standard Euclidean distance.",
                },
            },
            "model": "AgglomerativeClustering", # Using AgglomerativeClustering for hierarchical
            "library": "sklearn.cluster"
        },
        "DBSCAN": {
            "description": "Density-Based Spatial Clustering of Applications with Noise. Finds clusters of arbitrary shape and marks outliers.",
            "parameters": {
                "eps": {
                    "type": "slider",
                    "label": "Epsilon (eps)",
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "default": 0.5,
                    "help": "The maximum distance between two samples for one to be considered as in the neighborhood of the other.",
                },
                "min_samples": {
                    "type": "slider",
                    "label": "Min Samples",
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "default": 5,
                    "help": "The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.",
                },
                "metric": {
                    "type": "selectbox",
                    "label": "Metric",
                    "options": ["euclidean", "manhattan", "chebyshev", "minkowski"],
                    "default": "euclidean",
                    "help": "The metric to use when calculating distance between instances in a feature array.",
                },
            },
            "model": "DBSCAN",
            "library": "sklearn.cluster"
        },
        "PCA": {
            "description": "Reduces dimensionality by finding principal components. Useful for visualization and feature extraction.",
            "parameters": {
                "n_components": {
                    "type": "slider",
                    "label": "Number of Components",
                    "min": 1,
                    "max": st.session_state.cleaned_data.shape[1] -1 if 'cleaned_data' in st.session_state and st.session_state.cleaned_data is not None else 2, # Dynamically set max
                    "step": 1,
                    "default": 2,
                    "help": "Number of principal components to keep. If None, all components are kept.",
                },
                "svd_solver": {
                    "type": "selectbox",
                    "label": "SVD Solver",
                    "options": ["auto", "full", "arpack", "randomized"],
                    "default": "auto",
                    "help": "SVD solver to use. 'auto' chooses the solver automatically based on the data and size.",
                },
                "whiten": {
                    "type": "checkbox",
                    "label": "Whiten",
                    "default": False,
                    "help": "When True (False by default) the components vectors are multiplied by r = n_samples / singular_values ** 2.",
                },
                "random_state": { # Added random_state for reproducibility
                    "type": "slider",
                    "label": "Random State",
                    "min": 0,
                    "max": 42,
                    "step": 1,
                    "default": 0,
                    "help": "Random state for reproducibility.",
                },
            },
            "model": "PCA",
            "library": "sklearn.decomposition"
        },
        "Isolation Forest": {
            "description": "Isolation Forest for anomaly detection. Isolates anomalies by randomly partitioning the data space.",
            "parameters": {
                "n_estimators": {
                    "type": "slider",
                    "label": "Number of Estimators",
                    "min": 50,
                    "max": 200,
                    "step": 10,
                    "default": 100,
                    "help": "The number of trees in the forest.",
                },
                "contamination": {
                    "type": "slider",
                    "label": "Contamination",
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "default": 0.1,
                    "help": "The amount of contamination of the data set, i.e. the proportion of outliers in the data set.",
                    "format": "%.2f" # Format to two decimal places
                },
                "max_samples": {
                    "type": "selectbox",
                    "label": "Max Samples",
                    "options": ["auto", "integer", "float"], # Can be 'auto' or integer or float
                    "default": "auto",
                    "help": "The number of samples to draw from X to train each base estimator.",
                },
                "random_state": { # Added random_state for reproducibility
                    "type": "slider",
                    "label": "Random State",
                    "min": 0,
                    "max": 42,
                    "step": 1,
                    "default": 0,
                    "help": "Random state for reproducibility.",
                },
            },
            "model": "IsolationForest",
            "library": "sklearn.ensemble"
        },
    },
}


with tab1:
    st.header("Data Cleaning and Machine Learning Simulator")
    import streamlit as st

    with st.expander("This tab helps you prepare your dataset for machine learning by automating data cleaning and model training. Expand here to know how the process works:👇"):
        st.write("1. **Upload Your Dataset** – Start by uploading your raw data file.🙂")  
        st.write("2. **Remove Unwanted Columns** – Select and drop any columns you don’t need.😃")  
        st.write("3. **Handle Missing Values** – Automatically fills missing data or removes incomplete rows based on the best approach.🙃")  
        st.write("4. **Encode Categorical Data** – Converts text-based categories into numbers so the model can understand them.😉")  
        st.write("5. **Scale Numerical Features** – Normalizes numbers so that all features contribute equally to the model.🙂")  
        st.write("6. **Train a Machine Learning Model** – Uses the cleaned dataset to train a model that can make predictions.🙃")  
        st.write("7. **Make Predictions** – Once the model is trained, use it to predict the target variable on new data.😃")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            st.session_state.original_data = data.copy() # Store a copy of the ORIGINAL data in session state
            st.write("Data Preview:")
            st.dataframe(data)

            # Data Cleaning
            columns_to_drop = st.multiselect("Select Irrelevant columns to drop", data.columns)
            st.session_state.cleaned_data = clean_data(data, columns_to_drop)

            st.session_state.cleaned_data = handle_missing_values(st.session_state.cleaned_data)  # Call the function to handle missing values
            st.session_state.cleaned_data, feature_scalers = scale_data(st.session_state.cleaned_data) # Get scalers back
            st.session_state['feature_scalers'] = feature_scalers # Store feature scalers in session state
            ordinal_encoding_config = {} # Initialize configuration dictionary

            if st.session_state.cleaned_data is not None: # Check if cleaned_data exists
                categorical_cols_for_encoding = st.session_state.cleaned_data.select_dtypes(include=['object']).columns


            st.session_state.cleaned_data=encode_categorical_columns(st.session_state.cleaned_data, ordinal_encoding_config) # Pass config to cached function
            
            
            if "cleaned_data" in st.session_state:
                st.write("**Cleaned Data Preview**:")
                st.dataframe(st.session_state.cleaned_data)  # Just for display

                # Convert DataFrame to CSV
                csv = st.session_state.cleaned_data.to_csv(index=False).encode("utf-8")

                # Add download button
                st.download_button(
                    label="Download your cleaned data",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )

            if st.button("Show Statistics"):
                st.write("Statistics:")
                st.dataframe(data.describe())
                
    



            import streamlit as st
            import pandas as pd
            

            # Assuming st.session_state.cleaned_data is already loaded in your app
            if 'cleaned_data' not in st.session_state:
                st.warning("Please upload and clean your data first.")
                st.stop()

            st.sidebar.header("Model Training")

            # ---------------------- Problem Type Selection ----------------------
            problem_type = st.sidebar.selectbox("Select Problem Type", ["Classification", "Regression", "Unsupervised Learning"])

            # ---------------------- Algorithm Selection (Dynamic) ----------------------
            algorithm_name = None # Initialize outside if block
            if problem_type: # Only proceed if problem type is selected
                algorithm_options_for_type = algorithm_options[problem_type]
                algorithm_names = list(algorithm_options_for_type.keys())
                algorithm_name = st.sidebar.selectbox("Select Algorithm", algorithm_names)

            # ---------------------- Feature and Target Selection (Conditional) ----------------------
            target_variable = None
            selected_features = []

            if problem_type != "Unsupervised Learning": # Feature/Target selection only for supervised learning
                target_variable = st.sidebar.selectbox("Select Target Variable", st.session_state.cleaned_data.columns)
                available_features = st.session_state.cleaned_data.columns.drop(target_variable, errors='ignore')
                all_option = "Select All"
                selected_features_options = [all_option] + list(available_features)
                selected_features = st.sidebar.multiselect(
                    "Select Input Features",
                    options=selected_features_options,
                    default=list(available_features)
                )
                if all_option in selected_features:
                    selected_features = list(available_features)
                st.sidebar.write("Selected Input Features:", selected_features)

            # ---------------------- Parameter Selection (Dynamic based on Algorithm) ----------------------
            params = {} # Dictionary to store parameter values
            if algorithm_name:
                st.sidebar.subheader("Algorithm Parameters")
                selected_algorithm_options = algorithm_options[problem_type][algorithm_name]

                st.sidebar.write(selected_algorithm_options["description"]) # Show algorithm description

                if "parameters" in selected_algorithm_options: # Check if parameters are defined
                    for param_name, param_details in selected_algorithm_options["parameters"].items():
                        param_label = param_details["label"]
                        param_type = param_details["type"]
                        param_default = param_details["default"]
                        param_help = param_details["help"]

                        st.sidebar.caption(param_help) # Display parameter description

                        if param_type == "slider":
                            params[param_name] = st.sidebar.slider(param_label,
                                                                    min_value=param_details["min"],
                                                                    max_value=param_details["max"],
                                                                    step=param_details["step"],
                                                                    value=param_default)
                        elif param_type == "selectbox":
                            param_key = f"selectbox_{param_name}"  # Create a unique key for each selectbox

                            # Check if the session state already has a value for this selectbox
                            if param_key not in st.session_state:
                                st.session_state[param_key] = param_default  # Set the default value in session state if it's not there yet

                            params[param_name] = st.sidebar.selectbox(
                            param_label,
                            options=param_details["options"],
                            key=param_key  # Use the key to link to session state
                                )
                        elif param_type == "checkbox":
                            params[param_name] = st.sidebar.checkbox(param_label, value=param_default)

            # ---------------------- Train Model Button and Model Training ----------------------
            train_button = st.sidebar.button("Train Model", key="train_model_button")

            if train_button:
                st.header("Model Training Results")

                if problem_type != "Unsupervised Learning": # Supervised Learning Check
                    if not selected_features or not target_variable:
                        st.error("Please select both target variable and input features for supervised learning.")
                    else:
                        try:
                            algorithm_info = algorithm_options[problem_type][algorithm_name]
                            model_class_name = algorithm_info["model"]
                            model_library = algorithm_info["library"]

                            # Dynamically import the model class
                            model_module = __import__(model_library, fromlist=[model_class_name])
                            model_class = getattr(model_module, model_class_name)

                            # Instantiate the model with selected parameters
                            model = model_class(**params)  # Pass parameter dictionary

                            if problem_type != "Unsupervised Learning": 
                                # Supervised Learning
                                
                                X = st.session_state.cleaned_data[selected_features]
                                y = st.session_state.cleaned_data[target_variable]
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Example split
                                if algorithm_name == "Polynomial Regression":
                                                            poly_features = PolynomialFeatures(degree=params['degree'],
                                                                                                include_bias=params['include_bias'],
                                                                                                interaction_only=params['interaction_only'])
                                                            X_poly_train = poly_features.fit_transform(X_train)
                                                            X_poly_test = poly_features.transform(X_test) # Use transform on test data

                                                            # **Correct Model Instantiation: Use LinearRegression, not PolynomialFeatures**
                                                            model = LinearRegression(fit_intercept=False) # Instantiate LinearRegression model
                                                            model.fit(X_poly_train, y_train) # Fit LinearRegression on POLYNOMIAL features
                                                            y_pred = model.predict(X_poly_test) # Predict using the LINEAR REGRESSION MODEL

                                model.fit(X_train, y_train) # Train the model
                                st.session_state['trained_model'] = model  # Store 'model' in session state with key 'trained_model'
                                y_pred = model.predict(X_test)
                            

                                st.subheader("Your Model Report") # Or Regression Metrics, etc. based on problem_type
                                if problem_type == "Classification":
                                    report = classification_report(y_test, y_pred)
                                    st.text(report) # Display classification report
                                    st.subheader("Confusion Matrix")
                                    conf_matrix = confusion_matrix(y_test, y_pred)
                                    st.write(pd.DataFrame(conf_matrix)) # Display confusion matrix as DataFrame
                                elif problem_type == "Regression":
                                    # Calculate regression metrics
                                    r2 = r2_score(y_test, y_pred)
                                    mae = mean_absolute_error(y_test, y_pred)
                                    mse = mean_squared_error(y_test, y_pred)
                                    rmse = np.sqrt(mse) # Root Mean Squared Error

                                    st.subheader("Regression Metrics")

                                    col1, col2, col3, col4 = st.columns(4) # Create columns for better layout

                                    with col1:
                                        st.metric("R-squared (R2)", value=f"{r2:.3f}") # Format to 3 decimal places
                                    with col2:
                                        st.metric("Mean Absolute Error (MAE)", value=f"{mae:.3f}")
                                    with col3:
                                        st.metric("Mean Squared Error (MSE)", value=f"{mse:.3f}")
                                    with col4:
                                        st.metric("Root Mean Squared Error (RMSE)", value=f"{rmse:.3f}")

                                    st.write("---") # Separator line

                                    st.write("**Interpretation of Metrics:**")
                                    st.write("- **R-squared (R2):**  Represents the proportion of variance in the dependent variable that is predictable from the independent variables.  Ranges from 0 to 1 (and can be negative for very poor models). Higher is better, closer to 1 indicates a better fit.")
                                    st.write("- **Mean Absolute Error (MAE):**  Average magnitude of the errors in predictions, without considering their direction. Lower is better, closer to 0 indicates more accurate predictions.")
                                    st.write("- **Mean Squared Error (MSE):** Average of the squared differences between predictions and actual values.  Squaring penalizes larger errors more heavily than MAE. Lower is better, closer to 0 indicates more accurate predictions.")
                                    st.write("- **Root Mean Squared Error (RMSE):**  Square root of MSE.  It's in the same units as the target variable, making it sometimes more interpretable than MSE. Lower is better, closer to 0 indicates more accurate predictions.")
                        except Exception as e:
                            st.error(f"Error during model training: {e}")
                            st.error("Please check your data, selected features, target variable, and parameters.")
                            st.exception(e) # Show full exception details for debugging
                            
                elif problem_type == "Unsupervised Learning": # Unsupervised Learning
                    try:
                        X = st.session_state.cleaned_data[selected_features if selected_features else st.session_state.cleaned_data.columns] # Use all columns if no features selected for unsupervised

                        # Dynamically import the model class
                        algorithm_info = algorithm_options[problem_type][algorithm_name]
                        model_class_name = algorithm_info["model"]
                        model_library = algorithm_info["library"]
                        model_module = __import__(model_library, fromlist=[model_class_name])
                        model_class = getattr(model_module, model_class_name)

                        # Instantiate the model with selected parameters
                        model = model_class(**params)

                        if X.empty: # Check if X is empty *before* fitting
                            st.error("Error: Input data (features) is empty. Please ensure your data is loaded and features are selected (if applicable).")
                            raise ValueError("Empty input data for unsupervised model.") # Raise an exception to be caught

                        model.fit(X) # Fit unsupervised model
                        st.session_state['trained_model'] = model # Store 'model' in session state with key 'trained_model

                        st.subheader(f"{algorithm_name} Results") # Generic Subheader for all unsupervised models

                        if algorithm_name == "K-Means":
                            labels = model.labels_ # Get cluster labels
                            #st.subheader("Cluster Labels:")
                            #st.write(labels) # Display cluster labels

                            import plotly.express as px # Import plotly for interactive plots

                            if X.shape[1] >= 2: # Visualize only if at least 2 features
                                st.subheader("Cluster Visualization")
                                # Reduce dimensionality to 3D if data has more than 3 features for visualization
                                if X.shape[1] > 3:
                                    from sklearn.decomposition import PCA
                                    pca_visual = PCA(n_components=3)
                                    X_reduced_visual = pca_visual.fit_transform(X)
                                else:
                                    X_reduced_visual = X.values # Use original features if <= 3

                                fig = px.scatter_3d(
                                    x=X_reduced_visual[:, 0],
                                    y=X_reduced_visual[:, 1],
                                    z=X_reduced_visual[:, 2] if X_reduced_visual.shape[1] > 2 else 0, # Z=0 if 2D
                                    color=labels.astype(str), # Color by cluster labels
                                    text=X.index.astype(str), # Hover text as index
                                    labels={'x': 'Feature 1', 'y': 'Feature 2', 'z': 'Feature 3', 'color': 'Cluster'},
                                    title=f"K-Means Clusters (Data projected to 3D)" if X.shape[1] > 3 else "K-Means Clusters (2D or 3D)"
                                )
                                st.plotly_chart(fig)
                            else:
                                st.warning("Data has less than 2 features, cannot create scatter plot visualization.")


                        elif algorithm_name == "Hierarchical Clustering":
                            labels = model.fit_predict(X) # Get cluster labels - use fit_predict for AgglomerativeClustering
                            #st.subheader("Cluster Labels:")
                            #st.write(labels) # Display cluster labels

                            import matplotlib.pyplot as plt # Import matplotlib for dendrogram
                            from scipy.cluster.hierarchy import dendrogram

                            if X.shape[0] < 50: # Dendrogram is more readable for smaller datasets
                                st.subheader("Dendrogram (Hierarchical Clustering)")
                                fig_dendrogram = plt.figure(figsize=(10, 7))
                                dendrogram(model, labels=X.index.astype(str)) # Use model from AgglomerativeClustering
                                plt.title("Hierarchical Clustering Dendrogram")
                                plt.xlabel("Data Point Index")
                                plt.ylabel("Distance")
                                st.pyplot(fig_dendrogram)
                            else:
                                st.warning("Dendrogram visualization is recommended for datasets with fewer than 50 data points for readability.")


                        elif algorithm_name == "DBSCAN":
                            labels = model.labels_ # Get cluster labels (including -1 for noise)
                           # st.subheader("Cluster Labels (including Noise):")
                     # Display cluster labels
                            #st.write("Noise points are labeled as -1.")

                            import plotly.express as px

                            if X.shape[1] >= 2: # Visualize only if at least 2 features
                                st.subheader("Cluster Visualization ")
                                # Reduce dimensionality to 3D if data has more than 3 features for visualization
                                if X.shape[1] > 3:
                                    from sklearn.decomposition import PCA
                                    pca_visual = PCA(n_components=3)
                                    X_reduced_visual = pca_visual.fit_transform(X)
                                else:
                                    X_reduced_visual = X.values # Use original features if <= 3

                                fig = px.scatter_3d(
                                    x=X_reduced_visual[:, 0],
                                    y=X_reduced_visual[:, 1],
                                    z=X_reduced_visual[:, 2] if X_reduced_visual.shape[1] > 2 else 0, # Z=0 if 2D
                                    color=labels.astype(str), # Color by cluster labels
                                    text=X.index.astype(str), # Hover text as index
                                    labels={'x': 'Feature 1', 'y': 'Feature 2', 'z': 'Feature 3', 'color': 'Cluster'},
                                    title=f"DBSCAN Clusters (Data projected to 3D, Noise in -1)" if X.shape[1] > 3 else "DBSCAN Clusters (2D or 3D, Noise in -1)"
                                )
                                fig.update_traces(marker=dict(size=5)) # Adjust marker size for better visibility
                                st.plotly_chart(fig)
                            else:
                                st.warning("Data has less than 2 features, cannot create scatter plot visualization.")


                        elif algorithm_name == "PCA":
                            explained_variance_ratio = model.explained_variance_ratio_
                            st.subheader("Explained Variance Ratio:")
                            st.write(explained_variance_ratio)
                            # You might want to display cumulative explained variance as well for component selection
                            cumulative_variance = np.cumsum(explained_variance_ratio)
                            st.subheader("Cumulative Explained Variance:")
                            st.line_chart(cumulative_variance) # Simple line chart for cumulative variance
                            st.write("The chart shows how much variance is explained by adding more principal components.")

                            import plotly.express as px
                            if X.shape[1] >= 2:
                                st.subheader("Data in Reduced Dimensional Space (2D Scatter)")
                                X_reduced_2d = model.transform(X)[:, :2] # Project to 2 components for 2D plot
                                pca_df = pd.DataFrame(X_reduced_2d, columns=['PC1', 'PC2'], index=X.index) # Create DataFrame for Plotly
                                fig_pca_scatter = px.scatter(pca_df, x='PC1', y='PC2',
                                                            title="PCA: Data projected onto first 2 Principal Components",
                                                            hover_data=[pca_df.index]) # Hover data as index
                                st.plotly_chart(fig_pca_scatter)
                            else:
                                st.warning("Data has less than 2 features, cannot create 2D scatter plot of PCA results.")


                        elif algorithm_name == "Isolation Forest":
                            anomaly_scores = model.decision_function(X) # Get anomaly scores (lower is more anomalous)
                            outlier_labels = model.predict(X) # Get outlier labels (-1 for outlier, 1 for inlier)

                            st.subheader("Anomaly Scores:")
                            st.line_chart(anomaly_scores) # Display anomaly scores
                            st.write("Lower scores indicate higher anomaly probability.")

                            st.subheader("Outlier Labels:")
                            st.write(outlier_labels) # Display outlier labels
                            st.write("-1 indicates outlier, 1 indicates inlier.")

                            import plotly.express as px
                            if X.shape[1] >= 2:
                                st.subheader("Anomaly Score Visualization (2D Scatter - if applicable)")
                                X_reduced_2d = PCA(n_components=2).fit_transform(X) # Reduce to 2D for visualization
                                anomaly_df = pd.DataFrame(X_reduced_2d, columns=['Feature 1', 'Feature 2'], index=X.index)
                                anomaly_df['Anomaly Score'] = anomaly_scores # Add anomaly scores to DataFrame
                                fig_anomaly_scatter = px.scatter(anomaly_df, x='Feature 1', y='Feature 2', color='Anomaly Score',
                                                                title="Isolation Forest: Anomaly Scores in 2D Feature Space (Projected)",
                                                                hover_data=[anomaly_df.index, 'Anomaly Score'], # Hover data
                                                                color_continuous_scale=px.colors.sequential.Plasma) # Color scale for anomaly scores
                                st.plotly_chart(fig_anomaly_scatter)
                            else:
                                st.warning("Data has less than 2 features, cannot create 2D anomaly score scatter plot.")

                        st.session_state['model_trained'] = True # Set session state flag when model is trained
                        #st.success("Model trained successfully!")

                    except ImportError as ie: # Catch ImportError for library issues
                        st.error(f"Import Error: Could not import necessary library for {algorithm_name}. Please ensure library '{model_library}' is installed.")
                        st.error(f"Details: {ie}")
                        st.exception(ie) 
                    except ValueError as ve: # Catch ValueError for data issues (e.g., empty data)
                        st.error(f"Value Error: Problem with input data for {algorithm_name}. Please check your data and selected features.")
                        st.error(f"Details: {ve}")
                        st.exception(ve) # Show full exception details for debugging
                    except Exception as e: # General exception catch for other errors
                        st.error(f"Error during model training for {algorithm_name}: {e}")
                        st.error("Please check your data, selected features, target variable, and parameters.")
                        st.exception(e) 

                    
            

            # ---------------------- Prediction Section in Sidebar (Selectbox Inputs) ----------------------
            # st.sidebar.header("Make Predictions")
            # predict_button = False
            # prediction_inputs = {}

            # if problem_type != "Unsupervised Learning" and algorithm_name and st.session_state.get('model_trained', False):
            #     st.sidebar.subheader("Input Features for Prediction")
            #     st.sidebar.write("Select values for the input features:")

            #     for feature in selected_features:
            #         # **Use st.session_state.original_data to get unique values**
            #         feature_values = st.session_state.cleaned_data[feature].unique()
            #         # Sort numeric values for better selectbox order (optional but good for numeric features)
            #         if pd.api.types.is_numeric_dtype(st.session_state.cleaned_data[feature].dtype): # Still use cleaned_data dtype check, as scaling doesn't change dtype
            #             feature_values = sorted(feature_values)

            #         # Limit options to a reasonable number for selectbox clarity
            #         num_options_to_display = min(10, len(feature_values)) # Display max 10 options or fewer if less unique values
            #         selectbox_options = list(feature_values[:num_options_to_display]) # Take top options

            #         default_option = selectbox_options[0] if selectbox_options else None # Default to first option if available

            #         param_key = f"selectbox_prediction_{feature}" # Unique key for each prediction selectbox

            #         # Check if session state already has a value for this selectbox
            #         if param_key not in st.session_state:
            #             st.session_state[param_key] = default_option  # Set default in session state if not yet set

            #         prediction_inputs[feature] = st.sidebar.selectbox(
            #             f"Select {feature}",
            #             options=selectbox_options,
            #             # default=default_option,  # REMOVE the default parameter
            #             key=param_key, # Use unique key for session state management
            #             format_func=lambda x: str(x)
            #         )

            #     predict_button = st.sidebar.button("Predict Target Variable")

            # elif problem_type == "Unsupervised Learning":
            #     st.sidebar.write("Prediction not applicable for Unsupervised Learning.")
            # elif not algorithm_name:
            #     st.sidebar.warning("Please select an algorithm first to enable prediction.")
            # elif not st.session_state.get('model_trained', False):
            #     st.sidebar.warning("Please train a model first to make predictions.")


            # if train_button: # Model Training Logic (No changes needed)
            #     st.header("Model Training Results")
            #     # ... (Rest of your train_button logic) ...
            #     st.session_state['model_trained'] = True
            #     st.success("Model trained successfully!")


            # if predict_button: # Prediction Logic
            #     st.header("Prediction Results")
            #     feature_scalers = st.session_state.get('feature_scalers') # Retrieve feature scalers

            #     if problem_type != "Unsupervised Learning" and algorithm_name and st.session_state.get('model_trained', True):
            #         try:
            #             model = st.session_state['trained_model'] # Get 'model' from session state
            #             # Prepare input data for prediction (using values from selectboxes)
            #             input_df = pd.DataFrame([prediction_inputs], columns=selected_features)

            #             # Ensure correct feature order
            #             input_df = input_df[selected_features]

            #             # **Scale input data (if feature_scalers exist) BEFORE encoding**
            #             input_df_scaled = input_df.copy() # Initialize input_df_scaled
            #             if feature_scalers: # Apply scaling using stored scalers
            #                 for feature in selected_features:
            #                     if feature in feature_scalers:
            #                         scaler = feature_scalers[feature]
            #                         input_df_scaled[feature] = scaler.transform(input_df[[feature]]) # Scale in place

            #             # **Encode scaled data**
            #             input_df_encoded = encode_categorical_columns(input_df_scaled.copy()) # Encode scaled data (make a copy)


            #             # Make prediction using the PREPROCESSED input data
            #             predicted_value = model.predict(input_df_encoded) # Predict using encoded and scaled input

            #             st.subheader("Predicted Target Variable:")
            #             if problem_type == "Classification":
            #                 st.write(f"Predicted Class: **{predicted_value[0]}**")
            #             elif problem_type == "Regression":
                            
            #                 target_scaler = st.session_state.get('target_scaler') # Retrieve target scaler
            #                 if target_scaler:
            #                     predicted_value_unscaled = target_scaler.inverse_transform(predicted_value.reshape(-1, 1)) # Inverse transform, reshape if needed
            #                     st.write(f"Predicted Value (Original Scale): **{predicted_value_unscaled[0, 0]:.3f}**") # Display unscaled value
            #                 else: # If no target scaler found (maybe target wasn't scaled)
            #                     st.write(f"Predicted Value (Scaled): **{predicted_value[0]:.3f}**") # Display scaled value with a note

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        st.error("Please check your selected input values and ensure a model is trained.")
                        st.exception(e) # Show full exception details for debugging

                else:
                    st.warning("Prediction is not applicable or model not trained. Please train a supervised learning model first.")
                        
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e) # Show full exception details for debugging
@st.cache_data
def generate_profile_report(df):
    return ProfileReport(df, explorative=True).to_html()

with tab2:
    st.title("Visualize your Data")


    with st.expander("**This tab helps you explore, understand, and analyze your dataset in seconds with automatically generated insights and visualizations**❣️"):
        st.write("1. *Overview & Alerts* – Gives a quick summary of your dataset and highlights any potential issues, such as too many missing values or duplicate rows.🙂")  

        st.write("2. *Dataset Statistics* – Shows key information like the number of rows, columns, duplicate values, and memory usage, helping you understand your dataset’s size and structure.🙃")  

        st.write("3. *Variables* – Lists all columns in your dataset, displaying their types (numeric, categorical, etc.) and key statistics like min, max, mean, unique values, and missing values.😉")  

        st.write("4. *Interaction* – Allows you to explore relationships between different columns using visualizations, helping you spot trends or dependencies.😃")  

        st.write("5. *Correlation* – Measures how strongly two columns are related, showing whether changes in one column affect another (useful for feature selection in machine learning).🙂")  

        st.write("6. *Missing Values* – Identifies gaps in your data, showing which columns have missing values and how much data is missing, so you can decide how to handle them.🙃")  

        st.write("7. *Sample Data* – Displays a few rows from your dataset so you can quickly see what your data looks like before analyzing it further.😃")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", key='bt2', type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            st.write("Data Preview:")
            st.dataframe(data)

            if st.button("Generate Profile Report"):
                st.session_state.profile_report = generate_profile_report(data)
                components.html(st.session_state.profile_report, height=800, scrolling=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
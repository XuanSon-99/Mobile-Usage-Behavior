# Portfolio 4: User Behavior Classification and Prediction on Mobile Devices

## Purpose of the Portfolio

This portfolio focuses on analysing user behavior patterns on mobile devices, using features like app usage time, screen-on time, battery drain, data usage, and demographics (e.g., age, gender). The objective is to classify users into distinct behavioral segments and predict future behavior using classification models.

## Key Questions:

The main objectives of this analysis are outlined through the following questions:

1. **Understand Key Influences** (Correlation Analysis):
   - *Question*: What factors have the most significant influence on user behavior and segmentation?

2. **Segment Users into Behavioral Categories** (Clustering Analysis):
   - *Question*: How can we categorise users into distinct segments based on their behavior patterns?

3. **Predict User Behavior** (Logistic Regression and KNN Model Analysis):
   - *Question*: Can we develop a model to predict future user behavior based on their past activities?

## Why This Portfolio Could Be Helpful

This portfolio provides valuable insights for:
- **Product and App Developers**: Understanding how different groups of users engage with their devices can help optimise app features and user interfaces.
- **Mobile Device Businesses**: Identifying key usage patterns can inform decisions on hardware improvements, battery optimisation, and customer segmentation.
- **Data Science and Machine Learning Learners**: Hands-on experience with feature selection, clustering, and predictive modeling using real-world datasets.

## What This Portfolio Offers

1. **Exploratory Data Analysis (EDA)**:
   - Visualising and understanding relationships between device usage metrics (e.g., app usage time, screen-on time, battery drain) and user behavior.
   - Correlation analysis to identify the most influential factors affecting user behavior segmentation.

2. **Clustering (User Segmentation)**:
   - **K-Means Clustering** is used to group users into distinct behavior categories based on device usage patterns.
   - **Silhouette Score** is applied to assess the quality of the clustering.

3. **Predictive Modeling**:
   - **Logistic Regression**: Used to classify users into behavior segments.
   - **K-Nearest Neighbors (KNN)**: Applied to classify users based on their nearest neighbors in the dataset.
   - **Recursive Feature Elimination (RFE)**: A feature selection technique used to identify the most important features for the classification models.

4. **Model Optimisation**:
   - **Grid Search and Cross-Validation** are used to fine-tune the KNN model's hyperparameters (e.g., finding the optimal number of neighbors, `K`).
   - Different **distance metrics** (e.g., Euclidean, Manhattan) are tested to evaluate their effect on the performance of the KNN model.

5. **Model Evaluation**:
   - Models are evaluated using metrics such as **accuracy** and **F1-score** to ensure a balance between precision and recall.

## How to Go Through Portfolio 4

To effectively explore this portfolio, follow these steps:

1. **Explore the Dataset**:
   - The dataset contains 661 records and 11 columns related to user behavior on mobile devices. The key features include:
     - `App Usage Time` (minutes per day)
     - `Screen On Time` (hours per day)
     - `Battery Drain` (mAh per day)
     - `Data Usage` (MB per day)
     - `Age` and `Gender`
     - `User Behavior Class` (target variable: behavior categories 1-4)

   The dataset can be reviewed in the notebook where the relationships between these features and user behavior are visualised and analysed.

2. **Data Preprocessing**:
   - **Standardisation**: Before applying machine learning algorithms, I standardise numerical features (e.g., `App Usage Time`, `Screen On Time`) using **StandardScaler** to ensure that all features are on a comparable scale.
   - **Handling Missing Data**: If applicable, missing data should be handled during this stage to avoid issues during model training.

3. **Clustering (K-Means)**:
   - **Clustering Analysis**: I apply K-Means clustering to segment users into behavior categories. The number of clusters is determined based on the data, and **silhouette score** is used to evaluate the quality of the clusters.

4. **Train Classification Models**:

### 4.1. **Logistic Regression Classification**
   - Logistic Regression is used as the first classification model to establish a baseline. The model predicts the behavior category (`User Behavior Class`) based on user device usage metrics.
   - **Reference**: [Logistic Regression in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### 4.2. **Recursive Feature Elimination (RFE)**
   - I use **Recursive Feature Elimination (RFE)** to identify the most important features for predicting user behavior. RFE removes less important features recursively, retaining only those that have the most predictive power.
   - **Reference**: [RFE in scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination)

### 4.3. **K-Nearest Neighbors (KNN) Classifier**
   - After selecting the most important features with RFE, I build a **1-NN classifier** (K=1), which classifies a user’s behavior based on their closest neighbor in the dataset.
   - The model is optimised using **Grid Search** and **Cross-Validation** to find the optimal value of `K` and evaluate the best distance metric (Euclidean or Manhattan).
   - **Reference**: [KNN in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)  
   **Reference**: [Grid Search in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

5. **Evaluate Model Performance**:

### 5.1. **Logistic Regression Evaluation**
   - **Accuracy**: Measures how well the model correctly classifies user behavior.
   - **F1-Score**: Since there may be imbalances between behavior classes, I use the F1-score to balance precision and recall.

### 5.2. **KNN Classifier Evaluation**
   - **Grid Search and Cross-Validation**: The hyperparameter `K` is optimised using grid search, and cross-validation ensures robust performance across different subsets of the data.
   - **Distance Metrics**: I compare KNN performance across different distance metrics (e.g., Euclidean, Manhattan) to determine which metric provides the best results.

## How to Download the Dataset
The user behavior dataset is available in this repository under the `data/user_behavior_dataset.csv` folder. You can download the file directly from [data/mobile_user_behavior.csv](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset).

## Getting Help with Technical Issues

For any technical issues or questions regarding this portfolio, please contact:

**Email**: xuanson.nguyen@students.mq.edu.au

## Managed by

This portfolio is managed and maintained by **Xuan Son Nguyen**.
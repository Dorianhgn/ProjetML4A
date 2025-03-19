Here is a structured breakdown of data processing and machine learning methods for the project, divided into two parts: predicting **Calories_Burned** and predicting **Experience_Level**.


# Repartition

### Exploratory analysis and data preprocessing

I think we should meet once and for all to finish this part all together, and then go on our parts.

### Prediction algorithms

- 2 persons works on predictiing **Calories_Burned**
- 2 persons works on predicting **Experience_Level**
- Each person redact the same thing in R and in Python.
- Each person works on his two notebook, by making a branch of the main branch. Then we'll be able to merge them all and combine into a good notebook.
- Finally, for the guys who works on the same prediction, they should reread what the other have done and give reviews about interpretation and choices. For example, I don't know, Sara and I (Dorian) are working on predicting Calories_Burned. She worked on GLM, and I work on Trees. After reviewing his code, I think that doing backward / foward feature selection with ^2 interaction is useful, I should discuss with her the matter.

Finally, when all notebooks will be over, when combining them, each duo will give feed back on the other duo, about the choices and what we could improve.

Since I've done already this course, you can ask me aswell, but in Ljubljana, we focused a lot on making the algorithms by hand and not too much about the interpretability as we do in R.

Between R and Python, I don't really know which one is better to do first, I would rather say R, since it is more hard because we can tweak everything. And the graphs are better for interpretation. Then, it will be easier to translate it with ChatGPT in Python I guess.

### Repartition detailed

#### Calories_burned

Machine learner 1 (Sara) : Linear regression, GLM, SVM

Machine learner 2 (Dorian) : Trees, Forest, XGBoost

Maybe both for neural networks.

#### Experience Level

Machine learner 1 (Lise) : Linear regression, GLM, SVM

Machine learner 2 (Matteo) : Trees, Forest, XGBoost

Maybe both for Neural Networks.

### Deadlines
- Finish the exploratory analys by the 10 march
- Do a review of machine learner 1 / 2 work (end of march)
- By the 11th of April, do a main review of everything except Neural Networks
- Work on Neural Networks during the holidays (14th april - 25 april).
- Do the big review beginning of May (combining notebooks, etc)
- Then work on the powerpoint during the first two weeks, and we'll be done and we can work on Signal exam and other exams.

### Additionnal things to do

**Add for each alogrithm the time that it takes to find the best params, or just to run, to compare each method !!!**

---

Here a brief breakdown I made with ChatGPT about what should be done. **I've added my opinions in *italics*.**

## **1. Exploratory Data Analysis (EDA) & Data Processing**

### **General Preprocessing (for both tasks)**
1. **Check dataset structure & missing values**
   - `df.info()`, `df.describe()`, `df.isnull().sum()`
   - Handle missing values (drop, impute, or replace)

2. **Convert categorical variables into numerical format**
   - `Gender`, `Workout_Type`, and `Experience_Level` should be converted using **one-hot encoding** or **label encoding**.

3. **Check for inconsistencies or outliers**
   - Boxplots for outliers (`plt.boxplot()`)
   - Distribution plots (`sns.histplot()`)

4. **Feature scaling and transformation (if needed)**
   - Normalize continuous variables (`MinMaxScaler` or `StandardScaler`)
   - Consider **log transformation** for skewed distributions (e.g., `Calories_Burned`).

5. **Correlation analysis**
   - Compute **correlation matrix** (`df.corr()`)
   - **Heatmaps** (`sns.heatmap(df.corr(), annot=True)`) to identify relationships.

6. **Visualizations**
   - **Univariate analysis:** Histograms & density plots (`sns.histplot()`)
   - **Bivariate analysis:**
     - Scatterplots (`sns.scatterplot()`)
     - Boxplots to compare categories (`sns.boxplot(x="Workout_Type", y="Calories_Burned")`)

7. **Principal Component Analysis (PCA)**
   - Reduce dimensionality of continuous features for visualization.
   - `sklearn.decomposition.PCA` in Python / `prcomp()` in R.

### **Ideas for calories burned**

1. **Data Cleaning & Encoding**
   - Check for missing values and outliers.
   - Convert categorical variables (Gender, Workout_Type, Workout_Frequency) into dummy variables.

2. **Feature Engineering & Scaling**
   - Normalize or standardize numerical variables (especially BPM values, BMI, etc.).
   - Create interaction terms if necessary (e.g., BMI × Workout Frequency). *Not necessary, since it is done in GLM ^2*
   - Log transformation for right-skewed variables (e.g., Calories_Burned).

3. **Exploratory Data Analysis (EDA)**
   - **Univariate Analysis**: Histograms & Boxplots for Calories_Burned.
   - **Bivariate Analysis**: Scatterplots (Calories_Burned vs. numerical features), boxplots (Calories_Burned vs. categorical variables).
   - **Correlation Heatmap** to see relationships between variables.
   - **PCA**: Check for redundancy in numerical features.

4. **Plots & Interpretability**
   - Histogram of **Calories_Burned**.
   - Scatterplot of **Calories_Burned vs. Session Duration**.
   - Boxplots of **Calories_Burned across Workout_Type**.
   - PCA biplot for feature relationships.


### **Ideas for Experience_Level**
1. **Data Cleaning & Encoding**
   - Ensure Experience_Level is treated as an ordered categorical variable.

2. **Feature Engineering & Scaling**
   - Standardize numerical variables. *already done*
   - Create interaction terms (e.g., Max_BPM × Workout Frequency). *doing it in GLM*
   - Possibly bin some variables (e.g., Age groups). *Could be really useful*

3. **Exploratory Data Analysis (EDA)**
   - **Univariate Analysis**: Distribution of **Experience_Level**.
   - **Bivariate Analysis**: Boxplots, histograms, correlation analysis.
   - **PCA** to check for feature redundancy.

4. **Plots & Interpretability**
   - Count plot of **Experience_Level**.
   - Boxplot of **Age vs. Experience_Level**.
   - Parallel coordinate plot for numerical features.

#### ***Ideas from DoDo***
- Maybe do as the TP on Ozone, break down Calories burned into categories and do categorical regression, then compare and will maybe better for interpretability purposes.


## **2. Predicting Calories_Burned (Regression Problem)**

### **Machine Learning Models for Calories_Burned**

#### **1. Linear Regression**
   - Fit a baseline linear model.
   - Assess residuals to check for heteroscedasticity.
   - Check for multicollinearity using VIF.
   - **Plot**: Residual vs. Fitted Plot, Q-Q plot.

#### **2. Ridge Regression (L2) & LASSO Regression (L1)**
   - Ridge: Prevents overfitting by penalizing large coefficients.
   - LASSO: Selects important features via regularization.
   - Hyperparameter tuning via cross-validation.
   - **Plot**: Coefficients vs. λ (regularization parameter).

#### **3. Generalized Linear Model (GLM)**
   - Use Gaussian distribution with an appropriate link function.
   - Compare different link functions (e.g., identity vs. log).
   - **Plot**: Deviance residuals to check model assumptions.

#### **4. Regression Trees**
   - Fit a decision tree regressor.
   - Prune the tree using cross-validation.
   - **Plot**: Decision tree structure.

#### **5. Support Vector Regression (SVR)**
   - Use RBF kernel.
   - Tune **C** and **ε** via cross-validation.
   - **Plot**: SVR predicted vs. actual values.

#### **6. Random Forest Regressor**
   - Train using bootstrapped samples.
   - Tune the number of trees and max depth.
   - **Plot**: Feature Importance.

#### **7. Boosting (XGBoost, LightGBM)**
   - Optimize number of trees and learning rate.
   - Use early stopping to prevent overfitting.
   - **Plot**: SHAP values for interpretability.

#### **8. Neural Networks (MLP)**
   - Use a simple feedforward network.
   - Try ReLU activations and dropout layers.
   - Tune hyperparameters (hidden layers, neurons).
   - **Plot**: Loss function evolution across epochs.

#### **Model Comparison**
   - Evaluate using **RMSE, MAE, and R²**.
   - Plot actual vs. predicted Calories_Burned.


## **3. Predicting Experience_Level (Classification Problem)**

### **Machine Learning Models for Experience_Level**

#### **1. Logistic Regression**
   - Train a multinomial logistic regression model.
   - Use **one-vs-rest** classification.
   - **Plot**: Confusion matrix.

#### **2. Decision Trees (Classification)**
   - Train a classification tree.
   - Tune tree depth using cross-validation.
   - **Plot**: Decision tree structure.

#### **3. Support Vector Machines (SVM)**
   - Try **linear** and **RBF** kernels.
   - Tune **C** and **γ** using grid search.
   - **Plot**: Decision boundary (if 2D possible).

#### **4. Random Forest Classifier**
   - Train using bootstrapped samples.
   - Tune the number of trees and max depth.
   - **Plot**: Feature importance.

#### **5. Boosting (XGBoost, LightGBM)**
   - Optimize number of estimators, learning rate.
   - Use **early stopping** to prevent overfitting.
   - **Plot**: SHAP values for interpretability.

#### **6. Neural Networks (MLP)**
   - Use a feedforward neural network.
   - Optimize layers, neurons, and dropout.
   - **Plot**: Loss curve.

#### **Model Comparison**
   - Evaluate using **Accuracy, F1-score, Precision, Recall**.
   - Plot **ROC curves** and **Confusion Matrices**.

$\newline$

# **Final Deliverables**
- **Python & R Notebooks** with commented code.
- **Slides for the oral defense** covering:
  - **Introduction**: Dataset and objectives.
  - **Data Processing**: Cleaning, encoding, feature selection.
  - **EDA**: Key visualizations.
  - **Model Training & Comparison**: Performance metrics and interpretations.
  - **Conclusion**: Key findings and future work.


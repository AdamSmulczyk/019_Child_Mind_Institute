### Full and extensive analysis can be found in the file eda.py.

* I. PLANNING
    * I.1 Introduction
    * I.2 Dataset description
    * I.3 Project assumptions
        * I.3.1 Defining the problem
        * I.3.2 Assessing the scope
        * I.3.3 Success metric
        * I.3.4 Feasibility of the ML application
* II.DATA COLLECTION AND PREPARATION
    * II.1 Import libraries and data files
    * II.2 Exploratory data analysis (EDA)
        * II.2.1 Reading data & target C=class distribution
        * II.2.2 Statistical summary
        * II.2.3 Correlation Matrics
        * II.2.4 Missing values, categorical data transformation
        * II.2.5 Distribution of attributes with fewer than 10 unique values
        * II.2.6 Distribution of numerical features
        * II.2.7 Distribution of categorical features
        * II.2.8 Distribution of target class in season
        * II.2.9 Correlations between Numerical Features
* III DATA PRE-PROCESSING
    * III.1 Target encoding
    * III.2 Filling nulls
    * III.3 Removing duplicates and unnecessary columns
    * III.4 Filling nulls
    * III.5 Filling nulls
    * III.6 Convert types (downcasting)
* IV DATA PROCESSING
    * IV.1 Skewness of distributions
    * IV.2 Detect outlier
    * IV.3 Categorical data transformation
    * IV.4 Normalizing
    * IV.5 TSN
    * IV.6 PCA
    * IV.7 Feature selection
    * IV.8 Imbalanced target - oversampling by SMOTEE
   
   
I.1 Introduction
The goal is to predict whether a mushroom is edible or poisonous based on its physical features, such as color, shape, and texture.

To tackle this problem, we'll be analyzing a special dataset. This dataset was created by a deep learning model that studied thousands of mushrooms. While the data is similar to a well-known mushroom dataset, there are some differences that make this project unique.

This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like leaflets three, let it be'' for Poisonous Oak and Ivy.

I.2 Dataset description
Full feature description is in eda.py file

I.3 Project assumptions
I.3.1 Defining the problem
This project will be implemented based on a real dataset, provided by the project organizer. The goal is to develop a model that can classify mushrooms as edible ('e') or poisonous ('p') using a set of physical attributes provided in the dataset.

I.3.2 Assessing the scope
The entire project was done in Python, using Jupyter. Defining the scope of the project, its purpose and priorities determines the type of approach to it. In this case, the main goal is to predict whether a mushroom is edible or poisonous based on its physical features, such as color, shape, and texture.

II.DATA COLLECTION AND PREPARATION
Conclusions:
* There is no duplicates in both files.
* There is 3116945 records but distribution is almost equal, e-45%, p-55%.
* With three million rows, the training dataset is huge. And with this large amount of data, we'll focus on gradient-boosted tree models and neural networks.
* Most features have missing values. We'll either use models which can deal with missing values natively, or we'll have to impute the missing values.
* Most features are categorical. We can experiment with one-hot encoding and target encoding.
* For numerical data skew is normal.
* Categorical data can't be dummies because nunique value for them is high.
* Positive Correlations:
    * 'stem-root' shows strong positive correlation with target class.
    * 'stem-width' shows strong positive correlation with 'cap-diameter'.
    * 'veil-color' shows strong positive correlation with 'stem-surface'.
    * 'cap-diameter', 'stem-surface','stem-width', 'stem-root' and 'stem-height' shows a moderate positive correlations with features.
* Negative Correlations:
    * 'veil-color' and 'spore-print-color' shows strongly negative correlation with target class.
    * Vehicle_Age and Policy_Sales_Channel status have a moderate negative correlation with Response.
    * Age status show strongly negative correlation with Policy_Sales_Channel, Vehicle_Age and Previously_Insured.
* We encounter a challenge: some categories don't show up very often in our data. This makes it hard to work with them. To fix this, we'll group these rare categories together into a new category called "Unknown".
* Distribution target class is almost equal.
* Cap-shape: although most of b and o type is poisoning.
* Distribution is very unbalanced: 'does-bruise-or-bleed', 'ring-type', 'habitat'.
* Gil-spacing: although most of s type is poisoning. For d type most is eating.
* The distribution of our numerical columns is right-skewed with outliers, meaning that most values are concentrated on the left side of the distribution, but there are some unusually high values (outliers) that are far away from the rest. This suggests that our data may not be normally distributed, which could impact our analysis and modeling results.
* There are most mushrooms in spring and autumn.
* Only in winter more mushrooms is eating than poisoning.

III DATA PRE-PROCESSING
Conclusions:
* Standardizing the Missing Data with null values to make it easier to handle.
* Removing duplicates and unnecessary columns.
* Aggregating categorical and numerical columns
* To maximize the accuracy of our predictions, we will replace all missing values in categorical columns with 'Unknown', allowing us to retain as many columns as possible for analysis.
* Convert types (downcasting).

IV DATA PROCESSING
Conclusions:
* Principal Component Analysis or PCA is a linear dimensionality reduction algorithm. In this technique, data is linearly transformed onto a new coordinate system such that the directions (principal components) capturing the largest variation in the data can be easily identified.
* We can preserve 99% of variance with 50 components if we use PCA.
* There is a lot of feature variables so instead of engineering new features, we might want to focus on eliminating uninformative features and focusing on only the crucial ones. PCAs might also be useful in this scenario where it allows us to pick out components which convey useful information about the data.
* Some of the features show close to 0 correlation with the target variable which could signal that it is useless to us.
* Mutual Information Score helps us to understand how much the feature variables tell us about the target variable.
Since our data have a lot of feature variables, we can take help of this to remove redundant feature variables. This may improve the proformance of our model.


### Author Details:
- Name: Adam Smulczyk
- Email: adam.smulczyk@gmail.com
- Profile: [Github](https://github.com/AdamSmulczyk)
- [Github Repository](https://github.com/AdamSmulczyk/018_Vector_Borne_Disease)
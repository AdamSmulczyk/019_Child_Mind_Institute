### Full and extensive analysis can be found in the file eda_problematic_internet_use.py.

- [I. PLANNING](#I)
    - [I.1 Introduction](#I.1)
    - [I.2 Dataset description](#I.2)
    - [I.3 Project assumptions](#I.3)
        - [I.3.1 Defining the problem](#I.3.1)
        - [I.3.2 Assessing the scope](#I.3.2)
        - [I.3.3 Success metric](#I.3.3)
        - [I.3.4 Feasibility  of the ML application](#I.3.4)
- [II.DATA COLLECTION AND PREPARATION](#II)
    - [II.1 Import libraries and data files](#II.1)
    - [II.2 Exploratory data analysis (EDA)](#II.2)
        - [II.2.1 Reading data & target C=class distribution](#II.2.1)
        - [II.2.2 Statistical summary](#II.2.2)
        - [II.2.3 Correlation Matrics](#II.2.3)
        - [II.2.4 Missing values](#II.2.4)
        - [II.2.5 Distribution of attributes with fewer than 10 unique values](#II.2.5)
            - [II.2.5.1 Distribution of Basic_Demos-Sex](#II.2.5.1)
            - [II.2.5.2 Distribution of FGC-FGC_CU_Zone](#II.2.5.2)
            - [II.2.5.3 Distribution of FGC-FGC_PU_Zone](#II.2.5.3)
            - [II.2.5.4 Distribution of Basic_Demos-Enroll_Season](#II.2.5.4)            
            - [II.2.5.5 Distribution of CGAS-Season](#II.2.5.5)            
            - [II.2.5.6 Distribution of Physical-SeasonE](#II.2.5.6)           
            - [II.2.5.7 Distribution of BIA-BIA_Activity_Level_num](#II.2.5.7)
            - [II.2.5.8 Distribution of Fitness_Endurance-Season](#II.2.5.8)
            - [II.2.5.9 Distribution of PCIAT-PCIAT_01](#II.2.5.9)            
            - [II.2.5.10 Distribution of PreInt_EduHx-computerinternet_hoursday](#II.2.5.10)          
       - [II.2.6 Distribution of PCIAT-PCIAT_Total](#II.2.6) 
       - [II.2.7 Distribution of Physical-BMI](#II.2.7)
       - [II.2.8 Distribution of Physical-Height](#II.2.8)       
       - [II.2.9 Distribution of Physical-Weight](#II.2.9)     
       - [II.2.10 Distribution of Physical-Diastolic_BP](#II.2.10)       
       - [II.2.11 Distribution of PreInt_EduHx-computerinternet_hoursday](#II.2.11)       
       - [II.2.12 Didtibution of categorica](#II.2.12)   
- [III DATA PRE-PROCESSING (data cleaning)](#III)     
    - [IV.1 Filling nulls](#IV.1)
    - [II.2 Feature Engineering](#IV.2) 
    - [III.3 Convert types (Downcasting)](#III.3)
    - [III.4 Skewness of distributions](#III.4)
    - [III.5 Detect outlier](#III.5)
    - [III.6 Categorical data transformation](#III.6)   
    - [III.7 Normalizing](#III.7)     
- [IV DATA PROCESSING](#IV)
- [V MODEL ENGINEERING](#V)
    - [V.1 XGBRegressor](#V.1)
        - [V.1.1 XGBRegressor - Evaluation](#V.1.1)
        - [V.1.2 XGBRegressor Tuning - RandomizedSearchCV](#V.1.2)
    - [V.2 LGBMClassifier Tuning - Optuna](#V.2) 
    - [V.3 RandomForestClassifier](#V.3)
    - [V.4 VotingRegressor](#V.4)
    - [V.4.1 VotingRegressor - Validation](#V.4.1)
    - [V.4.2 VotingRegressor - Evaluation](#V.4.2)
- [VI CONCLUSION](#VI)  
   
   
I.1 Introduction
The goal is to develop a predictive model capable of analyzing children's physical activity data to detect early indicators of problematic internet and technology use. This will enable prompt interventions aimed at promoting healthier digital habits.

Your work will contribute to a healthier, happier future where children are better equipped to navigate the digital landscape responsibly.

I.2 Dataset description
Full feature description is in eda.py file

I.3 Project assumptions
I.3.1 Defining the problem
Can you predict the level of problematic internet usage exhibited by children and adolescents, based on their physical activity? The goal is to develop a predictive model that analyzes children's physical activity and fitness data to identify early signs of problematic internet use. Identifying these patterns can help trigger interventions to encourage healthier digital habits.

I.3.2 Assessing the scope
The entire project was done in Python, using Jupyter. Defining the scope of the project, its purpose and priorities determines the type of approach to it. In this case, the main goal is to achieve a predictive model result that exceeds the satisfaction value achieved by the organizer. This provlem challenges you to develop a predictive model capable of analyzing children's physical activity data to detect early indicators of problematic internet and technology use. This will enable prompt interventions aimed at promoting healthier digital habits.

The main problem of this project was to predict the level of problematic internet usage exhibited by children and adolescents, based on their physical activity. The goal is to develop a predictive model that analyzes children's physical activity and fitness data to identify early signs of problematic internet use. Identifying these patterns can help trigger interventions to encourage healthier digital habits. Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two outcomes.
After thorough data exploration, we trained several regression machine learning models including Logistic Regression, XGBRegressor and LGBMRegressor. The models achieved the following r2/quadratic_weighted_kappa. Scores for ewach of these three models was tremendous good around 99%.

I strongly suspected that there was one or more features in X data that is nearly perfectly correlated with the Y data. Usually this is bad, because those variables don't explain Y but are either explained by Y or jointly determined with Y.

After thorough data exploration that PCIAT atributes was perfectly correlated with the target, after dropping them our scores r2/quadratic_weighted_kappa dropped to around 89%.


### Author Details:
- Name: Adam Smulczyk
- Email: adam.smulczyk@gmail.com
- Profile: [Github](https://github.com/AdamSmulczyk)
- [Github Repository](https://github.com/AdamSmulczyk/018_Vector_Borne_Disease)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from models_cmi import *
from data_processor_cmi import preprocess_data


# In[2]:


def main():
    print('-' * 80)
    print('train')
    
    print("Preprocessing data...")
    train_file = 'train.csv'
    train = preprocess_data(train_file)
   
    X = train.drop(columns=['id', 'sii'])
    y = train['sii']
 
    SEED = 42
    XGB_Params = {'max_depth': 13, 
                  'min_child_weight': 5, 
                  'learning_rate': 0.02,
                  'colsample_bytree': 0.6, 
                  'max_bin': 3000, 
                  'n_estimators': 1500,
                  'random_state': SEED}

    print("Initializing workflow...")
#     RFC_1 = RandomForestModel(n_estimators=50, random_state=SEED)
#     XGB_2 = XGBoostModel(**XGB_Params)

#     voting_estimators = [('RandomForest', RFC_1.model), ('XGBoost', XGB_2.model)]
#     workflow_voting = Workflow_6()
#     workflow_voting.run_workflow(
#         model_name='VotingModel',
#         model_kwargs={'estimators': voting_estimators, 'voting': 'soft', 'weights': [1.0, 2.0]},
#         X=X,
#         y=y,
#         test_size=0.2,
#         random_state=42,
#         scoring='accuracy'
#     )
    
    XGB_1 = Workflow_8()
    XGB_1.run_workflow(
        model_name='XGBoostModel',
#         model_kwargs=XGB_Params,
       model_kwargs= {'random_state': SEED},    
        X=X,
        y=y,
        test_size=0.2,
        random_state=42,
        scoring='r2'
    )
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





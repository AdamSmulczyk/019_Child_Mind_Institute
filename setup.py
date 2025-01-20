#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, find_packages

setup(
    name="cmi_regressor",
    version="1.1.0", 
    author="Adam Smulczyk",
    author_email="adam.smulczyk@gmail.com",
    description="A tool for training and evaluating machine learning models on CMI.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AdamSmulczyk/019_Child_Mind_Institute",
    packages=find_packages(),  
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "scikit-learn>=0.24.0",
        "xgboost>=1.3.0",
        "lightgbm>=3.2.0",
        "catboost>=0.26",
    ],  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  
    entry_points={
        "console_scripts": [
            "CMIa=main_cmi:main",
        ]
    },
)


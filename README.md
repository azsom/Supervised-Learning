# Supervised-Learning, an AI+Training course series

Supervised Learning is a course series that walks through all steps of the classical supervised machine learning pipeline. We use python and packages like scikit-learn, pandas, numpy, and matplotlib. The course series focuses on topics like cross validation and splitting strategies, evaluation metrics, supervised machine learning algorithms (like linear and logistic regression, support vector machines, and tree-based methods like random forest, gradient boosting, and XGBoost), and interpretability. You can complete the courses in sequence or complete individual courses based on your interest. 

## Prerequistes

We use jupyter notebooks and python in this course series. The env.yml file contains the python and package versions used to develop the codes in this repository. You can directly use that file to create an environment with conda. On linux and mac, you need to give the following commands in the terminal:

*conda env create -n [name_of_env] -f [path_to_yaml]*

*conda activate [name_of_env]*

Unfortunately I have very little experience with windows so I'm not sure how conda works with that operating system.

Once your environment is created, please run the test_environment.ipynb. It checks the versions of your python and packages (like pandas, sklearn, xgboost). If the notebook returns all OK, you should be able to run and reproduce all notebooks in this repo without issues. If some FAILs are returned, you should install/update those packages first. 

## Description

### Part 1: Introduction to machine learning and the bias-variance tradeoff
This course starts with a high-level overview of supervised machine learning focusing on regression and classification problems, what questions can be answered with these tools, and what the ultimate goal of a machine learning pipeline is. Then we will walk through the math behind linear and logistic regression models with regularization. Finally, we put together a simple pipeline using a toy dataset to illustrate the bias-variance tradeoff, a key concept in machine learning that drives how models are selected.

## Authors

Andras Zsom (andras_zsom@brown.edu).

## License

This project is licensed under the MIT License.

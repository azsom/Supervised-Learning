# Supervised-Learning, an AI+Training course series

Supervised Learning is a course series that walks through all steps of the classical supervised machine learning pipeline. We use python and packages like scikit-learn, pandas, numpy, and matplotlib. The course series focuses on topics like cross validation and splitting strategies, evaluation metrics, supervised machine learning algorithms (like linear and logistic regression, support vector machines, and tree-based methods like random forest, gradient boosting, and XGBoost), and interpretability. You can complete the courses in sequence or complete individual courses based on your interest. 

If you are interested in taking these courses, please sign up on the [AI+training site](https://aiplus.odsc.com/courses/supervised-machine-learning-series).

## Prerequistes

We use jupyter notebooks and python in this course series. The env.yml file contains the python and package versions used to develop the codes in this repository. You can directly use that file to create an environment with conda. On linux and mac, you need to give the following commands in the terminal:

*conda env create -n [name_of_env] -f [path_to_yaml]*

*conda activate [name_of_env]*

Unfortunately I have very little experience with windows so I'm not sure how conda works with that operating system.

Once your environment is created, please run the test_environment.ipynb. It checks the versions of your python and packages (like pandas, sklearn, xgboost). If the notebook returns all OK, you should be able to run and reproduce all notebooks in this repo without issues. If some FAILs are returned, you should install/update those packages first. 

## Description

### Part 1: Introduction to machine learning and the bias-variance tradeoff
This course starts with a high-level overview of supervised machine learning focusing on regression and classification problems, what questions can be answered with these tools, and what the ultimate goal of a machine learning pipeline is. Then we will walk through the math behind linear and logistic regression models with regularization. Finally, we put together a simple pipeline using a toy dataset to illustrate the bias-variance tradeoff, a key concept in machine learning that drives how models are selected.

### Part 2: How to prepare your data for supervised machine learning
Part 2 of the course series is on how to prepare your data for training and evaluating a machine learning model. Two steps are covered: how to split and preprocess your data. My experience is that beginner practitioners often make a mistake referred to as data leakage when splitting their dataset. Data leakage means that you use information in the model training process which will not be available at prediction time. The unfortunate side effect is that the model seems to perform well in production but poorly in deployment. Two modules are dedicated to splitting with the hope that the participants will be well-equipped to avoid data leakage upon completing the modules. The third module is on preprocessing. There are two driving concepts behind preprocessing: the feature matrix needs to be numerical (no strings or any other data types are allowed when using sklearn), and some machine learning models converge faster and perform better if all features are standardized. 

### Part 3: Evaluation metrics in supervised machine learning
Part 3 of the course series focuses on the target variable and evaluation metrics that measure how well the machine learning model makes predictions. Classification problems have a larger variety of metrics and it might require quite some deliberation to choose the right metric for your problem. We will discuss what questions you should ask yourself while making these decisions and what are some recommended choices in certain scenarios. It is easier to choose an evaluation metric in regression. The choices you need to make are mostly subjective and usually do not have a strong impact on the overall predictive power of your model. We will also discuss how to calculate the baseline value of each evaluation metric. A baseline is a result of a very simple solution that uses only the known target variable for prediction. Baselines are useful anchors to compare the performance of your ML model against and we want our model to outperform the baseline.

### Part 4: Non-linear Supervised Machine Learning Algorithms
We review four non-linear supervised machine learning algorithms in part 4 of the course series (K-Nearest Neighbors, Support Vector Machines, Random Forests, XGBoost). When you work on a project, generally you should try as many algorithms as you can on your dataset because it is difficult to know apriori which algorithm will perform best. Thus it is important to understand how various algorithms work, what hyperparameters need to be tuned, what the pros and cons of each algorithm are, etc. While we will not cover the in-depth math behind these algorithms as we did with linear and logistic regression in part 1, you will have a solid intuitive understanding of how the algorithms work upon completing this course. We will use a couple of toy datasets and visualizations I found helpful when learning about the properties of a new algorithm. As a result, you will be well-equipped to master other algorithms we do not cover here by yourself. I will also describe a couple of insights I gained about these algorithms over the years that might not be obvious to new users. Deep learning libraries (i.e., tensorflow, keras, pytorch) are beyond this course series.

### Part 5: Missing data in supervised ML
Datasets are almost never complete and this can introduce various biases to your analysis. Due to these biases, your supervised machine learning model can produce incorrect predictions. The goal of this course is to give you an idea of why some of the most common approaches for dealing with missing values often introduce some type of bias. I will also describe the methods and techniques that can help you to arrive at an unbiased conclusion in the face of missing data.

### Part 6: Interpretability
A crucial part of the ML pipeline is to explain the predictions which is the focus of part 6 in the course series. It can be difficult to understand exactly how supervised machine learning models (especially non-linear ones) work which is why ML models are sometimes called black boxes. Such black boxes are fine if the only thing we care about is predictive accuracy. However often predictions are not enough and the model needs to provide explanations along with the predictions. E.g., if the model predicts that a patient has a certain disease, the doctor needs to be able to explain to the patient how the model made that diagnosis. We will start with global feature importance metrics which measure how strongly each feature contributes to the predictions generally. Then we will move on to local feature importance metrics which describe how much each feature contributes to the prediction of one data point specifically. I demonstrate all metrics on the same dataset to highlight that different metrics rank the features differently so model interpretability depends on the metric(s) you use. 

## Authors

Andras Zsom (andras_zsom@brown.edu).

## License

This project is licensed under the MIT License.

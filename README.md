# Executive Summary

This project involves the comprehensive evaluation and optimization of TensorFlow Decision Forests for a classification task. The primary objective was to explore various hyperparameters to improve model performance and determine the most effective settings for the dataset. This report details the data exploration and preprocessing steps, the selection and training of the model, the hyperparameters evaluated, and the final results. The project demonstrates the advantages of using Gradient Boosted Trees and provides insights into the model's strengths and weaknesses compared to other machine learning models.

## Table of Contents
1. [Introduction](#1-introduction)
    - [1.1 Overview of the Project](#11-overview-of-the-project)
    - [1.2 Objectives and Goals](#12-objectives-and-goals)
    - [1.3 Importance of the Project](#13-importance-of-the-project)
2. [Data Exploration and Preprocessing](#2-data-exploration-and-preprocessing)
    - [2.1 Description of the Dataset](#21-description-of-the-dataset)
    - [2.2 Data Cleaning and Preprocessing Steps](#22-data-cleaning-and-preprocessing-steps)
    - [2.3 Exploratory Data Analysis (EDA)](#23-exploratory-data-analysis-eda)
3. [Model Selection and Training](#3-model-selection-and-training)
    - [3.1 Overview of TensorFlow Decision Forests](#31-overview-of-tensorflow-decision-forests)
    - [3.2 Baseline Model Training](#32-baseline-model-training)
    - [3.3 Improved Model Training](#33-improved-model-training)
    - [3.4 Hyperparameter Tuning](#34-hyperparameter-tuning)
    - [3.5 Model Ensembling](#35-model-ensembling)
4. [Hyperparameters Evaluated](#4-hyperparameters-evaluated)
    - [4.1 List of Hyperparameters Considered](#41-list-of-hyperparameters-considered)
    - [4.2 Impact of Each Hyperparameter on Model Performance](#42-impact-of-each-hyperparameter-on-model-performance)
    - [4.3 Range of Values Explored for Each Hyperparameter](#43-range-of-values-explored-for-each-hyperparameter)
    - [4.4 Final Hyperparameter Settings Used](#44-final-hyperparameter-settings-used)
5. [Evaluation and Results](#5-evaluation-and-results)
    - [5.1 Model Performance Metrics](#51-model-performance-metrics)
    - [5.2 Comparison of Different Models and Parameters](#52-comparison-of-different-models-and-parameters)
    - [5.3 Final Model Selection and Justification](#53-final-model-selection-and-justification)
6. [Pros and Cons of Model Selection](#6-pros-and-cons-of-model-selection)
    - [6.1 Advantages of Gradient Boosted Trees](#61-advantages-of-gradient-boosted-trees)
    - [6.2 Comparison with Other Models](#62-comparison-with-other-models)
    - [6.3 Reasons for Selecting Gradient Boosted Trees](#63-reasons-for-selecting-gradient-boosted-trees)
7. [Conclusion](#7-conclusion)
    - [7.1 Summary of Findings](#71-summary-of-findings)
    - [7.2 Lessons Learned](#72-lessons-learned)
    - [7.3 Future Work](#73-future-work)
8. [Appendices](#8-appendices)
    - [8.1 Additional Charts, Graphs, or Code Snippets](#81-additional-charts-graphs-or-code-snippets)
    - [8.2 References or Further Reading](#82-references-or-further-reading)

## 1. Introduction

### 1.1 Overview of the Project

This project aims to evaluate and optimize TensorFlow Decision Forests for a classification task. By exploring various hyperparameters and their impact on model performance, the goal is to determine the most effective settings that enhance the accuracy and efficiency of the model. This evaluation involves thorough data exploration, preprocessing, and detailed model training processes.

### 1.2 Objectives and Goals

The primary objectives of this project are:
- To understand the data and perform necessary preprocessing.
- To train a baseline model using TensorFlow Decision Forests.
- To improve the model through hyperparameter tuning and model ensembling.
- To evaluate the impact of different hyperparameters on model performance.
- To select the final model based on comprehensive performance metrics.

### 1.3 Importance of the Project

This project is significant as it showcases the power of Gradient Boosted Trees within TensorFlow Decision Forests for solving classification tasks. By systematically evaluating and tuning hyperparameters, the project highlights the importance of model optimization in achieving superior performance. The insights gained from this project are valuable for data scientists and machine learning practitioners aiming to leverage advanced techniques for enhanced model accuracy and reliability.

## 2. Data Exploration and Preprocessing

### 2.1 Description of the Dataset

The dataset used in this project is a comprehensive collection of data points relevant to the classification task at hand. It includes various features that contribute to the model's ability to make accurate predictions. The dataset is divided into training and testing sets to facilitate proper model evaluation and validation. The key characteristics of the dataset include the number of features, the target variable, and the distribution of classes.

### 2.2 Data Cleaning and Preprocessing Steps

Data cleaning and preprocessing are critical steps in preparing the dataset for model training. The steps undertaken in this project include:
- **Handling Missing Values**: Identifying and imputing or removing missing values to ensure the dataset is complete.
- **Feature Engineering**: Creating new features or transforming existing features to better represent the underlying data patterns.
- **Normalization/Standardization**: Scaling numerical features to a common range to improve model performance.
- **Encoding Categorical Variables**: Converting categorical variables into numerical formats using techniques such as one-hot encoding or label encoding.
- **Splitting Data**: Dividing the dataset into training and testing sets to enable proper model validation.

### 2.3 Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is conducted to gain insights into the dataset and identify any underlying patterns or anomalies. Key EDA steps include:
- **Descriptive Statistics**: Summarizing the central tendency, dispersion, and shape of the dataset's distribution.
- **Data Visualization**: Creating plots such as histograms, scatter plots, and box plots to visualize the distribution and relationships between features.
- **Correlation Analysis**: Examining the correlation between features to identify potential multicollinearity issues and understand feature interactions.
- **Class Distribution**: Analyzing the distribution of the target variable to address any class imbalance issues.

## 3. Model Selection and Training

### 3.1 Overview of TensorFlow Decision Forests

TensorFlow Decision Forests is a powerful library for training, evaluating, and serving decision forest models. It provides implementations of various decision tree algorithms, including Random Forests and Gradient Boosted Trees, which are highly effective for both classification and regression tasks. This section provides an overview of the TensorFlow Decision Forests library, its features, and its advantages.

### 3.2 Baseline Model Training

The baseline model serves as the initial benchmark for evaluating the performance of the decision forest model. The steps involved in training the baseline model include:
- **Model Initialization**: Setting up the initial model using default hyperparameters.
- **Training**: Fitting the model to the training data.
- **Evaluation**: Assessing the model's performance on the testing set using various metrics such as accuracy, precision, recall, and F1 score.

### 3.3 Improved Model Training

Building on the baseline model, improved model training involves optimizing the model to enhance its performance. This includes:
- **Feature Selection**: Identifying and selecting the most relevant features to reduce overfitting and improve model accuracy.
- **Hyperparameter Tuning**: Systematically exploring different hyperparameter values to find the optimal settings for the model.
- **Cross-Validation**: Implementing cross-validation techniques to ensure the model's robustness and generalizability.

### 3.4 Hyperparameter Tuning

Hyperparameter tuning is a crucial step in improving model performance. This involves:
- **Grid Search**: Exhaustively searching through a specified subset of hyperparameters to identify the best combination.
- **Random Search**: Randomly sampling hyperparameters from a defined space and selecting the best performing set.
- **Bayesian Optimization**: Using probabilistic models to select hyperparameters that maximize the model's performance.

### 3.5 Model Ensembling

Model ensembling involves combining multiple models to improve overall performance. Techniques used in this project include:
- **Voting Ensemble**: Combining predictions from multiple models and using a majority vote or averaging to make the final prediction.
- **Stacking**: Training a meta-model on the predictions of several base models to achieve better performance.
- **Bagging and Boosting**: Implementing methods like Bagging (Bootstrap Aggregating) and Boosting to reduce variance and bias in the model.

## 4. Hyperparameters Evaluated

### 4.1 List of Hyperparameters Considered

In this project, several hyperparameters were considered to optimize the performance of the TensorFlow Decision Forests model. These hyperparameters include:
- `min_examples`
- `categorical_algorithm`
- `growing_strategy`
- `max_depth`
- `max_num_nodes`
- `shrinkage`
- `num_candidate_attributes_ratio`
- `split_axis`
- `sparse_oblique_normalization`
- `sparse_oblique_weights`
- `sparse_oblique_num_projections_exponent`

### 4.2 Impact of Each Hyperparameter on Model Performance

The impact of each hyperparameter on model performance was assessed by evaluating the accuracy and loss metrics for different configurations. Here’s a detailed explanation using the Titanic dataset as an example:

- **min_examples**: This parameter specifies the minimum number of samples required to make a split in a decision tree. For example, if `min_examples` is set to 10, a decision node will only be created if there are at least 10 samples in the subset. For the Titanic dataset, this helps ensure that splits are made only when there is sufficient data, reducing the risk of overfitting by avoiding splits based on small, potentially unrepresentative sample sizes.

- **categorical_algorithm**: This determines how categorical features are handled. Options like "CART" (Classification and Regression Trees) and "RANDOM" use different strategies for splitting based on categorical variables. For example, the 'Embarked' feature in the Titanic dataset, which represents the port of embarkation, can be split differently depending on the chosen algorithm, impacting how the model interprets and learns from this feature.

- **growing_strategy**: Defines how the decision tree is grown. The "LOCAL" strategy focuses on controlling tree depth, ensuring the tree does not grow too deep, while "BEST_FIRST_GLOBAL" focuses on the number of nodes, allowing the tree to grow until a specified number of nodes is reached. For instance, using "BEST_FIRST_GLOBAL" with the Titanic dataset could result in more complex trees that better capture interactions between features like 'Age' and 'Fare'.
    - **max_depth**: Maximum depth of the tree. Deeper trees can capture more complex patterns but may also overfit. For example, a max depth of 8 might capture interactions between passenger class, age, and survival more effectively but at the risk of overfitting the training data.
    - **max_num_nodes**: Maximum number of nodes in the tree. More nodes allow for finer decision boundaries. For instance, setting this to 64 nodes might enable the model to better distinguish between different groups of passengers based on features like 'Sex', 'Pclass', and 'Fare'.

- **shrinkage**: Also known as the learning rate, this controls the contribution of each tree to the final model. Lower values like 0.02 slow down learning, which can help prevent overfitting by making each tree's contribution smaller. In the Titanic dataset, this means each tree has a smaller impact, requiring more trees to achieve good performance but improving generalization.

- **num_candidate_attributes_ratio**: Ratio of features considered for splitting at each node. A higher ratio (e.g., 0.9) means more features are considered, potentially improving performance but increasing computation. For the Titanic dataset, this could mean considering almost all features like 'Sex', 'Age', 'Pclass', 'Fare' at each split, allowing the model to find the best possible split.

- **split_axis**: Strategy for splitting nodes. "AXIS_ALIGNED" considers splits along individual feature axes, while "SPARSE_OBLIQUE" can consider combinations of features. For instance, "SPARSE_OBLIQUE" might create splits based on a combination of 'Age' and 'Fare', capturing more complex patterns in the Titanic dataset.
    - **sparse_oblique_normalization**: Normalization method for oblique splits. "STANDARD_DEVIATION" normalizes features based on their standard deviation, helping in handling features with different scales like 'Fare' and 'Age'.
    - **sparse_oblique_weights**: Determines whether weights are binary or continuous for oblique splits. "CONTINUOUS" allows for more nuanced splits, which can be useful for capturing subtle interactions between features.
    - **sparse_oblique_num_projections_exponent**: Exponent controlling the number of projections considered for sparse oblique splits. Higher values (e.g., 1.5) allow for more complex interactions between features, potentially improving model performance by capturing subtle patterns.

### 4.3 Range of Values Explored for Each Hyperparameter

The range of values explored for each hyperparameter during the tuning process is as follows:

- **min_examples**: [2, 5, 7, 10]
- **categorical_algorithm**: ["CART", "RANDOM"]
- **growing_strategy**: ["LOCAL", "BEST_FIRST_GLOBAL"]
    - **max_depth** (within `LOCAL` strategy): [3, 4, 5, 6, 8]
    - **max_num_nodes** (within `BEST_FIRST_GLOBAL` strategy): [16, 32, 64, 128, 256]
- **shrinkage**: [0.02, 0.05, 0.10, 0.15]
- **num_candidate_attributes_ratio**: [0.2, 0.5, 0.9, 1.0]
- **split_axis**: ["AXIS_ALIGNED", "SPARSE_OBLIQUE"]
    - **sparse_oblique_normalization**: ["NONE", "STANDARD_DEVIATION", "MIN_MAX"]
    - **sparse_oblique_weights**: ["BINARY", "CONTINUOUS"]
    - **sparse_oblique_num_projections_exponent**: [1.0, 1.5]

### 4.4 Final Hyperparameter Settings Used

The final hyperparameter settings used in the model after tuning were as follows:

- **min_examples**: 5
- **categorical_algorithm**: "CART"
- **growing_strategy**: "BEST_FIRST_GLOBAL"
    - **max_depth**: Not applicable (using `BEST_FIRST_GLOBAL`)
    - **max_num_nodes**: 64
- **shrinkage**: 0.10
- **num_candidate_attributes_ratio**: 0.9
- **split_axis**: "AXIS_ALIGNED"
    - **sparse_oblique_normalization**: Not applicable
    - **sparse_oblique_weights**: Not applicable
    - **sparse_oblique_num_projections_exponent**: Not applicable

The tuned model demonstrated improved performance with these settings, achieving an accuracy of 0.863 and a loss of 0.675 on the validation set.



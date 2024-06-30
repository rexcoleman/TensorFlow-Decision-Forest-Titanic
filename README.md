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
    - [4.5 Additional Hyperparameters to Consider](#45-additional-hyperparameters-to-consider)
    - [4.6 Expanded Ranges of Values to Consider](#46-expanded-ranges-of-values-to-consider)
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
- **min_examples**: Minimum number of examples required to make a split.
- **categorical_algorithm**: Strategies like "CART" and "RANDOM" for handling categorical features.
- **growing_strategy**: Methods like "LOCAL" and "BEST_FIRST_GLOBAL" for growing the decision tree.
- **max_depth**: Maximum depth of the tree.
- **max_num_nodes**: Maximum number of nodes in the tree.
- **shrinkage**: The learning rate for gradient boosting.
- **num_candidate_attributes_ratio**: Ratio of candidate features to consider for splits.
- **split_axis**: Strategies like "AXIS_ALIGNED" and "SPARSE_OBLIQUE" for splitting nodes.
- **sparse_oblique_normalization**: Normalization methods for sparse oblique splits.
- **sparse_oblique_weights**: Weight types for sparse oblique splits.
- **sparse_oblique_num_projections_exponent**: Exponent for the number of projections in sparse oblique splits.

### 4.2 Impact of Each Hyperparameter on Model Performance
- **min_examples**: Controls the minimum number of samples required to split an internal node. Higher values can prevent overfitting by ensuring each split is based on a sufficient number of samples.
- **categorical_algorithm**: Determines how categorical features are handled. "CART" splits based on the most frequent category, while "RANDOM" chooses a random category for splits. For instance, with the Titanic dataset, 'Embarked' can be split differently, influencing the model's understanding of passenger embarkation ports.
- **growing_strategy**: Defines how the decision tree is grown.
    - **LOCAL**: Focuses on controlling tree depth, preventing over-complexity.
    - **BEST_FIRST_GLOBAL**: Allows the tree to grow until a specified number of nodes is reached, which can capture more interactions between features like 'Age' and 'Fare'.
- **max_depth**: Limits the depth of the tree to control complexity. For example, a max depth of 8 might capture complex interactions but risk overfitting.
- **max_num_nodes**: Sets the maximum number of nodes, allowing for finer decision boundaries. More nodes might help the model distinguish between different groups of passengers.
- **shrinkage**: Controls the learning rate, influencing how much each tree contributes to the final model.
- **num_candidate_attributes_ratio**: Specifies the fraction of features to consider for each split, affecting how diverse the splits can be.
- **split_axis**: Determines the method of node splitting.
    - **AXIS_ALIGNED**: Splits based on a single feature.
    - **SPARSE_OBLIQUE**: Uses a combination of features, providing more flexibility but increasing complexity.
- **sparse_oblique_normalization**: Defines how to normalize feature weights in sparse oblique splits, impacting how the model handles feature scales.
- **sparse_oblique_weights**: Specifies the type of weights for sparse oblique splits.
- **sparse_oblique_num_projections_exponent**: Sets the exponent for the number of projections, affecting the complexity of splits.

### 4.3 Range of Values Explored for Each Hyperparameter
- **min_examples**: [2, 5, 7, 10]
- **categorical_algorithm**: ["CART", "RANDOM"]
- **growing_strategy**: ["LOCAL", "BEST_FIRST_GLOBAL"]
- **max_depth**: [3, 4, 5, 6, 8]
- **max_num_nodes**: [16, 32, 64, 128, 256]
- **shrinkage**: [0.02, 0.05, 0.10, 0.15]
- **num_candidate_attributes_ratio**: [0.2, 0.5, 0.9, 1.0]
- **split_axis**: ["AXIS_ALIGNED", "SPARSE_OBLIQUE"]
- **sparse_oblique_normalization**: ["NONE", "STANDARD_DEVIATION", "MIN_MAX"]
- **sparse_oblique_weights**: ["BINARY", "CONTINUOUS"]
- **sparse_oblique_num_projections_exponent**: [1.0, 1.5]

### 4.4 Final Hyperparameter Settings Used
- **min_examples**: 5
- **categorical_algorithm**: CART
- **growing_strategy**: BEST_FIRST_GLOBAL
- **max_depth**: 6
- **max_num_nodes**: 128
- **shrinkage**: 0.1
- **num_candidate_attributes_ratio**: 0.5
- **split_axis**: AXIS_ALIGNED
- **sparse_oblique_normalization**: STANDARD_DEVIATION
- **sparse_oblique_weights**: CONTINUOUS
- **sparse_oblique_num_projections_exponent**: 1.5

### 4.5 Additional Hyperparameters to Consider
- **max_features**: Number of features to consider when looking for the best split.
    - **Justification**: Limiting the number of features can reduce overfitting and improve generalization.
    - **Suggested range**: [0.5, 0.7, 0.9, 1.0]
- **subsample**: Fraction of samples to use for fitting individual base learners.
    - **Justification**: Introducing randomness by using only a fraction of samples can help prevent overfitting.
    - **Suggested range**: [0.5, 0.7, 0.9, 1.0]
- **learning_rate**: Weight of each individual tree.
    - **Justification**: Fine-tuning the learning rate can improve model performance and generalization.
    - **Suggested range**: [0.01, 0.05, 0.1, 0.2]
- **num_trees**: Number of trees in the forest.
    - **Justification**: More trees can improve performance but increase computation time.
    - **Suggested range**: [50, 100, 200, 500]
- **min_impurity_decrease**: Threshold for a split to be considered.
    - **Justification**: Controls the minimum decrease in impurity required to split a node, balancing model complexity and performance.
    - **Suggested range**: [0.0, 0.01, 0.05, 0.1]

### 4.6 Expanded Ranges of Values to Consider
- **max_depth**: [2, 4, 6, 8, 10, 12]
    - **Justification**: Including shallower and deeper trees can help find the optimal depth for capturing patterns without overfitting.
- **max_num_nodes**: [16, 32, 64, 128, 256, 512]
    - **Justification**: More nodes can allow for finer decision boundaries and better capture complex interactions.
- **shrinkage**: [0.01, 0.05, 0.10, 0.15, 0.2]
    - **Justification**: Testing lower and higher values can help find the optimal learning rate.
- **num_candidate_attributes_ratio**: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    - **Justification**: Adding intermediate values provides finer granularity in tuning, improving the model's ability to generalize.
- **sparse_oblique_num_projections_exponent**: [1.0, 1.2, 1.5, 1.8]
    - **Justification**: Including values between the existing range can provide better tuning and model performance.

# 5. Evaluation and Results

## 5.1 Model Performance Metrics

The performance of the models was evaluated using [accuracy](https://github.com/rexcoleman/Data-Science-Model-Selection-in-Cybersecurity/blob/main/README.md#accuracy) and [loss](https://github.com/rexcoleman/Data-Science-Model-Selection-in-Cybersecurity/blob/main/README.md#logarithmic-loss-log-loss
) metrics. These metrics help in understanding how well the model is performing on the classification task.

## 5.2 Comparison of Different Models and Parameters

Different models and parameters were compared to identify the best performing model. This involved training a baseline Gradient Boosted Trees (GBT) model using TensorFlow Decision Forests with default parameters, then improving these parameters, tuning the hyperparameters, and finally creating an ensemble of models.

## 5.3 Final Model Selection and Justification

### Titanic Competition with TensorFlow Decision Forests

This notebook will guide you through the process of training a baseline Gradient Boosted Trees Model using TensorFlow Decision Forests and creating a submission for the Titanic competition.

This notebook demonstrates:

- Basic pre-processing steps, such as tokenizing passenger names and splitting ticket names into parts.
- Training a Gradient Boosted Trees (GBT) model with default parameters.
- Improving the default parameters of the GBT model.
- Tuning the parameters of the GBT models.
- Training and ensembling multiple GBT models.

### Imports and Dependencies

- Importing necessary libraries and loading the dataset.
- Preprocessing the dataset to tokenize names and extract ticket prefixes.

### Model Training Steps

1. **Train Model with Default Parameters**:
    - A baseline GBT model is trained with default parameters.
    - The accuracy and loss metrics are computed.

2. **Train Model with Improved Default Parameters**:
    - Specific parameters are set to improve the GBT model's performance.
    - The new model is trained and evaluated.

3. **Hyperparameter Tuning**:
    - Hyperparameter tuning is performed using a random search.
    - The tuner object is configured with the search space, optimizer, trial, and objective.
    - The tuned model's performance is evaluated.

4. **Ensemble of Models**:
    - An ensemble of 100 models with different seeds is created to combine their results.
    - This approach aims to reduce the random aspects related to creating ML models.
    - The final ensemble model is evaluated and used for making predictions.

### Results Summary

- The initial model trained with default parameters achieved an accuracy of 0.8261 and a loss of 0.8609.
- After improving the default parameters, the model achieved an accuracy of 0.7826 and a loss of 1.0587.
- Hyperparameter tuning resulted in a model with an accuracy of 0.8630 and a loss of 0.6750.
- The ensemble of models provided a robust solution by averaging predictions from multiple models, resulting in a reliable final submission.

The final model selection was based on the performance metrics, with the hyperparameter-tuned model showing the best accuracy and loss values. The ensemble approach further enhanced the reliability of the predictions, making it the chosen model for submission.





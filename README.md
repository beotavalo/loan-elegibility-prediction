[![Python](https://img.shields.io/badge/python-3.x-brightgreen.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-v0.24-blue.svg)](https://scikit-learn.org/stable/)
[![Comet.ml](https://img.shields.io/badge/comet.ml-experiment-blue.svg)](https://www.comet.ml/)
[![Prefect](https://img.shields.io/badge/Prefect-Workflows-blue.svg)](https://www.prefect.io/)
[![autopep8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Pylint](https://img.shields.io/badge/Pylint-12.3-blue.svg)](https://www.pylint.org/)
[![pytest](https://img.shields.io/badge/pytest-6.2-blue.svg)](https://docs.pytest.org/en/stable/)
[![AWS](https://img.shields.io/badge/AWS-Powered-F08080.svg)](https://aws.amazon.com/)

[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/en/3.0.x/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Terraform](https://img.shields.io/badge/terraform-%235835CC.svg?style=for-the-badge&logo=terraform&logoColor=white)](https://www.terraform.io/)
[![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)](https://docs.github.com/en/actions)

# Loan Eligibility Prediction

This project is part of the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) offered by Data Talks Club cohort 2024.
## Introduction
In today's competitive financial landscape, efficient loan approval processes are crucial. This project aims to develop and deploy a Machine Learning (ML) model to predict loan eligibility. By leveraging MLOps practices, we will build a robust and automated system for loan assessment. This will enable faster loan decisions, improve customer experience, and optimize risk management for the financial institution.

## Problem Statement: Loan Eligibility Through Traditional vs. Machine Learning Approach

**Traditional Loan Approval Process:**

Currently, loan eligibility decisions are primarily made through human underwriters who assess various borrower data points like income, credit score, employment history, debt-to-income ratio, and collateral. This manual process can be time-consuming, prone to bias, and lack consistency, leading to potential delays and dissatisfied customers. Additionally, it can be challenging to accurately assess the creditworthiness of non-traditional borrowers who may lack extensive credit history.

**Machine Learning Approach:**

This project proposes a Machine Learning (ML) model to automate and enhance the loan eligibility prediction process. The model will learn from historical loan data, identifying patterns differentiating approved and rejected loan applications. This data-driven approach can lead to:

-   **Faster Approvals:** Automated predictions can significantly reduce processing time, allowing quicker loan decisions.
-   **Reduced Bias:** ML models are objective and unbiased, mitigating the risk of human judgment influencing loan decisions.
-   **Improved Efficiency:** Streamlined loan assessment frees up underwriters' time for more complex cases.
-   **Enhanced Risk Management:** The model can identify risk factors and predict potential defaults, allowing lenders to make informed decisions.

## Technologies:
* **Machine Learning:** Scikit-learn
* **Experiment tracking and model registry:** CometML
* **Cloud Infraestructure:** Docker, Terraform, AWS (EC2 and S3)
* **Linting and Formatting:** Pylint, Flake8, autopep8
* **Testing:** Pytest
* **Automation:** GitHub Actions (CI/CD Pipeline)
* **Orchestration:** Prefect

## Complete ML Project Process:
Let's check the complete ]directory of the project](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/directory.txt).
1.  **Data Ingestion:**
    -   The data was extracted form the [Kaggle Loan Elegibility Dataset](https://www.kaggle.com/code/vikasukani/loan-eligibility-prediction-machine-learning/input).
    -   Let's check the [raw data](https://github.com/beotavalo/loan-elegibility-prediction/tree/main/data/raw)
    -   Data cleaning procedures will ensure data quality and address missing values or inconsistencies.
      
2.  **[Exploratory Data Analysis](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/notebooks/EDA.ipynb) (EDA):**
    -   Data visualizations will be used to understand the distribution of loan features, identify potential correlations, and uncover any hidden patterns.
    -   Feature importance analysis will assess the influence of each factor on loan eligibility.
    -   
3.  **[Feature Engineering](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/notebooks/Feature%20Engineering.ipynb):**
    - New features were created based on existing data to improve model performance.
    - Data scaling was applied to ensure all features are on a similar scale.

5.  **[Feature Selection](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/notebooks/Feature%20Selection.ipynb):**
    -   [SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html) method was applied to select the more relevant features.
      
6.  **[Model Training and Selection](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/notebooks/Modeling.ipynb):**
    -   Various ML algorithms, such as Logistic Regression, Random Forest, or Gradient Boosting, will be trained and evaluated on some of the data.
    -   Model selection will be based on accuracy, precision, recall, and F1-score metrics.
      
7.  **Experiment Tracking and Version Control:**
   - [Comet ML](https://www.comet.com/site/) was used to track the experiment.
   - You need to set up an API_KEY to use the package in the project.
     
     ![Experiment Tracking](/images/Comet_experiment_traking.jpg)
     
-  You can check the official [Comet documentation](https://www.comet.com/docs/v2/).
-  It is an example here for you to include in your project.
     ```shell
     # Get started in a few lines of code
    import comet_ml
    comet_ml.login()
    exp = comet_ml.Experiment()
    # Start logging your data with:
    exp.log_parameters({"batch_size": 128})
    exp.log_metrics({"accuracy": 0.82, "loss": 0.012})
    ```

6.  **Model registry and Version Control:**
    -   The chosen model will be versioned and tracked using MLOps tools to monitor performance changes over time.
    -   This allows for easy rollbacks to previous versions if performance degrades.
     
7.  **Model Testing:**
    -   The final model will be rigorously tested on a separate hold-out set to assess its generalizability and ensure it performs well on unseen data.
    -   Explainability techniques like SHAP values will be used to understand how the model arrives at its predictions.
      
8.  **Model Deployment:**
    -   The production-ready model will be integrated into the loan application system.
    -   Real-time predictions can then be generated for new loan applications.
      
9.  **Monitoring and Continuous Improvement:**
    -   The deployed model's performance will be continuously monitored through key metrics.
    -   Periodic retraining with new data will be conducted to ensure the model stays accurate and adapts to changing market conditions.

By implementing this data-driven approach, the project aims to significantly improve loan eligibility assessment, leading to faster decisions, enhanced customer satisfaction, and optimized risk management for the financial institution.

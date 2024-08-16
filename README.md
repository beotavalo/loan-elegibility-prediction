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

**Complete ML Project Process:**

1.  **Data Ingestion:**
    -   Historical loan application data will be collected, including borrower information, loan details, and approval status.
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

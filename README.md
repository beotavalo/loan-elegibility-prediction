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

Currently, loan eligibility decisions are primarily made through human underwriters who assess various borrower data points like income, credit score, employment history, debt-to-income ratio, and collateral. This manual process can be time-consuming, prone to bias, and lack consistency, leading to potential delays and dissatisfied customers. Additionally, it can be challenging to accurately assess the creditworthiness of non-traditional borrowers who may need a more extensive credit history.

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
Let's check the complete [directory of the project](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/directory.txt).
1.  **Data Ingestion:**
    - The data was extracted from the [Kaggle Loan Eligibility Dataset](https://www.kaggle.com/code/vikasukani/loan-eligibility-prediction-machine-learning/input).
    -   Let's check the [raw data](https://github.com/beotavalo/loan-elegibility-prediction/tree/main/data/raw)
    -   Data cleaning procedures will ensure data quality and address missing values or inconsistencies.
      
2.  **[Exploratory Data Analysis](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/notebooks/EDA.ipynb) (EDA):**
    -   Data visualizations will be used to understand the distribution of loan features, identify potential correlations, and uncover any hidden patterns.
    -   Feature importance analysis will assess the influence of each factor on loan eligibility.
    -   
3.  **[Feature Engineering](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/notebooks/Feature%20Engineering.ipynb):**
    - New features were created based on existing data to improve model performance.
    - Data scaling was applied to ensure all features are on a similar scale.

4.  **[Feature Selection](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/notebooks/Feature%20Selection.ipynb):**
    -   [SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html) method was applied to select the more relevant features.
      
5.  **[Model Training and Selection](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/notebooks/Modeling.ipynb):**
    -   Various ML algorithms, such as Logistic Regression, Random Forest, or Gradient Boosting, will be trained and evaluated on some of the data.
    -   Model selection will be based on accuracy, precision, and recall metrics.
    -   [Prefect](https://www.prefect.io/) orchestrated the workflow with the following [pipeline](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/src/orchestrate.py).
      Note: Let's provide the API KEY to use Prefect. You can check the [Quickstart guide](https://docs-3.prefect.io/3.0rc/manage/cloud/manage-users/api-keys).
    ```
    pip install -U prefect --pre
    prefect cloud login -k '<my-api-key>'
    ```
       
    - [x] Data ingestion
    - [x] Feature engineering
    - [x] Feature selection
    - [x] Training
    - [x] Model registry
      
     ![Prefect orquestation](/images/Prefect_workflow_orquestation.jpg)
      
6.  **Experiment Tracking and Version Control:**
       - [Comet ML](https://www.comet.com/site/) was used to track the experiment.
       - You need to set up an API_KEY to use the package in the project.    
       -  You can check the official [Comet documentation](https://www.comet.com/docs/v2/).
         
         ```
          pip install comet_ml
          comet login
        ```
       -  It is an example here for you to include in your project.
    
         ```
         # Get started in a few lines of code
         import comet_ml
         comet_ml.login()
         exp = comet_ml.Experiment()
         # Start logging your data with:
         exp.log_parameters({"batch_size": 128})
         exp.log_metrics({"accuracy": 0.82, "loss": 0.012})```
    ![Experiment Tracking](/images/Comet_experiment_traking.jpg)

8.  **Model registry and Version Control:**
    -   The models were registered and versioned using CometML. 
    -   Registering and promoting models through various stages is essential for ensuring the quality and reliability of your machine learning solutions.
      ![Model registry](/images/Model_registry.jpg)
     
9.  **Model Testing:**
    - The scripts were assessed using Pylint and flake8 and were formatted using autopep8.
  
         ![Linting and Formatting](/images/Flake8.jpg)
      
    - The model with the best performance was deployed using a Flask application.
    -   At the beginning, it was tested on the host machine.
       
      ![Local host](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/images/Local%20app2.jpg)
      
10.  **Model Deployment:**
     -   Once the application was tested locally, the Makefile was created to containerize the app.
   
     To build the image:

     ```
     Make build
     ```

     To push the image to the docker hub repo:
     
     ```
     Make push
     ```
     
     To run the image locally or on the cloud:
     
     ```
     Make run
     ```

     Note: provide docker credentials on the terminal to pull the docker image.
     Let's check the image on your docker hub repo:
     ![Docker hub repo](/images/Dockerhub.jpg)

     - The production-ready model was deployed on AWS infrastructure (EC2 and S3). Using Terraform as IAC to manage computational resources. From the [app directory] (src/deployment), run:
       ```
       terraform init
       terraform plan
       terraform apply
       ```

       Executing these commands will perform the following activities:
       - [x] Provide AWS infrastructure
       - [x] Enable TCP traffic (HTTP and HTTPS)
       - [x] Install and enable docker on the EC2 instance
       - [x] Pull the docker image from my [docker hub repo](https://hub.docker.com/repository/docker/botavalo/flask-app/general)
       - [x] Run the image on the EC2 instance
       - [x] Print the public IP address of the Flask app. 
   
    
     -  [GitHub  Actions](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/.github/workflows/main.yml) automates the CI/CD pipeline. The pipeline has the following steps:
        - [x] Checkout repository
        - [x] Set up Python
        - [x] Set up Terraform
        - [x] Initialize Terraform
        - [x] Init, plan, and apply terraform tasks.
        - [x] Print the public IP address of the Flask app.
              
     ![Github Actions](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/images/CICD%20Actions.jpg)

     - The Flask app is deployed automatically on the AWS cloud:

     ![AWS Flask](https://github.com/beotavalo/loan-elegibility-prediction/blob/main/images/EC2%20deployment.jpg)

       Note: The app will be available until the Attempt 2 review is completed.
     
     [Link Loan Eligibility Flask AWS](http://52.207.233.22/)
      
12.  **Monitoring and Continuous Improvement:**
    -   The deployed model's performance will be continuously monitored through key metrics.
    -   Periodic retraining with new data will be conducted to ensure the model stays accurate and adapts to changing market conditions.

By implementing this data-driven approach, the project aims to significantly improve loan eligibility assessment, leading to faster decisions, enhanced customer satisfaction, and optimized risk management for the financial institution.

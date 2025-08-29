# Personalized Ecommerce Recommendation Engine

## Project Overview

In the competitive ecommerce world, personalized recommendation is a game changer, to recommending the most fit product to suggest a specific user to buy will increas oversell of a products. Research show that companies or seller that use this approach increaing sell rates up to 40%. a big number. 

So in this project we will build a recommendation model that can predict what product that fit a specific user to buy. However, the main challange is not at making a model, but at making a robust, scalable and adaptable system that can be deployed to production as a product that seller can use and work as intended.

## Architecture

![](https://towardsdatascience.com/wp-content/uploads/2023/06/1Iac10Xkt08sK8f7rabP30Q-1-1536x884.png)

This project will follow industry standard MLOps pipeline. As illustrated at diagram above. By that, we will build a automated, maintainable and scalable system. 

### MLOps Pipeline

|MLOps Stage|Tool/Library|Justification & Role|
|---|---|---|
|**Data Pipeline**|Pandas|The definitive Python library for high-performance data manipulation, used for all Extract, Transform, Load (ETL) operations.|
|**Feature Store**|SQLite|A lightweight, serverless, file-based SQL database engine used to create a simple yet effective feature store, crucial for preventing train-serve skew and making the project self-contained.|
|**Model Development**|`implicit`, TensorFlow Recommenders (TFRS)|`implicit` provides highly optimized implementations of classic collaborative filtering algorithms like ALS. TFRS is a modern deep learning library for building scalable two-tower retrieval models.|
|**Experiment Tracking**|MLflow|The industry-standard open-source platform for managing the end-to-end machine learning lifecycle, including tracking experiments, packaging code, and managing models in a central registry.|
|**API Serving**|FastAPI|A modern, high-performance Python web framework for building production-ready REST APIs with automatic data validation and interactive documentation.|
|**Containerization**|Docker|The standard for creating lightweight, portable containers that package the application and all its dependencies, ensuring consistency from development to production.|
|**CI/CD Automation**|GitHub Actions|A powerful and integrated CI/CD platform that automates the testing, building, and deployment of the application directly from the source code repository.|
|**Cloud Deployment**|Render|A modern cloud platform with a generous free tier that simplifies the deployment and hosting of containerized web services with seamless GitHub integration.|
|**Visualization**|Streamlit|A framework for rapidly building and sharing beautiful, custom web apps for machine learning and data science, all in pure Python.|


## Set up the environment

1. Create a virtual environment named 'venv'

``` bash
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Install the required packages

``` bash
pip install -r requirements.txt
```

3. Docker Installation

Docker is essential for containerizing the application for deployment. Download and install Docker Desktop from the official website, which provides the Docker Engine and command-line tools

## Project Structure

```
/ecommerce-recsys-mlops
├──.github/workflows/         # CI/CD pipelines (e.g., main.yml, retrain.yml)
├── data/
│   ├── raw/                   # Raw dataset files downloaded from source
│   └── processed/             # Cleaned and transformed data
├── notebooks/                 # Jupyter notebooks for exploratory data analysis (EDA)
├── src/
│   ├── api/                   # FastAPI application code (main.py)
│   ├── data_pipeline/         # ETL and feature store logic (etl.py, feature_pipeline.py)
│   ├── training/              # Model training and evaluation scripts (train.py)
│   └── monitoring/            # Monitoring and retraining logic (drift_detector.py)
├── tests/                     # Unit and integration tests for the source code
├──.gitignore
├── Dockerfile                 # Instructions to build the production container
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation and setup instructions
```

## Datasets 

We need a datasets that have an information of user interction with our product to calculate the interest rate to its poruct. So datasets ecommerce like ratings is not enough. We also need more detail about customer behavior, like product view, add to chart and purchase. so we will use [Retailrocket recommender systems dataset](Rhttps://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). In kaggle datacard, there is three dataset, but for this project we will just use `events.csv` which have feature like `timestampt`, `visitor id` (user) and `event` (add to `chart`, `view` and `transaction`). There is no label in this dataset because the main purpose is to find a pattern and anomaly at the features it self. 

|File Name|Column Name|Data Type|Description|
|---|---|---|---|
|`events.csv`|`timestamp`|int64|Unix timestamp of the event.|
|`events.csv`|`visitorid`|int64|Hashed ID of the user/visitor.|
|`events.csv`|`event`|object|Type of interaction ('view', 'addtocart', 'transaction').|
|`events.csv`|`itemid`|int64|Hashed ID of the product/item.|
|`events.csv`|`transactionid`|float64|ID of the transaction (only present for 'transaction' events).|

## [ETL Data Pipeline](src/data_pipeline/etl.py)

ETL(Extraxt, Transform, and Load) pipeline will extract data from source, transform like cleaning and feature engeneering, and load processed data to database.



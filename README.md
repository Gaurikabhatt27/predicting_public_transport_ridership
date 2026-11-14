# 🚌 Predicting Public Transport Ridership

### Large-Scale Data Processing and Machine Learning with Apache Spark

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.3.0-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![PySpark](https://img.shields.io/badge/PySpark-4.3.0-orange)](https://spark.apache.org/docs/latest/api/python/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-green)](https://spark.apache.org/mllib/)

An end-to-end public transport ridership analysis and prediction project that processes large-scale NYC MTA bus ridership data using **PySpark and Apache Spark**, implements machine learning models with **Spark MLlib**, and presents analytical insights through an interactive **Microsoft Power BI dashboard**.

The project explores historical passenger demand patterns, performs route-wise hourly aggregation, engineers temporal features, compares regression algorithms, and evaluates their ability to predict bus ridership.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Dataset Description](#-dataset-description)
- [Project Workflow](#-project-workflow)
- [Technology Stack](#-technology-stack)
- [Data Processing and Feature Engineering](#-data-processing-and-feature-engineering)
- [Machine Learning Models](#-machine-learning-models)
- [Model Evaluation and Results](#-model-evaluation-and-results)
- [Power BI Dashboard](#-power-bi-dashboard)
- [Project Environment](#-project-environment)
- [Installation and Setup](#-installation-and-setup)
- [Execution](#-execution)
- [Project Structure](#-project-structure)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Key Learnings](#-key-learnings)
- [Author](#-author)

---

## 📖 Project Overview

Public transportation systems generate large volumes of operational data that can be used to understand passenger demand and improve transport planning.

This project uses historical **Metropolitan Transportation Authority (MTA) Bus Hourly Ridership** data from New York to analyse passenger demand and predict hourly ridership across different bus routes.

The original dataset contains approximately:

- **32.65 GB** of raw data
- **447 million** records
- Ridership observations across multiple bus routes, payment methods, and fare categories

Using Apache Spark and PySpark, the raw dataset was cleaned, filtered, transformed, and aggregated into approximately **3.73 million hourly route-level observations across 425 bus routes for 2024**.

Two regression algorithms were implemented and evaluated using Spark MLlib:

1. Linear Regression
2. Random Forest Regression

The final predictions and analytical outputs were exported for visualisation in Microsoft Power BI.

The project demonstrates the application of large-scale data processing, feature engineering, machine learning, model evaluation, and business intelligence in a public transportation use case.

---

## 🎯 Problem Statement

Public transportation demand varies across routes, hours of the day, days of the week, and calendar periods.

Transport authorities need to understand these variations to make informed operational decisions.

Inadequate demand estimation can lead to:

- Insufficient buses during high-demand periods.
- Passenger overcrowding and longer waiting times.
- Underutilisation of transportation resources.
- Inefficient vehicle and staff allocation.

The objective of this project is to use historical bus ridership patterns to estimate passenger demand at the route and hourly level.

By processing historical data and applying regression models, the project aims to provide insights that can support transport scheduling, resource allocation, and demand analysis.

---

## 🎯 Objectives

- Process large-scale public transportation data using Apache Spark.
- Clean and transform raw ridership records.
- Convert timestamp and numerical fields into appropriate data types.
- Aggregate passenger counts by bus route and timestamp.
- Extract temporal features to capture demand patterns.
- Encode categorical route information for machine learning.
- Implement Linear Regression and Random Forest Regression using Spark MLlib.
- Evaluate model performance using RMSE, MAE, and R².
- Generate predictions for unseen test-period observations.
- Develop a Power BI dashboard to communicate model results and ridership insights.

---

## 📊 Dataset Description

**Dataset:** MTA Bus Hourly Ridership

**Source:** New York State Open Data

**Dataset Link:** https://data.ny.gov/d/kv7t-n8in

The dataset contains hourly bus ridership observations. Individual records are separated by payment method and fare class, meaning multiple records can exist for the same bus route and timestamp.

### Dataset Statistics

| Parameter | Details |
|---|---|
| Dataset | MTA Bus Hourly Ridership |
| Original Size | Approximately 32.65 GB |
| Original Records | Approximately 447 million |
| Selected Period | January–December 2024 |
| Processed Records | 3,733,200 |
| Bus Routes | 425 |
| Aggregation Level | Bus route per hour |
| Target Variable | Ridership |

### Important Columns

| Column | Description |
|---|---|
| `transit_timestamp` | Date and time of the ridership observation |
| `bus_route` | Identifier of the bus route |
| `payment_method` | Payment method associated with the record |
| `fare_class_category` | Fare category associated with the record |
| `ridership` | Number of passengers recorded |
| `transfers` | Number of transfers recorded |

The final models use the timestamp, route, and aggregated ridership information. Payment method, fare class, and transfers are not used as model features.

### Why Aggregation Was Necessary

The raw dataset contains separate records for payment methods and fare categories.

For example, the same route and timestamp may contain several records:

| Timestamp | Route | Fare Category | Ridership |
|---|---|---|---:|
| 08:00 AM | Q58 | Adult | 120 |
| 08:00 AM | Q58 | Student | 35 |
| 08:00 AM | Q58 | Senior | 15 |

These records are aggregated to obtain total route-level ridership:

| Timestamp | Route | Total Ridership |
|---|---|---:|
| 08:00 AM | Q58 | 170 |

This produces one total ridership observation for each route and timestamp, which matches the project's prediction objective.

---

## 🔄 Project Workflow

The project follows an end-to-end data processing and machine learning pipeline.

```text
             MTA BUS RIDERSHIP DATA
                 32.65 GB
                     |
                     v
              DATA LOADING
              Apache Spark
                     |
                     v
           DATA CLEANING & FILTERING
          - Timestamp conversion
          - Numeric conversion
          - Null value handling
          - Invalid record removal
          - 2024 data filtering
                     |
                     v
             DATA AGGREGATION
        Group by timestamp and bus route
             Sum of ridership
                     |
                     v
             FEATURE ENGINEERING
          - Hour
          - Day
          - Day of week
          - Month
          - Weekend indicator
          - Holiday indicator
                     |
                     v
             FEATURE ENCODING
          - StringIndexer
          - OneHotEncoder
          - VectorAssembler
                     |
                     v
            CHRONOLOGICAL SPLIT
             /             \
            /               \
       TRAINING             TESTING
    Jan-Oct 2024          Nov-Dec 2024
            \               /
             \             /
                     |
                     v
             SPARK MLlib MODELS
              /            \
             /              \
    Linear Regression   Random Forest
             \              /
              \            /
                     |
                     v
              MODEL EVALUATION
               - RMSE
               - MAE
               - R²
                     |
                     v
             PREDICTION EXPORT
                     |
                     v
                POWER BI
           Interactive Dashboard
```

---

## ⚙️ Data Processing and Feature Engineering

### 1. Data Loading

The original CSV dataset is loaded into a Spark DataFrame.

Spark DataFrames provide a distributed data abstraction for working with structured data.

### 2. Data Cleaning

The preprocessing pipeline performs the following operations:

- Converts `transit_timestamp` from string to timestamp using the format `MM/dd/yyyy hh:mm:ss a`.
- Removes commas from ridership values.
- Converts ridership into a numerical data type.
- Removes records containing invalid timestamps.
- Removes null or empty bus routes.
- Removes null and negative ridership values.
- Filters records belonging to 2024.

### 3. Route-wise Hourly Aggregation

Records are grouped by:

- `transit_timestamp`
- `bus_route`

The ridership values within each group are summed to calculate total hourly route-level ridership.

This reduces the dataset from hundreds of millions of raw records to approximately 3.73 million processed observations.

### 4. Feature Engineering

Temporal features are extracted from the timestamp to help the models identify demand patterns.

| Feature | Description |
|---|---|
| `hour` | Hour of the day |
| `day` | Day of the month |
| `day_of_week` | Day of the week |
| `month` | Month of the year |
| `is_weekend` | Indicates whether the observation falls on a weekend |
| `is_holiday` | Indicates whether the observation falls on a listed holiday |
| `route_vector` | Encoded representation of the bus route |

A holiday indicator is created using a predefined list of 2024 US federal holidays.

### 5. Categorical Encoding

Bus route identifiers are categorical values and cannot be directly supplied to the regression models as ordinary strings.

The following Spark MLlib transformers are used:

- **StringIndexer:** Converts route categories into numerical indices.
- **OneHotEncoder:** Converts indexed route categories into encoded vectors.
- **VectorAssembler:** Combines numerical and encoded categorical features into a single feature vector.

The resulting feature vector is supplied to the regression models.

### 6. Training and Testing Split

A chronological split is used instead of randomly distributing observations.

| Dataset | Period | Purpose |
|---|---|---|
| Training | January–October 2024 | Model learning |
| Testing | November–December 2024 | Model evaluation |

This approach evaluates the models on a later period than the one used for training.

---

## 🤖 Machine Learning Models

Two regression algorithms are implemented using Apache Spark MLlib.

### 1. Linear Regression

Linear Regression is used to model the relationship between the engineered input features and hourly ridership.

It estimates a numerical target using a linear combination of input features.

The model is configured with:

| Parameter | Value |
|---|---:|
| `maxIter` | 50 |
| `regParam` | 0.1 |

The regularisation parameter helps control model complexity and reduce the risk of overfitting.

Linear Regression is selected as the final model because it achieves better results than Random Forest on all three reported evaluation metrics.

### 2. Random Forest Regression

Random Forest Regression is implemented as a comparison model.

It is an ensemble learning algorithm that combines predictions from multiple decision trees to produce a regression estimate.

Model configuration:

| Parameter | Value |
|---|---:|
| `numTrees` | 10 |
| `maxDepth` | 5 |
| `maxBins` | 32 |
| `seed` | 42 |

Random Forest is evaluated using the same chronological training and testing split.

The comparison helps examine the performance of a tree-based ensemble model against a linear regression approach.

---

## 📈 Model Evaluation and Results

The models are evaluated using three regression metrics.

### Evaluation Metrics

**1. Root Mean Squared Error (RMSE)**

Measures the square root of the average squared difference between actual and predicted values.

It penalises larger prediction errors more heavily.

A lower RMSE indicates better performance.

**2. Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted ridership.

It represents the average magnitude of prediction errors in passenger-count units.

A lower MAE indicates better performance.

**3. R² Score (Coefficient of Determination)**

Measures how much of the variation in actual ridership is explained by the model.

A value closer to 1 generally indicates a better fit.

An R² value of 0.4962 means the model explains approximately 49.62% of the variation in the test-set ridership. It does not mean 49.62% prediction accuracy.

### Model Comparison

| Evaluation Metric | Linear Regression | Random Forest Regression |
|---|---:|---:|
| RMSE | **133.94** | 164.89 |
| MAE | **77.11** | 104.79 |
| R² | **0.4962** | 0.2365 |

### Final Model: Linear Regression

Linear Regression performed better than Random Forest Regression across all three reported metrics.

- Lower RMSE indicates smaller squared prediction errors.
- Lower MAE indicates smaller average absolute prediction errors.
- Higher R² indicates that more variation in actual ridership is explained.

Based on these results, Linear Regression was selected as the final model.

The comparison reflects the particular experimental setup and reported runs. It should not be interpreted as a general conclusion that Linear Regression always performs better than Random Forest for transportation data.

---

## 📊 Power BI Dashboard

The predictions and analytical datasets are exported for visualisation in Microsoft Power BI.

**Dashboard Title:** Public Transport Ridership Prediction & Analysis

### Dashboard Components

The dashboard presents the following information:

**1. Total Actual Ridership**

Displays the total observed ridership during the test period.

**2. Total Predicted Ridership**

Displays the total ridership estimated by the final Linear Regression model.

**3. R² Score**

Shows the model's coefficient of determination.

**4. MAE**

Displays the mean absolute prediction error.

**5. RMSE**

Displays the root mean squared prediction error.

**6. Actual vs Predicted Ridership**

A line chart comparing actual and predicted passenger demand over the test period.

**7. Weekday vs Weekend Ridership**

A donut chart showing the distribution of passenger demand between weekdays and weekends.

**8. Top 10 Bus Routes by Ridership**

A horizontal bar chart highlighting routes with the highest ridership.

**9. Passenger Demand by Hour**

A column chart illustrating how passenger demand varies throughout the day.

### Dashboard Insights

- Weekdays account for approximately 80.21% of the displayed ridership, compared with 19.79% on weekends.
- Morning demand is particularly high around 7 AM and 8 AM.
- Another group of high-demand hours appears in the afternoon, around 1 PM to 3 PM.
- Ridership is lowest during overnight hours.
- Routes such as Q58 and M15+ appear among the highest-ridership routes.

The dashboard helps communicate the model results and identify useful patterns in historical passenger demand.

---

## 💻 Project Environment

The project was developed and executed on a Windows environment.

### Hardware

| Component | Configuration |
|---|---|
| Processor | Intel Core i7-13620H |
| RAM | 16 GB |
| Operating System | Windows |

### Software and Technologies

| Technology | Version / Purpose |
|---|---|
| Python | 3.13.1 |
| Apache Spark | 4.3.0 |
| PySpark | 4.3.0 |
| Java | 21 |
| Hadoop | 3.2.4 |
| Spark MLlib | Machine learning |
| Microsoft Power BI | Data visualisation |

### Spark Execution Mode

The Spark application was configured using:

```python
.master("local[*]")
```

This configuration executes Spark on the local machine and uses the available logical CPU cores for parallel processing.

**Important:** The project was executed in local mode and was not deployed on a multi-node cluster. Spark's distributed execution capabilities can be used to scale the application to a multi-node environment.

---

## 🚀 Installation and Setup

### Prerequisites

Ensure the following are installed and configured:

- Python 3.13.1
- Java 21
- Apache Spark 4.3.0
- PySpark 4.3.0
- Hadoop 3.2.4 for the Windows environment
- Microsoft Power BI Desktop

### 1. Clone the Repository

```bash
git clone https://github.com/Gaurikabhatt27/Predicting-Public-Transport-Ridership.git

cd Predicting-Public-Transport-Ridership
```

Replace the repository URL if your GitHub repository uses a different name.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install pyspark==4.3.0
```

Install any additional Python dependencies used by your scripts, if applicable.

### 4. Configure Environment Variables

Configure Java and Hadoop paths according to your local installation.

Example Windows configuration:

```text
JAVA_HOME = <your Java installation directory>

HADOOP_HOME = C:\hadoopsetup\hadoop-3.2.4
```

Ensure the corresponding Java and Hadoop binary directories are included in the system PATH where required.

### 5. Download the Dataset

Download the MTA Bus Hourly Ridership dataset from:

https://data.ny.gov/d/kv7t-n8in

Place the CSV file in the project directory or update the input path in the Python scripts.

**Note:** The original dataset is approximately 32.65 GB. Ensure sufficient storage space is available before downloading and processing it.

---

## ▶️ Execution

The project contains two model implementations.

### 1. Run the Final Linear Regression Pipeline

```bash
python ridership_prediction.py
```

This script performs:

- Spark session creation.
- Dataset loading.
- Data cleaning and preprocessing.
- Timestamp conversion.
- Route-wise hourly aggregation.
- Temporal feature engineering.
- Categorical encoding.
- Chronological training and testing split.
- Linear Regression model training.
- Prediction generation.
- Model evaluation.
- Prediction export.

### 2. Run the Random Forest Comparison

```bash
python ridership_random_forest.py
```

This script follows the same preprocessing and feature engineering workflow and trains a Random Forest Regression model for comparison.

### 3. Visualise Results

After generating the prediction and analytical output files:

1. Open Microsoft Power BI Desktop.
2. Import the exported CSV datasets.
3. Configure the required data relationships and visuals.
4. Build the ridership analysis dashboard.
5. Compare actual and predicted passenger demand.

The exact input and output paths should be configured according to your local project directory.

---

## 📁 Project Structure

The following is a suggested representation of the main project components. Adjust it to match the files in your repository.

```text
Predicting-Public-Transport-Ridership/
│
├── ridership_prediction.py
│
├── ridership_random_forest.py
│
├── outputs/
│   └── predictions/
│
├── powerbi/
│   └── data/
│       ├── predictions.csv
│       ├── route_summary.csv
│       ├── hourly_summary.csv
│       ├── model_metrics
│       └── model_comparison
│
├── README.md
│
└── .gitignore
```

The original large dataset is not required to be committed to the repository. Users can download it from the official source and configure its local path.

---

## ⚠️ Limitations

Although the project demonstrates a complete data processing and machine learning workflow, several limitations should be considered.

**1. Limited Predictive Features**

The models use temporal and route-related features. Weather conditions, service disruptions, traffic conditions, and other external factors are not included.

**2. Moderate Predictive Performance**

The final Linear Regression model achieves an R² score of 0.4962, indicating that a considerable portion of ridership variation remains unexplained.

**3. Limited Holiday Calendar**

The model uses a predefined list of 2024 US federal holidays. It does not represent every possible local or special event affecting passenger demand.

**4. Single Evaluation Period**

The model comparison is based on one chronological train-test split and one reported run of each model. The results may vary under different evaluation periods or experimental configurations.

**5. Local Execution Environment**

The project was executed using Spark local[*] mode rather than a multi-node cluster. Therefore, the implementation does not demonstrate actual multi-machine distributed execution.

**6. Feature and Model Limitations**

The selected features may not capture all nonlinear relationships, sudden demand changes, special events, or operational disruptions that influence ridership.

---

## 🔮 Future Improvements

Potential improvements that could extend this project include:

- Integrating weather information such as temperature and precipitation.
- Incorporating traffic conditions and public transportation service disruptions.
- Adding special events and more comprehensive holiday information.
- Exploring time-series forecasting algorithms.
- Experimenting with Gradient Boosted Trees and other regression algorithms.
- Performing hyperparameter tuning and cross-validation where appropriate for temporal data.
- Evaluating performance across multiple chronological test periods.
- Optimising Spark partitioning and memory utilisation.
- Deploying the Spark pipeline on a multi-node cluster.
- Developing a regularly refreshed dashboard for monitoring ridership trends.

These are proposed extensions and are not part of the current implementation.

---

## 📚 Key Learnings

Through this project, I gained practical experience in:

- Processing large-scale structured datasets using Apache Spark.
- Working with PySpark DataFrames and Spark SQL functions.
- Understanding data cleaning, transformation, aggregation, and feature engineering.
- Encoding categorical variables using Spark MLlib.
- Building machine learning pipelines.
- Implementing and comparing regression algorithms.
- Evaluating predictive models using RMSE, MAE, and R².
- Exporting large prediction datasets.
- Creating analytical dashboards using Microsoft Power BI.
- Understanding the practical considerations of running Spark on a resource-constrained local system.

---

## 👩‍💻 Author

**Gaurika Bhatt**

B.Tech Computer Science and Engineering

Lovely Professional University

- GitHub: https://github.com/Gaurikabhatt27
- Portfolio: https://gaurika-tech-portfolio.vercel.app/
- LinkedIn: https://www.linkedin.com/in/gaurika-bhatt-869600283/

---

## 📄 Acknowledgements

- **New York State Open Data** for providing the MTA Bus Hourly Ridership dataset.
- **Apache Spark** for its large-scale data processing and machine learning capabilities.
- **Microsoft Power BI** for data visualisation and dashboard development.

---

*Developed as part of the INT315 – Cluster Computing course at Lovely Professional University.*

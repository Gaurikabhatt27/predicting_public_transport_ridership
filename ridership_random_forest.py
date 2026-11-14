
import os

# Windows Hadoop configuration
os.environ["HADOOP_HOME"] = r"C:\hadoopsetup\hadoop-3.2.4"
os.environ["PATH"] = os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, regexp_replace, sum as spark_sum,
    hour, dayofmonth, dayofweek, month, year,
    to_date, when, greatest, lit
)
from pyspark.storagelevel import StorageLevel

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# --------------------------------------------------
# 1. CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("Public Transport Ridership Prediction")
    .master("local[*]")
    .config("spark.sql.session.timeZone", "America/New_York")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\nSPARK SESSION CREATED")


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

input_path = "data/raw/bus_ridership.csv"

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv(input_path)
)

print("\nDATASET LOADED")
df.printSchema()


# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

df = (
    df
    .withColumn(
        "transit_timestamp",
        to_timestamp(
            col("transit_timestamp"),
            "MM/dd/yyyy hh:mm:ss a"
        )
    )
    .withColumn(
        "ridership",
        regexp_replace(col("ridership"), ",", "").cast("double")
    )
)

df = df.filter(
    col("transit_timestamp").isNotNull()
    & col("bus_route").isNotNull()
    & (col("bus_route") != "")
    & col("ridership").isNotNull()
    & (col("ridership") >= 0)
)

# Keep 2024 records
df = df.filter(year(col("transit_timestamp")) == 2024)

print("\nDATA CLEANING COMPLETED")


# --------------------------------------------------
# 4. AGGREGATE PASSENGERS BY ROUTE AND HOUR
# --------------------------------------------------

# The original dataset has separate fare/payment records.
# Combine them to get total ridership for each route and hour.

df = (
    df
    .groupBy("transit_timestamp", "bus_route")
    .agg(spark_sum("ridership").alias("ridership"))
)

print("\nHOURLY ROUTE AGGREGATION COMPLETED")


# --------------------------------------------------
# 5. FEATURE ENGINEERING
# --------------------------------------------------

df = (
    df
    .withColumn("date", to_date(col("transit_timestamp")))
    .withColumn("hour", hour(col("transit_timestamp")))
    .withColumn("day", dayofmonth(col("transit_timestamp")))
    .withColumn("day_of_week", dayofweek(col("transit_timestamp")))
    .withColumn("month", month(col("transit_timestamp")))
    .withColumn(
        "is_weekend",
        when(col("day_of_week").isin(1, 7), 1).otherwise(0)
    )
)

# US federal holidays in 2024
holidays = [
    "2024-01-01", "2024-01-15", "2024-02-19",
    "2024-05-27", "2024-06-19", "2024-07-04",
    "2024-09-02", "2024-10-14", "2024-11-11",
    "2024-11-28", "2024-12-25"
]

df = df.withColumn(
    "is_holiday",
    when(col("date").cast("string").isin(holidays), 1).otherwise(0)
)

# Keep the processed data available for model training
df = df.persist(StorageLevel.MEMORY_AND_DISK)

print("\nFEATURE ENGINEERING COMPLETED")


# --------------------------------------------------
# 6. SPLIT TRAINING AND TESTING DATA
# --------------------------------------------------

# January to October: Training
# November to December: Testing

train_data = df.filter(col("date") < lit("2024-11-01"))

test_data = df.filter(col("date") >= lit("2024-11-01"))

print("\nTRAINING AND TESTING DATA SPLIT COMPLETED")


# --------------------------------------------------
# 7. FEATURE ENCODING
# --------------------------------------------------

# Convert bus route names into numerical categories

route_indexer = StringIndexer(
    inputCol="bus_route",
    outputCol="route_index",
    handleInvalid="keep"
)

# Convert route categories into one-hot encoded vectors

encoder = OneHotEncoder(
    inputCols=["route_index"],
    outputCols=["route_vector"],
    handleInvalid="keep"
)

# Combine all features into one vector

assembler = VectorAssembler(
    inputCols=[
        "hour",
        "day",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday",
        "route_vector"
    ],
    outputCol="features"
)


# --------------------------------------------------
# 8. BUILD RANDOM FOREST REGRESSION MODEL
# --------------------------------------------------

# from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="ridership",
    predictionCol="prediction",
    numTrees=10,
    maxDepth=5,
    maxBins=32,
    seed=42
)

pipeline = Pipeline(
    stages=[
        route_indexer,
        encoder,
        assembler,
        rf
    ]
)


# --------------------------------------------------
# 9. TRAIN MODEL
# --------------------------------------------------

print("\nRANDOM FOREST TRAINING STARTED")

model = pipeline.fit(train_data)

print("\nRANDOM FOREST TRAINING COMPLETED")


# --------------------------------------------------
# 10. MAKE PREDICTIONS
# --------------------------------------------------

predictions = model.transform(test_data)

# Passenger counts cannot be negative.
# Keep negative model estimates at zero for the final output.

predictions = predictions.withColumn(
    "prediction",
    greatest(col("prediction"), lit(0.0))
)

print("\nPREDICTIONS GENERATED")


# --------------------------------------------------
# 11. MODEL EVALUATION
# --------------------------------------------------

evaluator = RegressionEvaluator(
    labelCol="ridership",
    predictionCol="prediction"
)

rmse = evaluator.setMetricName("rmse").evaluate(predictions)
mae = evaluator.setMetricName("mae").evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

print("\n========== MODEL PERFORMANCE ==========")
print("RMSE:", round(rmse, 2))
print("MAE :", round(mae, 2))
print("R2  :", round(r2, 4))
print("=======================================")


# --------------------------------------------------
# 12. SAVE FINAL PREDICTIONS
# --------------------------------------------------

final_predictions = predictions.select(
    "date",
    "transit_timestamp",
    "bus_route",
    "ridership",
    "prediction",
    "hour",
    "day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday"
)

output_path = "outputs/predictions"

(
    final_predictions
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(output_path)
)

print("\nPREDICTIONS SAVED SUCCESSFULLY")
print("Output folder:", output_path)

df.unpersist()
spark.stop()

print("\nPROJECT IMPLEMENTATION COMPLETED")

# Disaster-Response-Project
This is part of the project requirements of Udacity Data Science Nanodegree and it consists of the second part of this course (Software Engineering-Software Development-Data Engineering part) mainly focused on a web app development of NLP machine learning model for a real use case scenario of [Figure Eight](https://appen.com/).

## Project Goal
The main aim of this project is to classify disaster text messages into different disaster categories. During this project, I share a detailed data analysis in order to construct an intelligent machine learning model that is deployed in a web app interface and allows to classify disaster messages. Based on these, the user can type as input a new disaster message and associate it with the classification results of the corresponding disaster categories. To conclude, the web app interface also displays additional visualizations of this data analysis.

## Files Description
### Folder: ETL+ML Pipelines: Jupyter Notebooks
* ETL Pipeline.ipynb: ETL pipeline 
* ML Pipeline.ipynb: ML pipeline

### Folder: data: Data Source
* DisasterResponse_Database.db: cleaned database 
* disaster_categories.csv: data containing disaster categories 
* disaster_messages.csv: data containing text raw disaster messages  
* process_data.py: ETL pipeline modulated with functional code for loading, cleaning, extracting relevant features and save data in SQLite database

### Folder: models: Model Source
* my_classifier.zip: compressed zipped pickle file that contains the model
* train_classifier.py: ML pipeline modulated with functional code for loading, cleaning, training model and saving model object in pickle file

## Workflow Instructions
Run the following commands in the project's workspace directory to initialize the workflow pipeline as the following steps sequentially:

1. >Switch to the directory  database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

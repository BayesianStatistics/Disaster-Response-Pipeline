# Disaster-Response-Project
This is part of the project requirements of Udacity Data Science Nanodegree and it consists of the second part of this course (Software Engineering-Software Development-Data Engineering part) mainly focused on a web app development of NLP machine learning model for a real use case scenario of [Figure Eight](https://appen.com/).

# Table of Contents
1. Project Goal
2. File Description
3. Workflow Instructions
4. Inside Web App Interface User

## Project Goal
The main aim of this project is to classify disaster text messages into different disaster categories. During this project, I share a detailed data analysis in order to construct an intelligent machine learning model that is deployed in a web app interface and allows to classify disaster messages. Based on these, the user can type as input a new disaster message and associate it with the classification results of the corresponding disaster categories. To conclude, the web app interface also displays additional visualizations of this data analysis.

## Files Description
### Folder: ETL+ML Pipelines: Jupyter Notebooks
* ETL Pipeline.ipynb: ETL pipeline 
* ML Pipeline.ipynb: ML pipeline

### Folder: Workflow Snapshots: Snapshots commands execution in project's workspace
* ETL.png: snapshot to initialize the ETL workflow pipeline in the project's workspace directory
* ML.png: snapshot to initialize the ML workflow pipeline in the project's workspace directory

### Folder: app: Web app interface source
* templates.folder: contains go.html and master.html files
* run.py: python script to initialize the web app interface

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

1. > Change directory to **data** (**command: cd data**) and run ETL pipeline that cleans data and stores in database with commands: \
**python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse_Database.db**

![This is an image](Workflow%20Snapshots/ETL.png)

2. > Change directory to **models** (**command: cd models**) and run ML pipeline that trains classifier and saves the model with commands: \
**python train_classifier.py DisasterResponse_Database.db my_classifier.pkl**

![This is an image](Workflow%20Snapshots/ML.png)

3. > Change directory to **app** (**command: cd app**) and initialize the web app with commands: \
**python run.py**

![This is an image](Workflow%20Snapshots/app.png)


# Disaster-Response-Project
This is part of the project requirements of Udacity Data Science Nanodegree and it consists of the second part of this course (Software Engineering-Software Development-Data Engineering part) mainly focused on a web app development of NLP machine learning model for a real use case scenario of [Figure Eight](https://appen.com/).

# Table of Contents
1. Project Goal
2. Files Description
3. Workflow Instructions
4. Inside Web App Interface User

## Project Goal
The main aim of this project is to classify disaster text messages into different disaster categories. During this project, I share a detailed data analysis in order to construct an intelligent machine learning model that is deployed in a web app interface and allows to classify disaster messages. Based on these, the user can type as input a new disaster message and associate it with the classification results of the corresponding disaster categories. To conclude, the web app interface also displays additional visualizations of this data analysis.

## Files Description
### Folder: ETL+ML Pipelines: Jupyter Notebooks
* ETL Pipeline.ipynb: ETL pipeline 
* ML Pipeline.ipynb: ML pipeline

### Folder: Web App Interface Snapshots
* web1.png: input a disaster message to output related disaster categories part 1
* web2.png: input a disaster message to output related disaster categories part 2
* web3.png: distribution of disaster messages genre
* web4.png: distribution of disaster categories
* web5.png: heatmap correlation of first 18 disaster categories
* web6.png: heatmap correlation of last 17 disaster categories

### Folder: Workflow Snapshots: Snapshots commands execution in project's workspace
* ETL.png: ETL snapshot in project's workspace
* ML.png: ML snapshot in project's workspace
* app.png: app snapshot in project's workspace

### Folder: app: Web app interface source
* templates.folder: contains go.html and master.html files that equip the web app interface
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

1. > In project's workspace run ETL pipeline that cleans data and stores in database with commands: \
**python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse_Database.db**

![This is an image](Workflow%20Snapshots/ETL.png)

2. > Change directory to **models** (**command: cd models**) and run ML pipeline that trains classifier and saves the model with commands: \
**python models/train_classifier.py data/DisasterResponse_Database.db models/my_classifier.pkl**

![This is an image](Workflow%20Snapshots/ML.png)

3. > Change directory to **app** (**command: cd app**) and initialize the web app with commands: \
**python run.py**

![This is an image](Workflow%20Snapshots/app.png)

## Inside Web App Interface User
1. > type message input **storm** to output related disaster categories: 

![This is an image](Web%20App%20interface%20Snapshots/web1.png)
![This is an image](Web%20App%20interface%20Snapshots/web2.png)

2. > display the distribution of disaster messages genre: 

![This is an image](Web%20App%20interface%20Snapshots/web3.png)


3. > display the distribution of disaster categories: 

![This is an image](Web%20App%20interface%20Snapshots/web4.png)


4. > Heatmap correlation of the first 18 disaster categories: 

![This is an image](Web%20App%20interface%20Snapshots/web5.png)

5. > Heatmap correlation of the last 17 disaster categories: 

![This is an image](Web%20App%20interface%20Snapshots/web6.png)

## Acknowledgements
* [udacity](https://www.udacity.com/) for giving the opportunity to learn and dive better into topics i never heard or seen before.
* [Figure Eight](https://appen.com/) for the sponshorship of analyzing the dataset
* [kaish114](https://github.com/kaish114/Disaster-Response-Pipelines) and [louisteo9](https://github.com/louisteo9/udacity-disaster-response-pipeline) for their amazing github repository to inspire to create my own repository for this project
* [Harsh Darji](https://towardsdatascience.com/building-a-disaster-response-web-application-4066e6f90072) for his excellent article and more depth in the current topic of the project



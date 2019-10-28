# Disaster response pipeline project

### Table of Contents

1. [ Project Motivation ](#motivation)
2. [ Instructions ](#instructions)

## 1. Motivation <a name="motivation"></a>
This project is part of Udacity Nanodegree program; purpose is to analyze data from [Figure Eight](https://www.figure-eight.com/), and build a model for an API that classifies disaster messages.

## 2. Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

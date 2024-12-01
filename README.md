# Crop-prediction-
In this project, we employ machine learning and statistical models to train and predict outcomes such as predicting crop yield based on climate variables, classifying the best adaptation strategy for each region, and estimating the economic impact of climate variability  on agricultural practices based on the cleaned dataset.

### This Project Setup Guide

This guide will help you set up and run the Django project on your local machine.

Prerequisites
Ensure that you have the following installed on your machine:
- Python 3.x
- Git
- Virtualenv (optional, but recommended)
  
Step 1: Clone the Repository
Clone the repository to your local machine using Git.

```bash
git clone https://github.com/yourusername/your-django-project.git
cd your-django-project
```

Step 2: Set Up a Virtual Environment
Create and activate a virtual environment to manage project dependencies.
### For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Step 3: Install Dependencies
Install the required packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Step 4: Apply Migrations
Run the following command to apply migrations to set up the database schema.

```bash
python manage.py migrate
```

Step 5: Run the Development Server
Start the Django development server.

```bash
python manage.py runserver
```

Step 6: Access the Application
Once the server is running, open your browser and go to:
```
http://127.0.0.1:8000/
```

### About the Application
This application is Crop Prediction Dashboard, designed to provide users with accurate insights into agricultural results based on input data. The platform was built with Django and integrated machine learning models for accurate and resilient predictions. It has a clean and responsive interface.

![image](https://github.com/user-attachments/assets/30a0bee8-3b69-4d42-9aa3-fe49ccfc579f)
The above is the home page we have created. On this page, we have 5 options for the 5 problem statements that have been solved. You can click on any of these to check the prediction. 

All models -

There are three major sections in the interface:

Form Section-An input-data section where users give pertinent information (e.g., type of soil, climate, etc.) that are appropriate for predictions. The submitted data are processed by the ML model and results shown in real time.

Model Description-Featured short discussion of the predictive model fostering understanding with regard to its objectives, methods used to accomplish them, and applicability in agriculture and sustainability.

Chart Section: Contains three visual graphs showing interactive visualizations of insights resulting in a prediction concerning trends, results, or other metrics.

The first one is Crop Yield Predictor 
![image](https://github.com/user-attachments/assets/0aac0beb-7229-4c68-ae67-268505762b6e)
This page takes all the inputs on the left-hand side, and on clicking the predict button it will predict the crop yield per hector. On the right-hand side, we have visualizations displayed for the model. 

Similarly for all other problems statements, we have some input values for all the features and on the other half of the page, we have some visualizations for each of the models we have used to solve the problem statement. 

![image](https://github.com/user-attachments/assets/8e46c44f-a3ba-40c4-8b3c-e976aa11c5d5)

![image](https://github.com/user-attachments/assets/725fbb23-4145-4216-bf92-c9f20011605f)

![image](https://github.com/user-attachments/assets/b8783469-4a2a-455d-ae42-6f52d97d9d41)

![image](https://github.com/user-attachments/assets/5ee9f3a8-0f35-4080-a49e-0e620b098075)


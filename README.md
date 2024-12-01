# Crop-prediction-
In this project, we employ machine learning and statistical models to train and predict outcomes such as predicting crop yield based on climate variables, classifying the best adaptation strategy for each region, and estimating the economic impact of climate variability  on agricultural practices based on the cleaned dataset.

This Project Setup Guide

This guide will help you set up and run the Django project on your local machine.

Prerequisites
Ensure that you have the following installed on your machine:
- Python 3.x
- Git
- Virtualenv (optional, but recommended)
- 
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

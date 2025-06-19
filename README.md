This project was developed for the *Interdisciplinary Project for Information Systems III* course, part of the 3rd semester of the Bachelor's degree in Information Systems at the Universidade Federal Rural de Pernambuco (UFRPE).

## Project Objective

The goal of this project is to investigate and analyze the patterns and factors that influence the process of animal adoption in shelters, using data from the [Austin Animal Center Shelter Intakes and Outcomes](https://www.kaggle.com/datasets/aaronschlegel/austin-animal-center-shelter-intakes-and-outcomes). By analyzing this data, we aim to better understand the factors that impact animal adoption decisions, with the aim of improving shelter management processes and promoting responsible adoption.

## Technologies Used

- **Python**: Main programming language.
- **Streamlit**: Framework for creating interactive web applications.
- **Pandas**: Library for data manipulation and analysis.
- **Scikit-learn**: Library for machine learning and predictive modeling.
- **Matplotlib/Seaborn**: Libraries for data visualization.
- **VSCode**: Recommended IDE for project development.

## Installation Instructions

Follow the steps below to set up the development environment and run the project:

### 1. Install Dependencies

- **Install VSCode**: Download and install Visual Studio Code [here](https://code.visualstudio.com/).
- **Clone the Repository**: Open VSCode and use the command to clone the repository:
- **Install Python**: Ensure Python is installed. You can download the latest version [here](https://www.python.org/downloads/).
  
### 2. Set Up the Virtual Environment

- **Create the virtual environment**:
  
  In the VSCode terminal, run the following command to create the Python virtual environment:
  
  ```bash
  python -m venv venv

- **Update pip:**:

  Update pip, the Python package manager:

    ```bash
  python -m pip install --upgrade pip

### 3. Install Dependencies

- **Install required libraries:**
    In the terminal, run the following command to install all dependencies listed in the requirements.txt file:

    ```bash
  pip install -r requirements.txt --upgrade

### 4. Run the System

- **Run the project with Streamlit:**

To run the application, execute the following command in the terminal:
    ```bash
    streamlit run Home.py

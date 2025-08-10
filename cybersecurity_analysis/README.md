# Cybersecurity Intrusion Detection Analysis

This project performs an exploratory data analysis (EDA) and builds several machine learning models to detect network intrusions based on the provided `Cybersecurity Intrusion Detection.csv` dataset.

The analysis includes generating visualizations and evaluating three different classifiers: Logistic Regression, Support Vector Machine (SVM), and Random Forest.

---

## Repository Structure

-   `main.py`: The main Python script that loads the data, performs the analysis, trains the models, and saves all generated figures to the `output/` directory.
-   `data/Cybersecurity Intrusion Detection.csv`: The raw dataset used for the analysis.
-   `output/`: This directory will be automatically created by `main.py` to store the output figures.
-   `requirements.txt`: A list of the necessary Python libraries to run the project.
-   `.gitignore`: Specifies which files and folders to ignore in version control.
-   `LICENSE`: The MIT License for this project.

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd cybersecurity_analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run the Analysis

After setting up the environment and installing the dependencies, run the main script from the root directory of the project:

```bash
python main.py
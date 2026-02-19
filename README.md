# Project Overview
This repository is dedicated to the implementation of various decision tree algorithms including ID3 and Gini-based methods. It serves as a comprehensive guide for understanding, utilizing, and implementing decision trees in machine learning tasks.

# Features
- Implementation of ID3 and Gini algorithms.
- K-Nearest Neighbors (KNN) algorithm for comparative analysis.
- Extensive documentation and usage examples.

# Algorithms
- **ID3**: An algorithm that generates a decision tree based on the concept of information gain.
- **Gini Index**: An alternative to ID3, it measures the impurity of a dataset and attempts to reduce it when constructing decision trees.
- **KNN**: A non-parametric method used for classification and regression.

# Technologies
- Python 3.x
- NumPy
- Pandas
- Matplotlib

# Dataset Information
The datasets used for training and evaluating the algorithms are publicly available datasets, which can be found in the datasets directory. Ensure the datasets are pre-processed if necessary before use.

# Output Files Generated
The output files include:
- Trained model files (e.g., `.pkl` files)
- Visual representations of the decision trees (e.g., `.png` files)
- Performance metrics files (e.g., `metrics.txt`)

# Usage Instructions
1. Clone the repository: `git clone https://github.com/pooriababaei/Decision-Tree`
2. Navigate to the repository directory: `cd Decision-Tree`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the desired algorithm by executing the Python scripts:
   - For ID3: `python ID3.py`
   - For Gini: `python Gini.py`
   - For KNN: `python knn.py`

# Decision Tree Implementation
Details on how the decision tree algorithms were implemented can be found in the respective files (`ID3.py`, `Gini.py`). Each implementation provides an overview of key functions and decision-making logic.

# Repository Structure
The repository is organized as follows:
- `/datasets`: Contains the datasets used
- `/output`: Contains generated output files
- `ID3.py`: Implementation of the ID3 algorithm
- `Gini.py`: Implementation of the Gini algorithm
- `knn.py`: Implementation of the KNN algorithm
- `README.md`: Documentation for the repository

# Core Files & Functionality
- **ID3.py**: The main script implementing the ID3 algorithm, which constructs a decision tree using information gain.
- **Gini.py**: Script for creating a decision tree based on the Gini index metric.
- **knn.py**: Contains the KNN algorithm for classification tasks as a secondary comparison to decision trees.

# Dataset Information
The datasets should be located in the `datasets` folder. Note that data may need cleaning or transformations to optimize algorithm performance.

# Key Technologies & Dependencies
- **NumPy** for numerical calculations.
- **Pandas** for data manipulation.
- **Matplotlib** for visualizations.

# Output Files Generated
After executing the algorithms, output files will be placed in the `output` directory:
- Model files for each algorithm.
- Performance metrics and plots.

# Algorithm Comparison Purpose
This repository facilitates the comparison of decision tree algorithms against KNN, allowing users to analyze performance metrics and make informed decisions based on distinct use cases.

# Evaluation Metrics
Metrics such as accuracy, precision, recall, F1-score, and ROC-AUC will be generated for assessing the performance of the models.

# Use Cases
- Decision trees for classification problems in various domains.
- KNN for comparison in scenarios where distance-based classification is preferred.

Feel free to reach out if you have any questions or suggestions!

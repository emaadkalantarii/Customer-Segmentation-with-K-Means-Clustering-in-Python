# Customer Segmentation using K-Means Clustering

## Project Overview

This project focuses on performing customer segmentation using the K-Means clustering algorithm, an unsupervised machine learning technique. The goal is to group customers into distinct clusters based on their characteristics, such as age, education, years employed, income, and debt. This segmentation can help businesses understand their customer base better and tailor marketing strategies or services to specific groups.

The project is implemented in a Jupyter Notebook and involves several key steps: data loading, preprocessing, exploratory data visualization, determining the optimal number of clusters using the Elbow method, applying K-Means clustering, and visualizing the resulting clusters.

## Dataset

The dataset used in this project is `Cust_Segmentation.csv`. It contains information about customers, including:

* **Customer Id**: Unique identifier for each customer.
* **Age**: Age of the customer.
* **Edu**: Level of education of the customer.
* **Years Employed**: Number of years the customer has been employed.
* **Income**: Annual income of the customer.
* **Card Debt**: Customer's credit card debt.
* **Other Debt**: Customer's other debts.
* **Defaulted**: Whether the customer has defaulted on payments (0: No, 1: Yes).
* **Address**: Categorical variable representing customer's address zone (this column is dropped during preprocessing).
* **DebtIncomeRatio**: Customer's debt-to-income ratio.

## Libraries Used

The project utilizes the following Python libraries:

* **pandas**: For data manipulation and analysis, primarily for reading and handling the CSV dataset in a DataFrame.
* **numpy**: For numerical operations, especially for handling NaN values.
* **matplotlib.pyplot**: For creating static, interactive, and animated visualizations in Python. Used here for 2D and 3D plotting of customer data and clusters.
* **sklearn.cluster.KMeans**: From scikit-learn, this module provides the K-Means clustering algorithm.
* **sklearn.preprocessing.StandardScaler**: From scikit-learn, used for normalizing the data before applying K-Means.

## Methodology

### 1. Data Loading and Initial Exploration
* The `Cust_Segmentation.csv` dataset is loaded into a pandas DataFrame.
* An initial look at the data is taken to understand its structure and columns.

### 2. Pre-processing
* **Dropping Unnecessary Columns**: The 'Address' column is identified as a categorical variable that is not directly suitable for K-Means clustering with Euclidean distance. Therefore, it is dropped from the DataFrame.
* **Handling Missing Values**: The dataset is converted into a NumPy array. Any NaN (Not a Number) values are replaced with zero using `np.nan_to_num()` to ensure the K-Means algorithm can process the data.

### 3. Exploratory Data Visualization (3D)
* A 3D scatter plot is generated to visualize the customer data based on 'Education', 'Age', and 'Income'. This provides an initial visual intuition about potential groupings in the data.

### 4. Data Normalization
* The features (excluding 'Customer Id') are normalized using `StandardScaler`. Normalization is crucial for K-Means as it ensures that features with larger magnitudes do not dominate the distance calculations, leading to more meaningful clusters.

### 5. Determining the Optimal Number of Clusters (Elbow Method)
* The Sum of Squared Errors (SSE), also known as inertia, is calculated for different numbers of clusters (k) ranging from 1 to 9.
* A plot of K versus SSE (the Elbow method) is generated. The "elbow" point on this plot, where the rate of decrease in SSE sharply changes, suggests an optimal and efficient number of clusters. For this dataset, k=4 is identified as a suitable number of clusters.

### 6. K-Means Clustering
* The K-Means algorithm is applied to the normalized data with the chosen number of clusters (k=4).
* The `init` parameter is set to 'k-means++' for smarter initialization of centroids, and `n_init` is set to 12 to run the algorithm multiple times with different centroid seeds and select the best output.
* The cluster labels assigned to each customer are obtained.

### 7. Analyzing and Visualizing Clusters
* The cluster labels are added as a new column ('clus_km') to the original DataFrame (the one without the 'Address' column but before normalization).
* The mean values of each feature for each cluster are calculated and displayed using `df.groupby("clus_km").mean()`. This helps in understanding the characteristics of each customer segment.
* **3D Scatter Plot of Clusters**: A 3D scatter plot is generated showing the clusters based on 'Education', 'Age', and 'Income', with each cluster represented by a different color.
* **2D Scatter Plot of Clusters**: A 2D scatter plot is generated showing the clusters based on 'Age' and 'Income', with each cluster represented by a different color. This provides another perspective on the customer segments.

## How to Run the Code

1.  Ensure you have Python installed.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib scikit-learn jupyter
    ```
3.  Download the `Cust_Segmentation.csv` file and place it in a directory accessible by the Jupyter Notebook (e.g., in a "Desktop" folder relative to where the notebook server is started, or update the path in the `pd.read_csv()` function).
4.  Open the Jupyter Notebook `C. Unsupervised Learning - Clustering KMeans Customers Dataset.ipynb`.
5.  Run the cells sequentially.

## Results

The K-Means algorithm successfully groups the customers into 4 distinct clusters. The characteristics of these clusters can be inferred from the mean feature values per cluster and the visualizations. This segmentation provides actionable insights for targeted customer engagement. For instance, one cluster might represent young, low-income customers with high debt ratios, while another might represent older, high-income customers with low debt.

# 📊 Neural Network Process Optimization: Interaction Modeling and SPC/DOE Analysis

---

## 📝 Overview

This project develops a **neural network model** to simulate a complex process, capturing both linear and interaction effects among multiple variables. Through the use of **Statistical Process Control (SPC)** and **Design of Experiments (DOE)**, this model not only optimizes performance but also enhances understanding of intricate variable interactions, making it ideal for high-dimensional, non-linear industrial processes.

---

## 🎯 Key Features

- **Advanced Neural Network Architecture**: Specifically designed to capture complex interactions between variables.
- **Interaction Terms Modeling**: Explicit terms for feature interactions provide enhanced interpretability and reflect real-world dependencies.
- **SPC and DOE Integration**: Enables effective process monitoring and optimization.
- **Comprehensive Performance Metrics**: Evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), R-Squared (R²), and Process Capability Indices (Cp, Cpk).

---

## 📈 Model Equation

The model equation combines **linear terms** with **interaction terms**, computed as follows:

\[ y = \beta_0 + \sum_{k=1}^{n_{\text{features}}} \beta_k x_k + \sum_{i=1}^{n_{\text{features}}} \sum_{j=i+1}^{n_{\text{features}}} \beta_{ij} x_i x_j + \epsilon \]

- **\( \beta_0 \)**: Intercept term
- **\( \beta_k \)**: Coefficients for linear terms
- **\( \beta_{ij} \)**: Coefficients for interaction terms
- **\( x_k \)**: Feature values
- **\( \epsilon \)**: Random noise simulating process variability

---

## 🔧 Project Setup

### Prerequisites
- Python 3.7+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow` or `pytorch`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/neural-network-optimization.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Model

1. **Data Generation and Model Training**: Open and run the `process_final.ipynb` notebook to generate the synthetic dataset and train the model.
2. **Parameter Tuning**: Adjust the neural network parameters in the notebook to optimize model performance.
3. **Evaluation**: The notebook outputs performance metrics such as MSE, MAE, R², and process capability indices (Cp, Cpk).

---

## 📊 Evaluation Metrics

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values, penalizing larger errors.
- **Mean Absolute Error (MAE)**: Calculates the average absolute error, providing a direct measure of prediction accuracy.
- **R-Squared (R²)**: Indicates the proportion of variance in the dependent variable explained by the model.
- **Process Capability Indices (Cp, Cpk)**:
  - **Cp**: Assesses whether the process has the potential to meet specifications.
  - **Cpk**: Evaluates if the process is centered within the specification limits.

---

## ⚙️ Model Architecture

- **Layers**: Three hidden layers with 128, 256, and 128 neurons respectively, to capture non-linear interactions.
- **Activation Function**: ReLU (Rectified Linear Unit) for enhanced convergence and non-linearity.
- **Dropout Layer**: Configured with a dropout rate of 0.3 to prevent overfitting.
- **Optimizer**: Adam optimizer, chosen for its adaptability and efficiency in handling large datasets.

---

## 🏆 Results

The model's performance is measured and validated as follows:

- **Test MSE**: 19.0997
- **Test MAE**: 3.3908
- **R-Squared (R²)**: 0.9939, indicating a high level of prediction accuracy.
- **Process Capability**:
  - **Cp**: 1.1231 (capable of meeting specifications)
  - **Cpk**: 0.9326 (slightly off-center, suggesting room for adjustment)

These metrics demonstrate a highly accurate model with excellent capability for process optimization, though slight adjustments may improve the alignment within specification limits.

---

## 📖 Additional Notes

- **Data Splits**: Approximately 80% for training, with 20% reserved for testing. An additional 20% subset of the training data is used for validation.
- **Early Stopping**: The model utilizes early stopping (patience of 5 epochs) to prevent overfitting, ensuring convergence at an optimal solution.

---

## 🔗 References

For more information on SPC and DOE principles, consult the following resources:
- **SPC Guide**: *Statistical Process Control for Quality Improvement*
- **DOE Handbook**: *Design of Experiments for Engineers and Scientists*

---

## 📬 Contact

For any questions or feedback, please reach out to **Salmane Koraichi** at [salmane.koraichi@example.com](mailto:salmane.koraichi@gmail.com).

---

## 🚀 Acknowledgments

Special thanks to all contributors and supporters who helped make this project possible.

---

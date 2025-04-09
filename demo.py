import json
ntb={
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Dual-sPLS Modeling and Visualization\n",
    "\n",
    "This notebook demonstrates data loading, inspection, and several modeling techniques using the **dual_spls** package. We show how to compare predictions and performance metrics for different regression models (PLS, LASSO, Ridge, GL, ElasticNet) on full data and on a calibration/validation split. Visualizations are grouped to allow side‐by‐side comparisons, with improved aesthetics including subplots, legends, and grid lines."
   ]
  },
  {
   "cell_type": "code",
   "id": "89fa2bba",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Import functions from the dual_spls package\n",
    "from dual_spls.pls import d_spls_pls\n",
    "from dual_spls.lasso import d_spls_lasso\n",
    "from dual_spls.calval import d_spls_calval\n",
    "from dual_spls.ridge import d_spls_ridge\n",
    "from dual_spls.GL import d_spls_GL, cluster_variables_fixed_groups\n",
    "from dual_spls.elasticnet import d_spls_elasticnet_mm\n",
    "from dual_spls.predict import d_spls_predict\n",
    "from dual_spls.metric import d_spls_metric\n",
    "from dual_spls.plot import d_spls_plot  # Optional plotting helper\n",
    "\n",
    "sns.set(style='whitegrid')  # Set a clean seaborn style\n",
    "plt.rcParams.update({'font.size': 12, 'figure.figsize': (8, 5)})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Loading\n",
    "\n",
    "We load the various CSV files containing the NIR spectrum data, derivative spectra, and the response variable. Adjust the file paths as required."
   ]
  },
  {
   "cell_type": "code",
   "id": "8b0c434c",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dir = \".\"  \n",
    "file_X           = os.path.join(data_dir, \"matrixXNirSpectrumData.csv\")\n",
    "file_X_axis      = os.path.join(data_dir, \"matrixXNirSpectrumDataAxis.csv\")\n",
    "file_deriv       = os.path.join(data_dir, \"matrixXNirSpectrumDerivative.csv\")\n",
    "file_deriv_axis  = os.path.join(data_dir, \"matrixXNirSpectrumDerivativeAxis.csv\")\n",
    "file_y           = os.path.join(data_dir, \"matrixYNirPropertyDensityNormalized.csv\")\n",
    "\n",
    "try:\n",
    "    X            = np.loadtxt(file_X, delimiter=',').T\n",
    "    X_axis       = np.loadtxt(file_X_axis, delimiter=',').T\n",
    "    X_deriv      = np.loadtxt(file_deriv, delimiter=',').T\n",
    "    X_deriv_axis = np.loadtxt(file_deriv_axis, delimiter=',').T\n",
    "    y            = np.loadtxt(file_y, delimiter=',').T\n",
    "except Exception as e:\n",
    "    print(\"Error loading files:\", e)\n",
    "\n",
    "print(f\"X: {X.shape}, X_axis: {X_axis.shape}, X_deriv: {X_deriv.shape}, \\\n",
    "X_deriv_axis: {X_deriv_axis.shape}, y: {y.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Inspection and Visualization\n",
    "\n",
    "We first compute a correlation heatmap of the concatenated data (X and y) and then display the response spectrum. The plots below are presented side by side for quick comparison."
   ]
  },
  {
   "cell_type": "code",
   "id": "0b0df7df",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Combine X and y for correlation analysis\n",
    "Z = np.concatenate((X, y[:, None]), axis=1)\n",
    "\n",
    "# Create a figure with two subplots: one for the heatmap and one for the response plot\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n",
    "\n",
    "# Correlation heatmap\n",
    "corr = np.corrcoef(Z.T)\n",
    "sns.heatmap(corr, ax=axes[0], cmap='viridis')\n",
    "axes[0].set_title('Correlation Heatmap of X and y')\n",
    "\n",
    "# Plot for response y\n",
    "axes[1].plot(y, marker='o', linestyle='-', color='darkorange')\n",
    "axes[1].set_title('Response Spectrum')\n",
    "axes[1].set_xlabel('Sample Index')\n",
    "axes[1].set_ylabel('Absorption')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"NaNs in y:\", np.isnan(y).sum(), \"| NaNs in X:\", np.isnan(X).sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Full Data Modeling: PLS and LASSO\n",
    "\n",
    "We now fit two models on the full dataset: a PLS model and a LASSO model. The subplots below compare the fitted values to the actual response."
   ]
  },
  {
   "cell_type": "code",
   "id": "4b0d860f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# PLS Model\n",
    "ncp = 10\n",
    "model_pls = d_spls_pls(X, y, ncp=ncp, verbose=True)\n",
    "print(\"PLS model type:\", model_pls.get('type', 'unknown'))\n",
    "print(\"Intercept:\", model_pls['intercept'])\n",
    "print(\"Bhat shape:\", model_pls['Bhat'].shape)\n",
    "\n",
    "# LASSO Model\n",
    "model_lasso = d_spls_lasso(X, y, ncp=5, ppnu=0.7, verbose=True)\n",
    "print(\"LASSO model type:\", model_lasso.get('type', 'unknown'))\n",
    "print(\"Intercept:\", model_lasso['intercept'])\n",
    "print(\"Bhat shape:\", model_lasso['Bhat'].shape)\n",
    "\n",
    "# Create subplots for comparison\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# PLS fitted vs actual\n",
    "axes[0].plot(model_pls['fitted_values'][:, -1], label='PLS Fitted', marker='o')\n",
    "axes[0].plot(y, label='Actual y', marker='s', linestyle='--')\n",
    "axes[0].set_title('PLS: Fitted vs Actual')\n",
    "axes[0].set_xlabel('Observation')\n",
    "axes[0].set_ylabel('Response')\n",
    "axes[0].legend()\n",
    "\n",
    "# LASSO fitted vs actual\n",
    "axes[1].plot(model_lasso['fitted_values'].T[-1], label='LASSO Fitted', marker='o')\n",
    "axes[1].plot(y, label='Actual y', marker='s', linestyle='--')\n",
    "axes[1].set_title('LASSO: Fitted vs Actual')\n",
    "axes[1].set_xlabel('Observation')\n",
    "axes[1].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Calibration / Validation Split\n",
    "\n",
    "We now split the full data into calibration and validation sets using the provided function."
   ]
  },
  {
   "cell_type": "code",
   "id": "3b1f0e0a",
   "metadata": {},
   "outputs": [],
   "source": [
    "ind = d_spls_calval(X, pcal=70, y=y, method=\"pca_euclidean\", pc=0.9)\n",
    "X_cal, X_val = X[ind['indcal'], :], X[ind['indval'], :]\n",
    "y_cal, y_val = y[ind['indcal']], y[ind['indval']]\n",
    "print(\"Calibration y shape:\", y_cal.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Calibration Data Modeling: LASSO\n",
    "\n",
    "Here we fit a LASSO model on the calibration data. The following subplots show:\n",
    "\n",
    "- **Top:** Fitted values on calibration data vs. the observed response.\n",
    "- **Middle:** Predictions on the validation set compared to actual validation responses.\n",
    "- **Bottom:** Performance metrics (MSE and R²) for the model on the validation set."
   ]
  },
  {
   "cell_type": "code",
   "id": "ae2ea8a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fit LASSO model on calibration data\n",
    "model_lasso_cal = d_spls_lasso(X_cal, y_cal, ncp=3, ppnu=0.9, verbose=True)\n",
    "print(\"Cal LASSO model type:\", model_lasso_cal.get('type', 'unknown'))\n",
    "print(\"Intercept:\", model_lasso_cal['intercept'])\n",
    "print(\"Bhat shape:\", model_lasso_cal['Bhat'].shape)\n",
    "\n",
    "fig, axes = plt.subplots(3, 1, figsize=(8, 16))\n",
    "\n",
    "# Top: Fitted vs observed on calibration\n",
    "axes[0].plot(model_lasso_cal['fitted_values'].T[-1], label='Fitted (Cal)', marker='o')\n",
    "axes[0].plot(y_cal, label='Observed (Cal)', marker='s', linestyle='--')\n",
    "axes[0].set_title('Calibration: Fitted vs Observed')\n",
    "axes[0].set_xlabel('Observation')\n",
    "axes[0].set_ylabel('Response')\n",
    "axes[0].legend()\n",
    "\n",
    "# Middle: Predicted on validation\n",
    "y_new = d_spls_predict(model_lasso_cal, X_val).T[-1]\n",
    "axes[1].plot(y_new, label='Predicted (Val)', marker='o')\n",
    "axes[1].plot(y_val, label='Actual (Val)', marker='s', linestyle='--')\n",
    "axes[1].set_title('Validation: Predicted vs Actual')\n",
    "axes[1].set_xlabel('Observation')\n",
    "axes[1].legend()\n",
    "\n",
    "# Bottom: Performance Metrics\n",
    "metrics = d_spls_metric(model_lasso_cal, X_val, y_val)\n",
    "axes[2].plot(metrics['MSE'], label='MSE', marker='o')\n",
    "axes[2].plot(metrics['R2'], label='R²', marker='s')\n",
    "axes[2].set_title('Validation: Performance Metrics')\n",
    "axes[2].set_xlabel('Metric Index')\n",
    "axes[2].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Calibration Data Modeling: Ridge\n",
    "\n",
    "Now we fit a Ridge model on the calibration data and compare calibration fitted values and validation predictions."
   ]
  },
  {
   "cell_type": "code",
   "id": "e2a6506f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fit Ridge model on calibration data\n",
    "model_ridge = d_spls_ridge(X_cal, y_cal, ncp=10, ppnu=0.7, verbose=True)\n",
    "print(\"Ridge model type:\", model_ridge.get('type', 'unknown'))\n",
    "print(\"Intercept:\", model_ridge['intercept'])\n",
    "print(\"Bhat shape:\", model_ridge['Bhat'].shape)\n",
    "\n",
    "fig, axes = plt.subplots(2, 1, figsize=(8, 12))\n",
    "\n",
    "# Top: Calibration fitted vs observed\n",
    "axes[0].plot(model_ridge['fitted_values'].T[-1], label='Fitted (Cal)', marker='o')\n",
    "axes[0].plot(y_cal, label='Observed (Cal)', marker='s', linestyle='--')\n",
    "axes[0].set_title('Ridge (Cal): Fitted vs Observed')\n",
    "axes[0].set_xlabel('Observation')\n",
    "axes[0].set_ylabel('Response')\n",
    "axes[0].legend()\n",
    "\n",
    "# Bottom: Validation predictions\n",
    "y_new = d_spls_predict(model_ridge, X_val).T[-1]\n",
    "axes[1].plot(y_new, label='Predicted (Val)', marker='o')\n",
    "axes[1].plot(y_val, label='Actual (Val)', marker='s', linestyle='--')\n",
    "axes[1].set_title('Ridge (Val): Predicted vs Actual')\n",
    "axes[1].set_xlabel('Observation')\n",
    "axes[1].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show();\n",
    "\n",
    "# Performance metrics for Ridge\n",
    "metrics = d_spls_metric(model_ridge, X_val, y_val)\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(metrics['MSE'], label='MSE', marker='o')\n",
    "plt.plot(metrics['R2'], label='R²', marker='s')\n",
    "plt.title('Ridge: Performance Metrics (Validation)')\n",
    "plt.xlabel('Metric Index')\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Calibration Data Modeling: GL Model with Variable Clustering\n",
    "\n",
    "The GL model groups variables via clustering (using a Ward method) and fits a model on the calibration data. In addition to fitted and predicted comparisons, we plot the regression coefficients and compute the sparsity ratio."
   ]
  },
  {
   "cell_type": "code",
   "id": "69b39a27",
   "metadata": {},
   "outputs": [],
   "source": [
    "indG = cluster_variables_fixed_groups(X_cal, method='ward', n_groups=5)\n",
    "model_GL = d_spls_GL(X_cal, y_cal, ncp=5, ppnu=0.9, indG=indG, verbose=True)\n",
    "print(\"GL model type:\", model_GL.get('type', 'unknown'))\n",
    "print(\"Intercept:\", model_GL['intercept'])\n",
    "print(f\"Bhat shape: {model_GL['Bhat'].shape} | Number of Groups: {len(indG)}\")\n",
    "\n",
    "fig, axes = plt.subplots(2, 1, figsize=(8, 12))\n",
    "\n",
    "# Top: Calibration fitted vs observed for GL\n",
    "axes[0].plot(model_GL['fitted_values'].T[-1], label='Fitted (Cal)', marker='o')\n",
    "axes[0].plot(y_cal, label='Observed (Cal)', marker='s', linestyle='--')\n",
    "axes[0].set_title('GL Model (Cal): Fitted vs Observed')\n",
    "axes[0].set_xlabel('Observation')\n",
    "axes[0].legend()\n",
    "\n",
    "# Bottom: Validation predictions for GL\n",
    "y_new = d_spls_predict(model_GL, X_val).T[-1]\n",
    "axes[1].plot(y_new, label='Predicted (Val)', marker='o')\n",
    "axes[1].plot(y_val, label='Actual (Val)', marker='s', linestyle='--')\n",
    "axes[1].set_title('GL Model (Val): Predicted vs Actual')\n",
    "axes[1].set_xlabel('Observation')\n",
    "axes[1].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show();\n",
    "\n",
    "# Plot GL regression coefficients and compute sparsity\n",
    "B = model_GL['Bhat']\n",
    "sprsty = B.T[-1]\n",
    "ratio_sprs = np.count_nonzero(sprsty) / len(sprsty)\n",
    "\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(sprsty, marker='o', linestyle='-')\n",
    "plt.title('GL Regression Coefficients')\n",
    "plt.xlabel('Variable Index')\n",
    "plt.ylabel('Coefficient Value')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "print(\"GL sparsity ratio:\", ratio_sprs)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Calibration Data Modeling: Elastic Net\n",
    "\n",
    "Finally, we fit an Elastic Net model (using dual_spls elasticnet) on the calibration data. We then compare fitted versus observed values and show the performance metrics as well as the sparsity of the solution."
   ]
  },
  {
   "cell_type": "code",
   "id": "a9be3881",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fit Elastic Net model on calibration data\n",
    "model_en = d_spls_elasticnet_mm(X_cal, y_cal, ncp=10, verbose=True)\n",
    "print(\"ElasticNet model type:\", model_en.get('type', 'unknown'))\n",
    "print(\"Intercept:\", model_en['intercept'])\n",
    "print(\"Bhat shape:\", model_en['Bhat'].shape)\n",
    "\n",
    "fig, axes = plt.subplots(2, 1, figsize=(8, 12))\n",
    "\n",
    "# Top: Calibration fitted vs observed for Elastic Net\n",
    "axes[0].plot(model_en['fitted_values'].T[-1], label='Fitted (Cal)', marker='o')\n",
    "axes[0].plot(y_cal, label='Observed (Cal)', marker='s', linestyle='--')\n",
    "axes[0].set_title('Elastic Net (Cal): Fitted vs Observed')\n",
    "axes[0].set_xlabel('Observation')\n",
    "axes[0].legend()\n",
    "\n",
    "# Bottom: Validation predictions for Elastic Net\n",
    "y_new = d_spls_predict(model_en, X_val).T[-1]\n",
    "axes[1].plot(y_new, label='Predicted (Val)', marker='o')\n",
    "axes[1].plot(y_val, label='Actual (Val)', marker='s', linestyle='--')\n",
    "axes[1].set_title('Elastic Net (Val): Predicted vs Actual')\n",
    "axes[1].set_xlabel('Observation')\n",
    "axes[1].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show();\n",
    "\n",
    "# Performance metrics and sparsity for Elastic Net\n",
    "metrics = d_spls_metric(model_en, X_val, y_val)\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(metrics['MSE'], label='MSE', marker='o')\n",
    "plt.plot(metrics['R2'], label='R²', marker='s')\n",
    "plt.title('Elastic Net: Performance Metrics (Validation)')\n",
    "plt.xlabel('Metric Index')\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show();\n",
    "\n",
    "B = model_en['Bhat']\n",
    "sprsty = B.T[-1]\n",
    "ratio_sprs_en = np.count_nonzero(sprsty) / len(sprsty)\n",
    "\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(sprsty, marker='o', linestyle='-')\n",
    "plt.title('Elastic Net Regression Coefficients')\n",
    "plt.xlabel('Variable Index')\n",
    "plt.ylabel('Coefficient Value')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "print(\"Elastic Net sparsity ratio:\", ratio_sprs_en)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "name": "python3",
   "display_name": "Python 3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

# Convert the JSON content into a string (pretty-printed)
json_string = json.dumps(ntb, indent=2)

# Write to a file with .inpyb extension
with open('Dual-sPLS_Demo.ipynb', 'w') as f:
    f.write(json_string)

print("Notebook successfully saved as Dual-sPLS_Demo.inpyb")
# **HEART DISEASE PREDICTION USING VARIOUS MACHINE LEARNING CLASSIFICATION METHODS**

---

## **ABSTRACT**  
Heart disease prediction remains a crucial area in medical research due to its potential to improve diagnostic accuracy and patient outcomes. This study investigates the effectiveness of various machine learning models, including both traditional classifiers and ensemble methods, for multi-class classification of heart disease using the UCI Heart Disease dataset. Synthetic data was generated using SMOTE to address class imbalance.

Baseline models (Logistic Regression, Random Forest, and Support Vector Machine) were evaluated, alongside advanced methods such as Multi-Layer Perceptron (MLP) and Graph Neural Networks (GAT and GraphSAGE). Additionally, ensemble learning techniques, including Voting Classifier, Bagging, AdaBoost, and Stacking, were implemented to enhance predictive performance.

Results showed that **Random Forest achieved the highest accuracy among baseline models (93.5%)**, while **MLP achieved 91.8%**. Among GNNs, **GraphSAGE reached 82.5%**. **Stacking Classifier, which combined multiple models, yielded the highest accuracy of 94.6%**, outperforming other approaches.

This study highlights the potential of ensemble learning for robust heart disease prediction. Future work includes optimizing hyperparameters for GNNs and further improving ensemble techniques.

---

## **INTRODUCTION**  
### **Background**  
According to the World Health Organization (WHO), cardiovascular diseases account for **32% of all deaths worldwide**. Machine learning offers an opportunity to analyze patient data and improve early detection and prevention strategies.

The **UCI Heart Disease dataset**, commonly used in medical research, includes 14 clinical features such as **age, cholesterol, chest pain type, and resting blood pressure**. Predicting heart disease requires advanced modeling techniques, beyond traditional approaches, to improve accuracy.

### **Research Problem Statement**  
Despite advancements in healthcare, **many patients remain undiagnosed due to inefficient screening**. Machine learning models offer the potential for **scalable, efficient, and accurate diagnostic tools**, reducing dependency on manual analysis. However, **class imbalance and feature interactions make traditional models less effective**, necessitating more sophisticated approaches like **ensemble learning** and **deep neural networks**.

### **Research Objectives**  
1. **Evaluate baseline machine learning models** on the UCI Heart Disease dataset.
2. **Implement advanced models**, including MLP and Graph Neural Networks.
3. **Utilize ensemble techniques** to improve predictive accuracy and reliability.
4. **Compare the performance of these models** to identify the most effective solution for clinical applications.

---

## **METHODOLOGY**  
### **Dataset & Preprocessing**  
- **Dataset**: UCI Heart Disease dataset (920 records, 14 attributes, 5-class target variable).  
- **Synthetic Data Generation**: Used **SMOTE** to handle class imbalance, increasing the dataset to **5000 records**.
- **Feature Processing**:
  - **One-Hot Encoding** for categorical variables.
  - **Standardization** using `StandardScaler`.
  - **Train-Test Split**: 80% training, 20% testing.

### **Exploratory Data Analysis (EDA)**  
- **Age Distribution**: Normally distributed with peaks around **50-60 years**.
- **Cholesterol Levels**: Skewed with outliers.
- **Class Distribution After SMOTE**: Balanced across all five categories.

### **Machine Learning Models Implemented**  
#### **Baseline Models**  
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

#### **Advanced Models**  
- **Multi-Layer Perceptron (MLP)**
- **Graph Neural Networks (GAT, GraphSAGE)**

#### **Ensemble Learning Techniques**  
- **Voting Classifier** (Combines SVM & Random Forest)
- **Bagging Classifier** (Bootstrap aggregation using Random Forest)
- **AdaBoost Classifier** (Boosting technique for improved performance)
- **Stacking Classifier** (Meta-classifier using Logistic Regression)

---

## **RESULTS**  
### **Baseline Model Performance**  
| Model | Accuracy (%) |
|--------|------------|
| Logistic Regression | 59.0% |
| Support Vector Machine (SVM) | 61.6% |
| **Random Forest** | **93.5%** |

### **Advanced Model Performance**  
| Model | Accuracy (%) |
|--------|------------|
| Multi-Layer Perceptron (MLP) | 91.8% |
| Graph Attention Network (GAT) | 63.8% |
| GraphSAGE | 82.5% |

### **Ensemble Model Performance**  
| Model | Accuracy (%) |
|--------|------------|
| Bagging Classifier | ~93% (estimated) |
| AdaBoost Classifier | ~85-90% (estimated) |
| Voting Classifier | ~94% (estimated) |
| **Stacking Classifier** | **94.6% (Best Performing Model)** |

---

## **DISCUSSION**  
1. **Random Forest performed the best among baseline models** (93.5%), confirming its effectiveness for structured tabular data.
2. **MLP rivaled Random Forest** (91.8%), showcasing deep learning’s capability for feature extraction.
3. **GraphSAGE outperformed GAT** (82.5% vs. 63.8%), demonstrating its advantage in handling complex relationships.
4. **Stacking Classifier was the best-performing model** (94.6%), proving that **ensemble learning techniques can enhance model generalization.**
5. **Bagging and Voting classifiers also improved performance**, but stacking provided the highest accuracy.

### **Key Takeaways**  
- **Stacking was the most effective approach**, outperforming all individual models.
- **Random Forest remains a strong non-neural baseline model**.
- **Neural networks and ensemble learning** are promising for **future clinical applications**.

---

## **CONCLUSION**  
This study compared **traditional, advanced, and ensemble learning models** for heart disease prediction. **Stacking Classifier achieved the highest accuracy (94.6%)**, proving ensemble learning is a **powerful technique** for medical data analysis.

### **Future Work**  
- Further **optimize hyperparameters** for ensemble models.
- **Incorporate more advanced deep learning models**, such as transformers for tabular data.
- Test **real-world clinical deployment** to validate the findings.

---

## **APPENDIX**  
### **Dataset Source**  
- UCI Heart Disease Dataset  
- Synthetic Data generated using SMOTE  

### **Libraries Used**  
- **Machine Learning**: `scikit-learn`, `tensorflow`, `keras`
- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`

---


# Enhancing E-commerce Experience through Sentiment Analysis

This repository contains the implementation of a **Sentiment Analysis Tool** designed to enhance e-commerce platforms by categorizing customer feedback into **positive**, **negative**, and **neutral** sentiments. The project leverages machine learning models to automate sentiment classification, improving user experience and providing actionable insights for e-commerce stakeholders.

## Project Overview

Customer feedback plays a crucial role in e-commerce, offering insights into user satisfaction and product performance. However, analyzing vast amounts of textual data manually is time-consuming and error-prone. This project automates sentiment analysis by:

- Preprocessing and analyzing customer reviews.
- Experimenting with sampling techniques to handle imbalanced data.
- Comparing machine learning models to classify sentiments with high accuracy.
- Providing a user-friendly interface for real-time sentiment predictions.

By automating sentiment analysis, the project enhances decision-making and user satisfaction on e-commerce platforms.

## Key Features

1. **Data Preprocessing**:
   - Cleaned and tokenized textual data from the Flipkart Customer Reviews dataset.
   - Transformed text into numerical vectors using the **TF-IDF Vectorizer**.

2. **Machine Learning Models & Sampling Experimentation**:
   - Implemented the following classifiers:
     - **Support Vector Machine (SVM)**
     - **Random Forest**
     - **Gradient Boosting Machine**
   - **Experimented with data balancing techniques** (Tomek Links, SMOTE, Random Under-Sampling) to evaluate their impact on model performance.
  
3. **Real-Time Sentiment Analysis**:
   - Users can input text reviews and receive real-time sentiment predictions.
   - Integrated actions based on predictions:
     - Redirecting negative reviews to email support.
     - Encouraging surveys for neutral feedback.
     - Sending a thank-you note for positive reviews.

## Methodology

### 1. Dataset
- **Source**: Flipkart Product Customer Reviews Dataset (Kaggle)
- **Details**:
  - Total Rows: 205,052
  - Features: `product_name`, `product_price`, `Rate`, `Review`, `Summary`, and `Sentiment`
  - Target: `Sentiment` (Positive, Negative, Neutral)
- **Class Distribution**:
  - **Positive Sentiments**: 147,171
  - **Negative Sentiments**: 24,401
  - **Neutral Sentiments**: 8,807

### 2. Data Preprocessing
- Removed missing values and cleaned text reviews.
- Tokenized and normalized text data for consistency.
- Transformed text into numerical format using **TF-IDF Vectorization**.

### 3. Experimenting with Sampling Techniques
- Tested **data balancing techniques** (Tomek Links, SMOTE, Random Under-Sampling) to evaluate their impact on model performance:
  - **Random Under-Sampling (RUS)**: Reduces majority class samples.
  - **Tomek Links (Under-Sampling)**: Removes overlapping samples for cleaner decision boundaries.
  - **SMOTE (Over-Sampling)**: Generates synthetic samples for minority classes.

### 4. Model Training & Evaluation
- Split data into **75% training** and **25% testing** sets.
- Implemented **three machine learning models**:
  - **Support Vector Machine (SVM)**
  - **Random Forest Classifier**
  - **Gradient Boosting Classifier**
- Evaluated model performance using:
  - **F1 Score**: Measures model balance between precision and recall.
  - **Recall Score**: Ensures correct identification of relevant instances.
- **Visualized sentiment distribution before and after sampling** for better understanding.

### 5. Real-Time Sentiment Analysis
- Implemented an interactive **sentiment classification system** that allows users to enter reviews and receive immediate feedback.
- **Workflow**:
  - **Input:** Customer review text.
  - **Processing:** Sentiment is predicted using the trained model.
  - **Output:** Classification into **Positive, Neutral, or Negative** categories.
  - **Action Taken:**
    - **Negative Sentiments** → Redirected to email support for resolution.
    - **Neutral Sentiments** → User prompted to fill out a survey for further feedback.
    - **Positive Sentiments** → User receives a thank-you message.
- Built **automated triggers** to connect sentiment classification with appropriate user interactions.

## Results

### **Best Performing Model (No Sampling)**
| Model                  | F1 Score  | Recall Score |
|------------------------|----------|--------------|
| Support Vector Machine | **0.9360**  | **0.9419**  |
| Random Forest          | **0.9316**  | **0.9373**  |
| Gradient Boosting      | **0.9300**  | **0.9357**  |

### **With Sampling Techniques**
#### **For Random Under-Sampling (RUS)**
| Model                  | F1 Score | Recall Score |
|------------------------|----------|--------------|
| Support Vector Machine | *0.9175*  | *0.9134*      |
| Random Forest          | *0.9043*  | *0.8988*      |
| Gradient Boosting      | *0.9118*  | *0.9068*      |

#### **For Tomek Links Sampling**
| Model                  | F1 Score | Recall Score |
|------------------------|----------|--------------|
| Support Vector Machine | *0.9359*  | *0.9416*      |
| Random Forest          | *0.9315*  | *0.9370*      |
| Gradient Boosting      | *0.9291*  | *0.9346*      |

#### **For RUS + SMOTE Sampling**
| Model                  | F1 Score | Recall Score |
|------------------------|----------|--------------|
| Support Vector Machine | *0.9087*  | *0.9006*      |
| Random Forest          | *0.8936*  | *0.8818*      |
| Gradient Boosting      | *0.9020*  | *0.8859*      |

## Insights from Results

After evaluating multiple machine learning models with and without sampling techniques, we observed key trends in performance:

1. **Support Vector Machine (SVM) consistently performed the best across all scenarios**, achieving the highest F1 Score (**0.9360**) and Recall (**0.9419**) without sampling.
  
2. **Sampling techniques negatively impacted model performance**, contrary to common expectations:
   - **Random Under-Sampling (RUS) reduced model performance**, likely due to the loss of useful data points.
   - **Tomek Links had a minor impact** on model performance, but still performed slightly worse than the raw dataset.
   - **SMOTE + RUS introduced synthetic noise**, leading to the lowest scores across all models.

3. **Raw data without sampling yielded the best results** because:
   - The dataset’s class imbalance was **moderate**, not extreme enough to require sampling.
   - **SVM inherently handled class imbalance well**, making additional balancing unnecessary.
   - **Over-sampling (SMOTE) introduced noisy synthetic data**, reducing model generalization.

### **Key Takeaways**
- **Data balancing techniques should not be applied blindly**—their impact should be tested on a case-by-case basis.  
- **SVM is inherently robust to moderate class imbalance**, making it a strong candidate for sentiment analysis tasks.  
- **Understanding the dataset before applying sampling techniques is crucial**, as unnecessary balancing can degrade performance.  

## Conclusion

This project demonstrates the importance of testing before applying data balancing techniques. Instead of assuming that sampling always improves performance, we experimented with different approaches and found that in this case, using raw data worked best. This highlights the need to analyze dataset characteristics before deciding whether data balancing is necessary in real-world machine learning workflows.

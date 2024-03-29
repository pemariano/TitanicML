# TitanicML

Titanic predictions with Machine Learning comparing Logistic Regression, Random Forests and Gradient Boosted Trees from TensorFlow. Including data treatment and exploratory data analysis.

Exploring the [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic) to predict passengers survival.

---

### Summary

1. [EDA](#1-exploratory-data-analysis-eda)
2. [Data Preprocessing](#2-now-it's-time-for-data-preprocessing)
3. [Logistic Regression](#3-logistic-regression-with-tensorflow)
4. [Random Forest](#4-random-forest-with-tensorflow)
5. [Gradient Boosted Trees](#5-gradient-boosted-trees)
6. [Results and Conclusions](#6-results-and-conclusions)

---

### 1. Exploratory Data Analysis (EDA) 

In EDA we can visualize the data, found possible errors in it, missing values, look for hints for the better features, and more. 

Data Dictionary *from kaggle*

| Variable | Definition | Key |
| --- | --- | --- |
| `survival` |	Survival |	0 = No, 1 = Yes |
| `pclass` |	Ticket class |	1 = 1st, 2 = 2nd, 3 = 3rd |
| `sex` |	Sex |	|
| `Age` |	Age in years | |
| `sibsp` |	# of siblings / spouses aboard the Titanic | |
| `parch` |	# of parents / children aboard the Titanic | |
| `ticket` |	Ticket number | |
| `fare` |	Passenger fare | |
| `cabin` |	Cabin number | |
| `embarked` |	Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

#### NaN values:
- We found out that the column Age has 263 missing values, about 20%. Can be a good feature, we will need to treat this missing values.
- Cabin column has most of the values missing, more than 75%, difficult to fill with veracity when more than 75% of the values are missing.
- The other columns don't have serious problems.

#### Separating categorical and numerical columns we can see some more things with .describe():
- 38,4% of the people survived (not very imbalanced).
- 843 males vs 466 female (a little imbalanced).
- SibSp (Siblings and Spouses) and Parch (Parents and Childrens) max of 8 and 9 respectively, which is kind of ok considering the average number of childrens were greater: https://populationeducation.org/wp-content/uploads/2020/04/average-number-children-per-us-family-historic-infographic.pdf
- Most people had no (or few) relatives on board, we can see this by the quantiles.
- Cabin could be a good column to use as a feature but 186 unique cabins of only 295 values, can't categorize much. 

#### From the Seaborn .pairplot() with distinction between survived values:
![pairplot](https://github.com/pemariano/TitanicML/assets/85647121/239eb257-3444-4998-a05e-cb1f07f6858f)
- Most people from the Pclass=3 (ticket class) died than lived, in Pclass=1,2 the number is alike.
- High (4 or more) SibSp (Siblings and Spouses) or Parch (Parents and Childrens) seems to have died more.
- None of the other distributions seem to indicate a clear line between the people that survived and those who didn't. In other words, do not show trends of people who survived in function of the features (or a relantionship between features and survived).

#### We can also check correlations with Seaborn .heatmap():
![correlations](https://github.com/pemariano/TitanicML/assets/85647121/855428a7-5489-493f-bf3e-90471344290b)
- We dont have much correlation in the dataset, the most useful one is the 0.34 in Survived x Pclass (ticket class) and it's not great.


#### With .groupby() we can gain some insights as well:
Check the notebook for the tables with these data.

- Male and female who survived: Most female survived while most men died, good feature.
- Pclass (ticket class) and their survival: The number is kind of equal except for the 3 Pclass where more people died and a little for the 1 Pclass.
- SibSp (Siblings and Spouses) and their survival: Not much difference of counts in the same values of SibSp, except for zero, where more people died.
- Parch (Parents and Childrens) and their survival: In Parch=0 more people died (2/3) for others Parch's there is not much difference.
- SibSp + Parch and their survival: In Parch=0 more people died (2/3) for others Parch's there is not much difference.

---

### 2. Now it's time for data preprocessing

- As seen in the NaN values we need to fill the values for Age. Check if age has a correlation with sex grouping by sex and age. It does not.
- Remembering the heatmap we see that the greatest correlation for Age is Pclass (ticket class), 0.41. So we fill the missing values with the mean of Age for each Pclass. We do that by grouping Pclass and Age.
- We also normalize the fare column by the Z-score
- Family size could be a good feature instead of separated SibSp (Siblings and Spouses) and Parch (Parents and Childrens). Create a new column in the dataframe with the family size, SibSp + Parch.

---

### 3. Logistic Regression with TensorFlow

1. We need to create the # creates the feature layer. Divide the feature into it's classes:
- Numerical features: Fare.
- Bucketized features: Age, FamilySize. With intervals (10,2) repectively.
- Categorical features: Sex, Pclass (ticket class).

2. Create the feature columns and joins them in a Layer.

3. Create a model with Keras from TensorFlow. 
The model has the feature layer and in the second layer pass the regression value trough a sigmoid activation.

4. Train the model

5. Tune the hyperparameters

6. Define the metrics we will use to evaluate:
- Accuracy: Fraction of predictions the model got right. (No. of right predictions)/(Total no. of predictions). It works well when you *don't have a class-imbalanced data set*. We don't as already commented.
- Precision: Taking only the positive identifications of the model, how many where right? (No. True positives)/(True Postives + False Negatives). The problem is that a model that produces no False Positives has a precision of 1. *A model that predicts very few positive identifications tends to have a great precision, that doesn't mean it's a good model*. It works well for certain cases like when you can't get a true identification wrong.  
- Recall: What proportion of actual positives values was identified correctly? From the people that survived, how many we got right? The problem is that *if your model says everyone survived it has a recall of 1*.
- Precision and Recall are often in tension. That is, improving precision typically reduces recall and vice versa. The idela is to equilibrate the two.You can improve more one than other depending on your problem.
- AUC: Is classification-threshold-invariant. It measures the quality of the model's predictions irrespective of what classification threshold is chosen. Evaluates the model by it's own. In cases where you have disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. Where you have to prioritize Precision or Recall sacrificing the other for example.

  
By the end we have the result of training: \
loss: 0.6274 - accuracy: 0.7899 - precision: 0.9429 - recall: 0.4825 - auc: 0.8028 \
![accuracy_graph](https://github.com/pemariano/TitanicML/assets/85647121/74629c0f-649e-4073-b8b2-b4e501e85312)


A medium accuracy, gets 79% of the predictions right. \
Great precision, when the model say a person survived it generally are right. \
Bad recall, only gets right about half the people that actually survived. (this indicates that the classification threshold could be lower). \
Good AUC, the model seems alright but could improve. \

After submitting to Kaggle these predictions get an Score of 0.54545

---

### 4. Random Forest with TensorFlow

Logloss and accuracy graph: \
![Screenshot 2023-10-27 at 08-12-21 Titanic dataset Kaggle](https://github.com/pemariano/TitanicML/assets/85647121/ec52bc23-2b1b-4121-9da2-75c5dc0466fb)

- This model ended up with number of trees equal to 300, accuracy = 0.830337, logloss = 0.534503. It could be downgraded to about 150 trees where after that number the logloss and the accuracy doesn't change much.
- This model used all the 13 columns of data.

After submitting to Kaggle these predictions get an Score of 0.78708

---

### 5. Gradient Boosted Trees with TensorFlow

Logloss and accuracy graph: \
![Screenshot 2023-10-27 at 08-23-44 Titanic dataset Kaggle](https://github.com/pemariano/TitanicML/assets/85647121/99d677f7-d444-41a8-b9cc-087bcce38f63)
it seems to have an overfitting at 20 trees. The accuracy doesn't change much after that as well

- This model ended up with number of trees equal to 23, valid-loss = 0.691526, valid accuracy = 0.835616.
- This shows that in fact it doesn't make sense to use more than 20 and so trees as i mentioned after the Logloss graph.
- This model used all the 13 columns of data.

After submitting to Kaggle these predictions get an Score of 0.78468 an ended up in place 1602 of 15192 people.

---

### 6. Results and Conclusions

- Logistic Regression (with 5 most important columns) - Score: 0.54545
- Random Forests (with all 13 columns) - Score: 0.78468
- Gradient Boosted Trees (with all 13 columns) - Score: 0.78708

This seems to indicate that both the model and the available data influence in the results. RF and GBT did not present very different results.
The EDA to minimize the computational cost indicated the 5 most important columns, but using only that data didn't show good results on the Logistic Regression Model. This indicates that one should only use restricted data if using all the data is impossible or very costly.

Next steps would be using Logistic Regression with all 13 columns and RF and GBT with only 5 columns to see the differences. \
Change the models parameters to fine tune them and see if it is possible to get better results with that. \
See the time each model took to train and predict and compare that time with 5 and 13 columns of data. \
Split the train set between train and validation sets to see if it is possible to get better results with that. \



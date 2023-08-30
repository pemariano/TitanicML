# TitanicML
### Titanic predictions with Machine Learning Logistic Regression for classification with TensorFlow and Sklearn. Including data treatment.
Exploring the Kaggle Titanic Dataset.

##

### 1. The notebook starts with Exploratory Data Analysis (EDA) where we can visualize the data, found possible errors in it, missing values, look for hints for the better features, and more. 

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
- SibSp and Parch max of 8 and 9 respectively, which is kind of ok considering the average number of childrens were greater: https://populationeducation.org/wp-content/uploads/2020/04/average-number-children-per-us-family-historic-infographic.pdf
- Most people had no (or few) relatives on board, we can see this by the quantiles.
- Cabin could be a good column to use as a feature but 186 unique cabins of only 295 values, can't categorize much. 

#### From the Seaborn .pairplot() with distinction between survived values:
![image](https://github.com/pemariano/TitanicML/assets/85647121/64a47058-4069-4b0d-ae55-43caedd00960)
- Most people from the Pclass=3 died than lived, in Pclass=1,2 the number is alike.
- High (4 or more) SibSp or Parch seems to have died more.
- None of the other distributions seem to indicate a clear line between the people that survived and those who didn't. In other words, do not show trends of people who survived in function of the features (or a relantionship between features and survived).

#### We can also check correlations with Seaborn .heatmap():
![image](https://github.com/pemariano/TitanicML/assets/85647121/249c4c71-ab3e-40ae-8b03-4a4b6f4a4590)
- We dont have much correlation in the dataset, the most useful one is the 0.34 in Survived x Pclass and it's not great.

#### With .groupby() we can gain some insights as well:
Check the notebook for the tables with these data.

- Male and female who survived: Most female survived while most men died, good feature.
- Pclass and their survival: The number is kind of equal except for the 3 Pclass where more people died and a little for the 1 Pclass.
- SibSp and their survival: Not much difference of counts in the same values of SibSp, except for zero, where more people died.
- Parch and their survival: In Parch=0 more people died (2/3) for others Parch's there is not much difference.
- SibSp + Parch and their survival: In Parch=0 more people died (2/3) for others Parch's there is not much difference.

##

### 2. Now it's time for data preprocessing

- As seen in the NaN values we need to fill the values for Age. Check if age has a correlation with sex grouping by sex and age. It does not.
- Remembering the heatmap we see that the greatest correlation for Age is Pclass, 0.41. So we fill the missing values with the mean of Age for each Pclass. We do that by grouping Pclass and Age.
- We also normalize the fare column by the Z-score
- Family size could be a good feature instead of separated SibSp and Parch. Create a new column in the dataframe with the family size, SibSp + Parch.

##

### 3. Logistic Regression with TensorFlow and Sklearn

1. We need to create the # creates the feature layer. Divide the feature into it's classes:
- Numerical features: Fare
- Bucketized features: Age, FamilySize with intervals (10,2)
- Categorical features: Sex, Pclass

2. Create the feature columns and joins them in a Layer.

3. Create a model with Keras from TensorFlow. 
The model has the feature layer and in the second layer pass the regression value trough a sigmoid activation.

4. Train the model

5. Tune the hyperparameters

6. Define the metrics we will use to evaluate:
- accuracy
- precision
- recall
- AUC

By the end we have the result of training:
#### loss: 0.6274 - accuracy: 0.7899 - precision: 0.9429 - recall: 0.4825 - auc: 0.8028
![image](https://github.com/pemariano/TitanicML/assets/85647121/94fe247d-4775-4429-aba1-fe6e77ce4696)

Which seem to be good. Great precision and AUC. An Ok accuracy. 

We evaluate the model against the test set resulting in the predictions at `TitanicPredictionsLogisticTF.csv`


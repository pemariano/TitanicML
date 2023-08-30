# TitanicML
### Titanic predictions with Machine Learning Logistic Regression for classification with TensorFlow and Sklearn. Including data treatment.
Exploring the Kaggle Titanic Dataset.

##

The notebook starts with Exploratory Data Analysis (EDA) where we can visualize the data, found possible errors in it, missing values, look for hints for the better features, and more. 

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



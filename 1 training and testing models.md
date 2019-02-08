Loading data into pandas

```
import pandas
data = pandas.read_csv("file_name.csv")
```



# NumPy Arrays

Now that we've loaded the data in Pandas, we need to split the input and output into numpy arrays, in order to apply the classifiers in scikit learn. This is done in the following way: Say we have a pandas dataframe called `df`, like the following, with four columns labeled `A`, `B`, `C`, `D`:



![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/June/594c51ea_dataframe/dataframe.png)



If we want to extract column `A`, we do the following:

```
>> df['A']
    0    1
    1    5
    2    9
    Name: A, dtype: int64
```

Now, if we want to extract more columns, we just need to specify them, as follows:

```
>> df[['B', 'D']]
```

And the result is the following DataFrame:



![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/June/594c52f3_smalldf/smalldf.png)



And finally, we turn these pandas DataFrames into NumPy arrays. The command for turning a DataFrame `df` into a NumPy array is very simple:

```
>> numpy.array(df)
```

Now, try it yourself! Working with the same dataframe that we loaded in pandas previously, split it into the features `X`, and the labels `y`, and turn them into NumPy arrays.

***Note:** The capitalization may look strange, as X is capitalized whereas y is lowercase, but this is standard notation, as X represents a matrix of (maybe) several columns, and y a single column vector.*



Example:

import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")

// todo: Separate the features and the labels into arrays called X and y

X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])



# Training Models in scikit learn

In this section, we'll still be working with the dataset of the previous sections.



![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/June/594c6b79_points/points.png)



In the last section, we learned one of the most important classification algorithms in Machine Learning, including the following:

- Logistic Regression
- Neural Networks
- Decision Trees
- Support Vector Machines

Now, we'll have the chance to use them in real data! In sklearn, this is very easy, all we do is define our classifier, and then use the following line to fit the classifier to the data (which we call `X`, `y`):

```
classifier.fit(X,y)
```

Here are the main classifiers we define, together with the package we must import:

### Logistic Regression

```
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
```

### Neural Networks

(note: This is only available on versions 0.18 or higher of scikit-learn)

```
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
```

### Decision Trees

```
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
```

### Support Vector Machines

```
from sklearn.svm import SVC
classifier = SVC()
```

### Example: Logistic Regression

Let's do an end-to-end example on how to read data and train our classifier. Let's say we carry our X and y from the previous section. Then, the following commands will train the Logistic Regression classifier:

```
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X,y)
```

This gives us the following boundary:



![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/June/594c0430_linear-boundary/linear-boundary.png)



# Quiz: Train your own model

Now, it's your turn to shine! In the quiz below, we'll work with the following dataset:



![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/June/594c04d8_circle-data/circle-data.png)



Your goal is to use one of the classifiers above, between Logistic Regression, Decision Trees, or Support Vector Machines (sorry, Neural Networks are still not available in this version of sklearn, but we will be upgrading soon!), to see which one will fit the data better. Click on `Test Run` to see the graphical output of your classifier, and in the quiz underneath this, enter the classifier that you think fit the data better!

##Example:

import pandas
import numpy

// Read the data

data = pandas.read_csv('data.csv')

// Split the data into X and y

X = numpy.array(data[['x1', 'x2']])
y = numpy.array(data['y'])

// import statements for the classification algorithms

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

// TODO: Pick an algorithm from the list:

- Logistic Regression

- Decision Trees

- Support Vector Machines

Define a classifier (bonus: Specify some parameters!)

and use it to fit the data

Click on `Test Run` to see how your algorithm fit the data!

#classifier = SVC()
classifier = DecisionTreeClassifier()
classifier.fit(X,y)



# Tuning Parameters

So it looks like 2 out of 3 algorithms worked well last time, right? This are the graphs you probably got:



![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/June/594d5e2e_curves/curves.png)



It seems that Logistic Regression didn't do so well, as it's a linear algorithm. Decision Trees managed to bound the data well (question: Why does the area bounded by a decision tree look like that?), and the SVM also did pretty well. Now, let's try a slightly harder dataset, as follows:



![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/June/594d5ffe_eggsdata/eggsdata.png)



Let's try to fit this data with an SVM Classifier, as follows:

```
 >>> classifier = SVC()
 >>> classifier.fit(X,y)
```

If we do this, it will fail (you'll have the chance to try below). However, it seems that maybe we're not exploring all the power of an SVM Classifier. For starters, are we using the right kernel? We can use, for example, a polynomial kernel of degree 2, as follows:

```
>>> classifier = SVC(kernel = 'poly', degree = 2)
```

Let's try it ourselves, let's play with some of these parameters. We'll learn more about these later, but here are some values you can play with. (For now, we can use them as a black box, but they'll be discussed in detail during the **Supervised Learning** Section of this nanodegree.)

- **kernel** (string): 'linear', 'poly', 'rbf'.
- **degree** (integer): This is the degree of the polynomial kernel, if that's the kernel you picked (goes with poly kernel).
- **gamma** (float): The gamma parameter (goes with rbf kernel).
- **C** (float): The C parameter.

In the quiz below, you can play with these parameters. Try to tune them in such a way that they bound the desired area! In order to see the boundaries that your model created, click on `Test Run`.

***Note:** The quiz is not graded. But if you want to see a solution that works, look at the solutions.py tab. The point of this quiz is not to learn about the parameters, but to see that in general, it's not easy to tune them manually. Soon we'll learn some methods to tune them automatically in order to train better models.*



Example:



import pandas
import numpy

// Read the data

data = pandas.read_csv('data.csv')

// Split the data into X and y

X = numpy.array(data[['x1', 'x2']])
y = numpy.array(data['y'])

// Import the SVM Classifier

from sklearn.svm import SVC

// TODO: Define your classifier.

// Play with different values for these, from the options above.

// Hit 'Test Run' to see how the classifier fit your data.

// Once you can correctly classify all the points, hit 'Submit'. 

classifier = SVC(kernel = 'rbf', gamma = 200)

// Fit the classifier

classifier.fit(X,y)



# Regression vs Classification



Regression models predict a value, like: 4 - 3 or 6.7

Classification it's one that determines the state as + -, yes/no or cat/dog



You should never use testing data for training

example on sklearn



// Import statements 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

// 

from sklearn.cross_validation import train_test_split

http://scikit-learn.org/0.16/modules/generated/sklearn.cross_validation.train_test_split.html

// Read in the data.

data = np.asarray(pd.read_csv('data.csv', header=None))

// Assign the features to the variable X, and the labels to the variable y. 

X = data[:,0:2]
y = data[:,2]

// Use train test split to split your data 

// Use a test size of 25% and a random state of 42

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=42)

// Instantiate your decision tree model

model = DecisionTreeClassifier()

// TODO: Fit the model to the training data.

model.fit(X_train,y_train)

// TODO: Make predictions on the test data

y_pred = model.predict(X_test)

// TODO: Calculate the accuracy and assign it to the variable acc on the test data.

acc = accuracy_score(y_test, y_pred)



Splitting a dataset into training and testing data is very easy with sklearn. All we need is the command `train_test_split`. The function takes as inputs `X` and `y`, and returns four things:

- `X_train`: The training input
- `X_test`: The testing input
- `y_train`: The training labels
- `y_test`: The testing labels

The call to the function looks as follows:

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
```

The last parameter, `test_size` is the percentage of the points that we want to use as testing. In the above call, we are using 25% of our points for testing, and 75% for training.

Let's practice! We'll again use the dataset from the previous section:



![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/June/594bdd75_points/points.png)



In the following quiz, use the `train_test_split` function to split the dataset into training and testing sets. The size of the testing set must be **20%** of the total size of the data. Call your training sets `X_train` and `y_train`, and your testing sets `X_test` and `y_test`.

Click on `Test Run` to see a visualization of the results, where the training set will be drawn as circles, and the testing set as squares. Then when you're done, click on `Submit` to check your code!



##### Reading the csv file

import pandas as pd
data = pd.read_csv("data.csv")

##### Splitting the data into X and y
import numpy as np
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

##### Import statement for train_test_split
from sklearn.cross_validation import train_test_split

##### TODO: Use the train_test_split function to split the data into
##### training and testing sets.
##### The size of the testing set should be 20% of the total size of the data.
##### Your output should contain 4 objects.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)
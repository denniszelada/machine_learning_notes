# 3 Model Selection

Type of Errors

Trying to kill Godzilla with a fly killer

or killing a fly with a bazooka

![1](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/1.png)

![2](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/2.png)

So let's see an example with a classification problem, we want to identify Dogs

![3](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/3.png)

![4](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/4.png)

![5](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/5.png)

![6](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/6.png)

![7](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/7.png)

![8](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/8.png)

![9](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/9.png)

![10](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/10.png)

![11](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/11.png)

![12](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/12.png)

![13](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/13.png)

![14](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/14.png)

![15](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/15.png)

![16](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/16.png)

![17](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/17.png)

![18](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/18.png)

You can think of UNDERFIT as not studying for a test

Overfiting as instead of studying just memorizing the content and at the end being unable to respond to questions outside of the content that you memorize

Just Right is when you study and you are prepared to answer any question.

### K-FOLD CROSS VALIDATION

![19](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/19.png)

How do we not 'lose' the training data?

![20](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/20.png)

In this case we split our data in bucket in this case K is of size 4

with can do this with sklearn as 

```sklearn
	from sklearn.model_selection import KFold
	kf = KFold(12,3)
	
	for train_indices, test_indices in kf:
		print train_indices, test_indices
```

It's important to randomize the data to avoid any biases

![21](/Users/denniszelada/projects/notes/machine_learning_notes/model_selection/21.png)

in Sklearn we can do this by

```sklearn
from sklearn.model_selection import KFold
kf = KFold(12,3, shuffle = True)
for train_indices, test_indices in kf:
	print train_indices, test_indices
```
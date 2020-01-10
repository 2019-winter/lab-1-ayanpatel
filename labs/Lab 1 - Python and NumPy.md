---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Ayan Patel


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


**YOUR ANSWER HERE**


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python
import numpy as np
a = np.full((6,4), 2)
a
```

## Exercise 2

```python
b = np.full((6,4), 1)
np.fill_diagonal(b, 3)
b
```

## Exercise 3

```python
a*b #works element by element
#np.dot(a,b) #does not work because shapes are not aligned, column and row number must match for mat mult inner dim must be same
```

## Exercise 4

```python
c = np.dot(a.transpose(), b)
d = np.dot(a, b.transpose())
print (c)
print (d)
#one has a row count of 4 and the other of 6 - outer dim is size of product
```

## Exercise 5

```python
def myFunc():
    print ("hello")
    
myFunc()
```

## Exercise 6

```python
def randArr():
    a = np.random.rand(3,2)
    print (a)
    print (np.sum(a))
    print (np.mean(a))
    
randArr()
```

## Exercise 7

```python
def countOnes(x):
    count = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] == 1:
                count += 1
    print (count)
    print (len(np.where(x==1)[0]))
    
countOnes(np.full((2,2), 1))
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
import pandas as pd
df = pd.DataFrame(np.full((6,4), 2))
df
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
df2 = pd.DataFrame(np.ones((6,4)))
np.fill_diagonal(df2.to_numpy(), 3)
df2
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
df*df2
```

## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
def countdf_ones(dfin):
    print (len(np.where(dfin==1)[0]))
    
countdf_ones(df2)
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
titanic_df['name']
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
titanic_df.set_index('sex',inplace=True)
tf = titanic_df.loc['female']
l = len(titanic_df.loc['female'])
print (tf)
print ("female passenger count: ", l)
```

## Exercise 14
How do you reset the index?

```python
titanic_df.reset_index()
```

```python

```

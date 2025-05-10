---
layout: page
title:
---
[<< Back to Detector](detector.md)<br>
[<< Back to Pt 1](pt1-getting-the-data.md)<br>
&nbsp;&nbsp;&nbsp;[Pt 3 - The Fun Part >>](pt3-the-fun-part.md)

## Pt 2 - Training The Model

In this section we will...
- Load the data we transformed in Pt 1.
- Re-arrange the data into a sparse matrix
- Apply TF-IDF Transformation to the word counts
- Train the Naive Bayes model
- Save model objects to re-usable files

## Load Data
If you're starting from a brand new script like I am, you can manually set the file path where the sqlite database from Pt I is located on your computer.


```python
db_path = r"path/to/database.db"
```

The `model_data` table we created in Pt 1 contains about 19 million rows. It would be convenient to load the entire dataset into a `pandas.DataFrame` before converting it into a sparse matrix, but a table of that size would be extremely memory intensive. 

Instead of loading the entire dataset at once, we will use the `fetchmany()` method from the `sqlite3` package to load just 50,000 rows at a time into a `DataFrame`, then append that data into our sparse matrix. 

This code below should take 1-2 minutes.


```python
import sqlite3 as sql
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sps

# Connect to database
con = sql.connect(db_path)
cur = con.cursor()

# Set the number of rows for .fetchmany()
cur.arraysize = 50_000

data_query = """
    select word_count, track_index, word_index
    from model_data
    where document_frequency between 0.001 and 0.6
"""

# Start a "timer" so we can observe how long the entire process took.
t = time()

# Query the database
cur.execute(data_query)

# Fetch the first 50k rows
temp_result = cur.fetchmany()

# Assign the data into a small dataframe for convenient 
# transformation into a numpy array
coordinates = pd.DataFrame(
    temp_result, 
    columns=["word_count", "track_index", "word_index"]
).values.T

# Fetch the second set of rows
temp_result = cur.fetchmany()

# While there are still rows being pulled by .fetchmany()
# repeat the process, converting each set of rows to a
# dataframe, then an array, then appending the new array
# to the existing one.
while len(temp_result) > 0:
    coordinates = np.hstack(
        [coordinates,
        pd.DataFrame(
            temp_result, 
            columns=["word_count", "track_index", "word_index"]
        ).values.T]
    )
    temp_result = cur.fetchmany()

con.close()

# Once the array is complete, convert to a sparse matrix
# using scipy.csr_array()
bow_csr = sps.csr_array(
    (coordinates[0], (coordinates[1], coordinates[2])),
    dtype=np.int32
)

print(f"Data Retrieval Time: {time() - t:.0f} seconds")
print(f"Observations: {bow_csr.shape[0]}")
print(f"Features: {bow_csr.shape[1]}")
```

    Data Retrieval Time: 54 seconds
    Observations: 237662
    Features: 5000
    

Next, we pull indices showing us which songs are part of the test set (rather than the training set), and which we assigned as death metal. 

We also create a dictionary and blacklist for the words we're using as features, which will be useful later for evaluating novel data (i.e. a brand new, user supplied song) through the model.


```python
test_indices_query = """
    select distinct track_index
    from model_data
    where is_test = 1
"""

targ_indices_query = """
    select distinct track_index
    from model_data
    where term_match = 1
"""

dict_query = """
    select distinct word, word_index 
    from model_data
    where document_frequency between 0.001 and 0.6
"""

blacklist_query = """
    select distinct word 
    from model_data
    where document_frequency < 0.001 or document_frequency > 0.6
"""

con = sql.connect(db_path)

print("Getting Test set indices")
test_indices = pd.read_sql(test_indices_query, con=con).values.ravel()

print("Getting target indices ([death metal = TRUE])")
targ_indices = pd.read_sql(targ_indices_query, con=con).values.ravel()

print("Getting word dictionary")
dictionary = pd.read_sql(dict_query, con=con)

print("Getting word blacklist")
blacklist = pd.read_sql(blacklist_query, con=con)

con.close()

print("Complete")
```

    Getting Test set indices
    Getting target indices ([death metal = TRUE])
    Getting word dictionary
    Getting word blacklist
    Complete
    

Using the indices we just pulled, create `X_train`, `X_test`, `y_train`, and `y_test` sets, in the traditional `sklearn` format.


```python
train_indices = np.array([i for i in range(bow_csr.shape[0]) if i not in test_indices])

X_train = bow_csr[train_indices]
X_test = bow_csr[test_indices]

y = np.array([True if i in targ_indices else False for i in range(bow_csr.shape[0])])
y_train = y[train_indices]
y_test = y[test_indices]
```

How many songs were *not* identified as death metal?


```python
baseline = 1 - len(targ_indices)/bow_csr.shape[0]
print(f"Non-Death Metal: {baseline*100:.1f}%")
```

    Non-Death Metal: 95.9%
    

## TF-IDF Transformation
Rather than training on word counts, we will train on each word's Term Frequency Inverse Document Frequency, or TF-IDF score. A full explanation on what it is and how to calculate is available [here](https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/#), but the basic idea is: TF-IDF is a score of a word's *importance* to identifying a document. A word that's very common in one document but rare in general will get a high TF-IDF score.


```python
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()
tfidf.fit(X_train)

X_train_tfidf = tfidf.transform(X_train)
X_train_tfidf.indices = X_train_tfidf.indices.astype(np.int32, casting="same_kind")
X_train_tfidf.indptr = X_train_tfidf.indptr.astype(np.int32, casting="same_kind")

print("TF-IDF training complete")
```

    TF-IDF training complete
    

## Complement Naive Bayes

Now that we've completely prepared our data, including transforming word counts to TF-IDF importance scores. Since we have an unbalanced data set (only about 4% of the data set is labeled 'death metal'), we fit a *Complement* Naive Bayes model. To quote the documentation from scikit-learn,

> \[Complement Naive Bayes\] is an adaptation of the standard multinomial naive Bayes algorithm that is particularly suited for imbalanced data sets. Specifically, CNB uses statistics from the complement of each class to compute the model’s weights.
>
> [scikit-learn.org, 1.9.3. Complement Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#complement-naive-bayes)


```python
from sklearn.naive_bayes import ComplementNB
import sklearn.metrics as met

clf = ComplementNB()
clf.fit(X_train_tfidf, y_train)

# score it (and add to scores)
X_test_tfidf = tfidf.transform(X_test)

y_prob_tfidf = clf.predict_proba(X_test_tfidf).T[1]

print("Complement Naive Bayes training complete")
```

    Complement Naive Bayes training complete
    

Before the final part, let's save our work.  Some models require proprietary methods to be saved for later use, but `TfidfTransformer()` and `ComplementNB()` objects can be saved as `.pickle` files using python's built-in `pickle` library. This way they can be used to make predictions in new scripts, on new data, without having to be re-trained.


```python
import pickle

print("Saving ComplementNB() object")
with open("CNB.pickle", "wb") as f:
    pickle.dump(clf, f)

print("Saving TfIdfTransformer() object")
with open("TFIDF.pickle", "wb") as f:
    pickle.dump(tfidf, f)

print("Complete")
```

    Saving ComplementNB() object
    Saving TfIdfTransformer() object
    Complete
    

In addition, we will also use the `to_csv()` method from the `pandas` library to save the term dictionary and blacklist.


```python
print("Saving dictionary.csv")
dictionary.to_csv("dictionary.csv", index=False)

print("Saving blacklist.csv")
blacklist.to_csv("blacklist.csv", index=False)

print("Complete")
```

    Saving dictionary.csv
    Saving blacklist.csv
    Complete
    

See you in the final, and in my opinion the most fun part! 

[<< Back to Detector](detector.md)<br>
[<< Back to Pt 1](pt1-getting-the-data.md)<br>
&nbsp;&nbsp;&nbsp;[Pt 3 - The Fun Part >>](pt3-the-fun-part.md)

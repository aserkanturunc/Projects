#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize Otter
import otter
grader = otter.Notebook("hw10.ipynb")


# # Homework 10: Spam/Ham Classification - Build Your Own Model
# ## Feature Engineering, Logistic Regression, Cross Validation
# ## Due Date: Thursday 11/22, 11:59 PM PST
# 
# **Collaboration Policy**
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: *list collaborators here*

# ## This Assignment
# In this homework, you will be building and improving on the concepts and functions that you implemented in homework 9 to create your own classifier to distinguish spam (junk or commercial or bulk) emails from ham (non-spam) emails. We will evaluate your work based on your model's accuracy and your written responses in this notebook.
# 
# After this assignment, you should feel comfortable with the following:
# 
# - Using `sklearn` libraries to process data and fit models
# - Validating the performance of your model and minimizing overfitting
# - Generating and analyzing precision-recall curves
# 
# ## Warning
# This is a **real world** dataset– the emails you are trying to classify are actual spam and legitimate emails. As a result, some of the spam emails may be in poor taste or be considered inappropriate. We think the benefit of working with realistic data outweighs these innapropriate emails, and wanted to give a warning at the beginning of the project so that you are made aware.

# In[2]:


# Run this cell to suppress all FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ## Score Breakdown
# Question | Points
# --- | ---
# 1 | 6
# 2 | 6
# 3 | 3
# 4 | 15
# Total | 30

# ## Setup and Recap
# 
# Here we will provide a summary of the homework 9 to remind you of how we cleaned the data, explored it, and implemented methods that are going to be useful for building your own model.

# In[3]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)


# ### Loading and Cleaning Data
# 
# Remember that in email classification, our goal is to classify emails as spam or not spam (referred to as "ham") using features generated from the text in the email. 
# 
# The dataset consists of email messages and their labels (0 for ham, 1 for spam). Your labeled training dataset contains 8348 labeled examples, and the unlabeled test set contains 1000 unlabeled examples.
# 
# Run the following cell to load in the data into DataFrames.
# 
# The `train` DataFrame contains labeled data that you will use to train your model. It contains four columns:
# 
# 1. `id`: An identifier for the training example
# 1. `subject`: The subject of the email
# 1. `email`: The text of the email
# 1. `spam`: 1 if the email is spam, 0 if the email is ham (not spam)
# 
# The `test` DataFrame contains 1000 unlabeled emails. You will predict labels for these emails and submit your predictions to the autograder for evaluation.

# In[4]:


from utils import fetch_and_cache_gdrive
fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'test.csv')

original_training_data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()



# In[5]:


#Training/Validation Split
#Recall that the training data we downloaded is all the data we have available for both 
#training models and validating the models that we train. We therefore 
#split the training data into separate training and validation datsets.
#You will need this validation data to assess 
#the performance of your classifier once you are finished training.
#As in homework 9, we set the seed (random_state) to 42. Do not modify this in the following questions, as our tests depend on this random seed.


# In[6]:


# Fill any missing or NAN values
print('Before imputation:')
print(original_training_data.isnull().sum())
original_training_data = original_training_data.fillna('')
print('------------')
print('After imputation:')
print(original_training_data.isnull().sum())


# ### Training/Validation Split
# 
# Recall that the training data we downloaded is all the data we have available for both training models and **validating** the models that we train. We therefore split the training data into separate training and validation datsets. You will need this **validation data** to assess the performance of your classifier once you are finished training. 
# 
# As in homework 9, we set the seed (random_state) to 42. **Do not modify this in the following questions, as our tests depend on this random seed.**

# In[7]:


# This creates a 90/10 train-validation split on our labeled data
from sklearn.model_selection import train_test_split
train, val = train_test_split(original_training_data, test_size=0.1, random_state=42)

# We must do this in order to preserve the ordering of emails to labels for words_in_texts
train = train.reset_index(drop=True)


# In[8]:


np.mean(train.loc[train['spam']==1,'subject'].str.contains('!'))# compare spam and ham emails w respect to if they have


# In[9]:


np.mean(train.loc[train['spam']==0,'subject'].str.contains('!')) # Spam has a lot more ! in the subject portion


# In[10]:


spam_emails=train.loc[train['spam']==1,'email']


# In[11]:


concat_emails=[]
def get_all_emails(series):
    for email in series:
        concat_emails.append(email)
get_all_emails(spam_emails)
joined=''.join(concat_emails)
csv_words=joined.split(' ')
word_and_count={}
def count_each_word(comma_sep_text):
    for word in comma_sep_text:
        if(len(word)>5 and word in word_and_count.keys()):
            word_and_count[word]=word_and_count[word]+1
        else:
            word_and_count[word]=1
word_and_count
def getMax_100_keys(dict):
    sorted=np.sort(list(dict.values()))
    length=len(sorted)
    max_vals=sorted[length-100:]
    max_keys=[]
    for key in dict.keys():
        if dict[key]>=min(max_vals):
            max_keys.append(key)
    return max_keys
count_each_word(csv_words)
_100spams=getMax_100_keys(word_and_count)
# I pick "business","please","e-mail","receive","information",
#"insurance","credit","orders","government","financial"as my
#spam identifying words 
spam_words=["business","e-mail","receive","information", "please","insurance","credit","orders","government","financial",
            "company","grants","number","within","online","simply","color:","<center>\n","report",]

ham_emails=original_training_data.loc[original_training_data['spam']==0,'email']
#Now doing the same for ham and pick 5 words there too

concat_emails_ham=[]
def get_all_emails_ham(series):
    for email in series:
        concat_emails_ham.append(email)
get_all_emails_ham(ham_emails)
joined_ham=''.join(concat_emails_ham)
csv_words_ham=joined_ham.split(' ')
word_and_count_ham={}
def count_each_word_ham(comma_sep_text):
    for word in comma_sep_text:
        if(len(word)>5 and word in word_and_count_ham.keys()):
            word_and_count_ham[word]=word_and_count_ham[word]+1
        else:
            word_and_count_ham[word]=1
def getMax_100_keys_ham(dict):
    sorted=np.sort(list(dict.values()))
    length=len(sorted)
    max_vals_ham=sorted[length-100:]
    max_keys_ham=[]
    for key in dict.keys():
        if dict[key]>=min(max_vals_ham):
            max_keys_ham.append(key)
    return max_keys_ham
count_each_word_ham(csv_words_ham)
_100hams=getMax_100_keys_ham(word_and_count_ham)
 # I pick really, because, software, mailing, message, between,
#better, support, that's, doesn't as the ham identifying words 
#as my ham identifying words
ham_words=["really","software","because","mailing", "message","better","support", "that's", "doesn't","between","would"
           ,"while","since","after","their","think"]
# making sure I did not pick words that are shared between the two top 100 words lists
print("ham: ")
print( _100hams)
print("spam: ")
print(_100spams)
for ham in ham_words:
    if ham in _100spams:
        ham_words.remove(ham)
for spam in spam_words:
    if spam in _100hams:
        spam_words.remove(spam)
print("spam words:",spam_words)
print("ham words:",ham_words)
len_greater_than_4_ham={}
def count_each_word_ham_len4(comma_sep_text):
    for word in comma_sep_text:
        if(len(word)>4 and word in len_greater_than_4_ham.keys()):
            len_greater_than_4_ham[word]=len_greater_than_4_ham[word]+1
        else:
            len_greater_than_4_ham[word]=1
count_each_word_ham_len4(csv_words_ham)
def getMax_100_keys_ham_len4(dict):
    sorted=np.sort(list(dict.values()))
    length=len(sorted)
    max_vals_ham_len4=sorted[length-100:]
    max_keys_ham_len4=[]
    for key in dict.keys():
        if dict[key]>=min(max_vals_ham_len4):
            max_keys_ham_len4.append(key)
    return max_keys_ham_len4
getMax_100_keys_ham_len4(len_greater_than_4_ham)


# In[13]:


train_exclam=train
train_exclam['has_money']=train['subject'].str.contains('money')
np.mean(train_exclam['has_money'])


# In[14]:


train_exclam=train
train_exclam['subject contains !']=train['subject'].str.contains('!')
np.mean(train_exclam['subject contains !'])


# In[15]:


train_exclam['email contains !']=train['email'].str.contains('!')
np.mean(train_exclam['email contains !'])


# In[ ]:


#each_letter_ham=joined_ham.split(' ')
#num_capital=0
#total_letters=0
#for word in each_letter_ham:
    #for letter in word:
       # total_letters=total_letters+1


# In[16]:


train_exclam['subject contains dollar']=train['subject'].str.contains('dollar')
#np.mean(train_exclam['email contains ?'])
np.mean(train_exclam['subject contains dollar'])


# In[17]:


train_exclam['subject contains sex']=train['subject'].str.contains('sex')
#np.mean(train_exclam['email contains ?'])
np.mean(train_exclam['subject contains sex'])


# ### Feature Engineering
# 
# In order to train a logistic regression model, we need a numeric feature matrix $X$ and a vector of corresponding binary labels $y$. To address this, in homework 9, we implemented the function `words_in_texts`, which creates numeric features derived from the email text and uses those features for logistic regression. 
# 
# For this homework, we have provided you with an implemented version of `words_in_texts`. Remember that the function outputs a 2-dimensional NumPy array containing one row for each email text. The row should contain either a 0 or a 1 for each word in the list: 0 if the word doesn't appear in the text and 1 if the word does. 

# In[18]:


def words_in_texts(words, texts):
    '''
    Args:
        words (list): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    import numpy as np
    indicator_array = 1 * np.array([texts.str.contains(word) for word in words]).T
    return indicator_array


# Run the following cell to see how the function works on some dummy text.

# In[19]:


words_in_texts(['hello', 'bye', 'world'], pd.Series(['hello', 'hello worldhello']))


# ### EDA and Basic Classification
# 
# In homework 9, we proceeded to visualize the frequency of different words for both spam and ham emails, and used `words_in_texts(words, train['email'])` to directly to train a classifier. We also provided a simple set of 5 words that might be useful as features to distinguish spam/ham emails. 
# 
# We then built a model using the using the [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier from `scikit-learn`.
# 
# Run the following cell to see the performance of a simple model using these words and the `train` dataframe.

# In[20]:


some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words,train['email'])# I should add the 0 0 0 1s from the contains(!) column 
#to this matrix
Y_train = train['spam']

X_train[:5], Y_train[:5]


# In[21]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,Y_train)
yhat=model.predict(X_train)
training_accuracy = np.mean(yhat==Y_train)
print("Training Accuracy: ", training_accuracy)


# In[22]:


identifying_words=spam_words+ham_words
My_X_train = words_in_texts(identifying_words,train['email'])
array_exclam=np.array(train_exclam['subject contains !'])
array_exclam=np.array([array_exclam]).reshape(7513,1) #reshaping so that I can append it in the below cell

array_exclam2=np.array(train_exclam['email contains !'])
array_exclam2=np.array([array_exclam2]).reshape(7513,1)


# In[23]:


array_money=np.array([train_exclam['has_money']])
array_money=array_money.reshape(7513,1)


# In[24]:


array_R=np.array([train_exclam['subject contains sex']])
array_R=array_R.reshape(7513,1)


# In[25]:


My_X_train = np.append(My_X_train,array_exclam2,1)


# In[26]:


My_X_train = np.append(My_X_train,array_exclam,1)


# In[27]:


My_X_train=np.append(My_X_train,array_money,1)


# In[ ]:


#My_X_train=np.append(My_X_train,array_R,1)


# In[28]:


My_X_train.shape


# In[29]:


#fitting the modeling

my_model = LogisticRegression()
my_model.fit(My_X_train,Y_train)
my_yhat=my_model.predict(My_X_train)
print("coeffs: ", my_model.coef_)
my_training_accuracy = np.mean(my_yhat==Y_train)
print("Training Accuracy: ", my_training_accuracy)


# In[ ]:


val


# In[30]:


val['email contains !']=val['email'].str.contains('!')
val['subject contains !']=val['subject'].str.contains('!')
val['has_money']=val['subject'].str.contains('money')


# In[31]:


array_A=np.array(val['subject contains !'])
array_A=np.array([array_A]).reshape(835,1) #reshaping so that I can append it in the below cell
array_B=np.array(val['email contains !'])
array_B=np.array([array_B]).reshape(835,1)
array_C=np.array([val['has_money']])
array_C=array_C.reshape(835,1)


# In[32]:


identifying_words=spam_words+ham_words
My_X_val = words_in_texts(identifying_words,val['email'])
My_X_val = np.append(My_X_val,array_A,1)
My_X_val = np.append(My_X_val,array_B,1)
My_X_val = np.append(My_X_val,array_C,1)
Y_val=val['spam']
my_yhat_val=my_model.predict(My_X_val)
my_val_training_accuracy = np.mean(my_yhat_val==Y_val)
print("Training Accuracy: ", my_val_training_accuracy)


# 

# ### Evaluating Classifiers

# In our models, we are evaluating accuracy on the training set, which may provide a misleading accuracy measure. In homework 9, we calculated various metrics to lead us to consider more ways of evaluating a classifier, in addition to overall accuracy. Below is a reference to those concepts.
# 
# Presumably, our classifier will be used for **filtering**, i.e. preventing messages labeled `spam` from reaching someone's inbox. There are two kinds of errors we can make:
# - False positive (FP): a ham email gets flagged as spam and filtered out of the inbox.
# - False negative (FN): a spam email gets mislabeled as ham and ends up in the inbox.
# 
# To be clear, we label spam emails as 1 and ham emails as 0. These definitions depend both on the true labels and the predicted labels. False positives and false negatives may be of differing importance, leading us to consider more ways of evaluating a classifier, in addition to overall accuracy:
# 
# **Precision** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FP}}$ of emails flagged as spam that are actually spam.
# 
# **Recall** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FN}}$ of spam emails that were correctly flagged as spam. 
# 
# **False-alarm rate** measures the proportion $\frac{\text{FP}}{\text{FP} + \text{TN}}$ of ham emails that were incorrectly flagged as spam. 
# 
# The two graphics below may help you understand precision and recall visually:
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/700px-Precisionrecall.svg.png" width="500px">
# 
# Note that a true positive (TP) is a spam email that is classified as spam, and a true negative (TN) is a ham email that is classified as ham.

# # Moving Forward - Building Your Own Model
# 
# With this in mind, it is now your task to make the spam filter more accurate. In order to get full credit on the accuracy part of this assignment, you must get at least **88%** accuracy on the test set. To see your accuracy on the test set, you will use your classifier to predict every email in the `test` DataFrame and upload your predictions to Gradescope.
# 
# **Gradescope limits you to four submissions per day**. This means you should start early so you have time if needed to refine your model. You will be able to see your accuracy on 70% of the test set when submitting to Gradescope, but we will be evaluating your model on the entire test set so try to score slightly above 88% on gradescope if you can.
# 
# Here are some ideas for improving your model:
# 
# 1. Finding better features based on the email text. Some example features are:
#     1. Number of characters in the subject / body
#     1. Number of words in the subject / body
#     1. Use of punctuation (e.g., how many '!'s were there?)
#     1. Number / percentage of capital letters 
#     1. Whether the email is a reply to an earlier email or a forwarded email
# 1. Finding better (and/or more) words to use as features. Which words are the best at distinguishing emails? This requires digging into the email text itself. 
# 1. Better data processing. For example, many emails contain HTML as well as text. You can consider extracting out the text from the HTML to help you find better words. Or, you can match HTML tags themselves, or even some combination of the two.
# 1. Model selection. You can adjust parameters of your model (e.g. the regularization parameter) to achieve higher accuracy. Recall that you should use cross-validation to do feature and model selection properly! Otherwise, you will likely overfit to your training data.
# 
# You may use whatever method you prefer in order to create features, but **you are not allowed to import any external feature extraction libraries**. In addition, **you are only allowed to train logistic regression models**. No random forests, k-nearest-neighbors, neural nets, etc.
# 
# We have not provided any code to do this, so feel free to create as many cells as you need in order to tackle this task. However, answering questions 7, 8, and 9 should help guide you.
# 
# ---
# 
# **Note:** *You may want to use your **validation data** to evaluate your model and get a better sense of how it will perform on the test set.* Note, however, that you may overfit to your validation set if you try to optimize your validation accuracy too much.
# 
# ---

# <!-- BEGIN QUESTION -->
# 
# ### Question 1: Feature/Model Selection Process
# 
# In this following cell, describe the process of improving your model. You should use at least 2-3 sentences each to address the follow questions:
# 
# 1. How did you find better features for your model?
# 2. What did you try that worked or didn't work?
# 3. What was surprising in your search for good features?
# 
# <!--
# BEGIN QUESTION
# name: q1
# manual: True
# points: 6
# -->

# 1)I found the most common words that appeared in the spam and ham emails. After that I used the words in text function to create the matrix that has a row for each email and column for each word.
# 2)I found that spam emails tend to have a higher chance to have '!' in their subject so I added this to my model, which improved the accuracy.
# 3)I found that words such as their, you're, should, through, words that are typically used in dail conversations appeared more in ham emails, which is surprising at first but also intuitive as a lot of the ham emails are personal.

# <!-- END QUESTION -->
# 
# 
# 
# ### Question 2: EDA
# 
# In the cell below, show a visualization that you used to select features for your model. 
# 
# Include:
# 
# 1. A plot showing something meaningful about the data that helped you during feature selection, model selection, or both.
# 2. Two or three sentences describing what you plotted and its implications with respect to your features.
# 
# Feel free to create as many plots as you want in your process of feature selection, but select only one for the response cell below.
# 
# **You should not just produce an identical visualization to question 3.** Specifically, don't show us a bar chart of proportions, or a one-dimensional class-conditional density plot. Any other plot is acceptable, **as long as it comes with thoughtful commentary.** Here are some ideas:
# 
# 1. Consider the correlation between multiple features (look up correlation plots and `sns.heatmap`). 
# 1. Try to show redundancy in a group of features (e.g. `body` and `html` might co-occur relatively frequently, or you might be able to design a feature that captures all html tags and compare it to these). 
# 1. Visualize which words have high or low values for some useful statistic.
# 1. Visually depict whether spam emails tend to be wordier (in some sense) than ham emails.

# <!-- BEGIN QUESTION -->
# 
# Generate your visualization in the cell below and provide your description in a comment.
# 
# <!--
# BEGIN QUESTION
# name: q2
# manual: True
# format: image
# points: 6
# -->

# In[33]:


# Write your description (2-3 sentences) as a comment here:
# 
#
#

# Write the code to generate your visualization here:
spam_word_in_spam=[]
for word in spam_words:
    spam_word_in_spam.append(word_and_count[word])
spam_word_in_ham=[]
for word in spam_words:
    spam_word_in_ham.append(word_and_count_ham[word])
ham_word_in_spam=[]
for word in ham_words:
    ham_word_in_spam.append(word_and_count[word])
ham_word_in_ham=[]
for word in ham_words:
    ham_word_in_ham.append(word_and_count_ham[word])
    
fig, ax = plt.subplots(2)

ax[0].bar(spam_words[:10], spam_word_in_ham[:10])
ax[1].bar(spam_words[:10], spam_word_in_spam[:10])
#In the first graph the orange bars represent the spam words that appear in the spam emails
# And the blue bars represents the spam words that appear in ham emails
# We see that the orange bars are significantly taller, showing that the words I identified as
#spam words appear a lot more often in the spam emails

ax[0].bar(ham_words[:10], ham_word_in_ham[:10])
ax[1].bar(ham_words[:10], ham_word_in_spam[:10])
#In the first graph the orange bars represent the ham words that appear in the spam emails
# And the blue bars represents the ham words that appear in ham emails
# We see that the blue bars are significantly taller, showing that the words I identified as
# ham words appear a lot more often in the ham emails


# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# ### Question 3: ROC Curve
# 
# In most cases we won't be able to get 0 false positives and 0 false negatives, so we have to compromise. For example, in the case of cancer screenings, false negatives are comparatively worse than false positives — a false negative means that a patient might not discover that they have cancer until it's too late, whereas a patient can just receive another screening for a false positive.
# 
# Recall that logistic regression calculates the probability that an example belongs to a certain class. Then, to classify an example we say that an email is spam if our classifier gives it $\ge 0.5$ probability of being spam. However, *we can adjust that cutoff*: we can say that an email is spam only if our classifier gives it $\ge 0.7$ probability of being spam, for example. This is how we can trade off false positives and false negatives.
# 
# The ROC curve shows this trade off for each possible cutoff probability. In the cell below, plot a ROC curve for your final classifier (the one you use to make predictions for Gradescope) on the training data. Refer to Lecture 20 or [Section 23.7](http://www.textbook.ds100.org/ch/23/classification_sensitivity_specificity.html?highlight=roc#roc-curves) of the course text to see how to plot an ROC curve.
# 
# <!--
# BEGIN QUESTION
# name: q3
# manual: True
# points: 3
# -->

# In[ ]:


from sklearn.metrics import roc_curve

# Note that you'll want to use the .predict_proba(...) method for your classifier
# instead of .predict(...) so you get probabilities, not classes

...


# <!-- END QUESTION -->
# 
# # Question 4: Test Predictions
# 
# The following code will write your predictions on the test dataset to a CSV file. **You will need to submit this file to the "Homework 10 Test Predictions" assignment on Gradescope to get credit for this question.**
# 
# Save your predictions in a 1-dimensional array called `test_predictions`. **Please make sure you've saved your predictions to `test_predictions` as this is how part of your score for this question will be determined.**
# 
# Remember that if you've performed transformations or featurization on the training data, you must also perform the same transformations on the test data in order to make predictions. For example, if you've created features for the words "drug" and "money" on the training data, you must also extract the same features in order to use scikit-learn's `.predict(...)` method.
# 
# **Note: You may submit up to 4 times a day. If you have submitted 4 times on a day, you will need to wait until the next day for more submissions.**
# 
# Note that this question is graded on an absolute scale based on the accuracy your model achieves on the overall test set, and as such, your score does not depend on your ranking on Gradescope. Your public Gradescope results are based off of your classifier's accuracy on 70% of the test dataset and your score for this question will be based off of your classifier's accuracy on 100% of the test set.
# 
# *The provided tests check that your predictions are in the correct format, but you must additionally submit to Gradescope to evaluate your classifier accuracy.*
# 
# <!--
# BEGIN QUESTION
# name: q4
# points: 3
# -->

# In[34]:


print('Before imputation:')
print(test.isnull().sum())
test = test.fillna('')
print('------------')
print('After imputation:')
print(test.isnull().sum())


# In[35]:


test['email contains !']=test['email'].str.contains('!')
test['subject contains !']=test['subject'].str.contains('!')
test['has_money']=test['subject'].str.contains('money')
array_AA=np.array(test['subject contains !'])
array_AA=np.array([array_AA]).reshape(1000,1) #reshaping so that I can append it in the below cell
array_BB=np.array(test['email contains !'])
array_BB=np.array([array_BB]).reshape(1000,1)
array_CC=np.array([test['has_money']])
array_CC=array_CC.reshape(1000,1)
identifying_words=spam_words+ham_words
My_X_test = words_in_texts(identifying_words,test['email'])
My_X_test = np.append(My_X_test,array_AA,1)
My_X_test = np.append(My_X_test,array_BB,1)
My_X_test = np.append(My_X_test,array_CC,1)
my_model.predict(My_X_test)


# In[47]:


test_predictions = my_model.predict(My_X_test)


# In[37]:


grader.check("q4")


# The following cell generates a CSV file with your predictions. **You must submit this CSV file to the "Homework 10 Test Predictions" assignment on Gradescope to get credit for this question.**

# In[50]:


from datetime import datetime

# Assuming that your predictions on the test set are stored in a 1-dimensional array called
# test_predictions. Feel free to modify this cell as long you create a CSV in the right format.

# Construct and save the submission:
submission_df = pd.DataFrame({
    "Id": test['id'], 
    "Class": test_predictions,
}, columns=['Id', 'Class'])
timestamp = datetime.isoformat(datetime.now()).split(".")[0]
submission_df.to_csv("submission_{}.csv".format(timestamp), index=False)

print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
print('You may now upload this CSV file to Gradescope for scoring.')


# ## Congratulations! You have completed homework 10!

# ---
# 
# To double-check your work, the cell below will rerun all of the autograder tests.

# In[43]:


grader.check_all()


# ## Submission
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**

# In[46]:


# Save your notebook first, then run this cell to export your submission.
grader.export()


#  

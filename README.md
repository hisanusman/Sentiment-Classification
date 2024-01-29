# Sentiment-Classification
The dataset consists of two types of data. i.e., images and text.

Sentiment Classification: Given an Internet meme, the first task is to classify it as a positive, very positive, negative, very negative or neutral meme.

• Training 6 classifiers (3 for images and 3 for text) on the provided dataset using sklearn library. Majority voting is done on the basis of labels returned by classifiers. <br>
• Displaying the confusion matrix, accuracy, recall, precision and F1-meausre for the 6 classifiers that have been trained. 
• Flask is used to connect the jupyter notebook with the HTML front end page so that a user can upload their own image (meme) and get the classification of that image. (Flask is a Python web framework that makes it easy to create a fully featured web application. It is quite simple, easy to learn and is a very demanding skill in 2022). 
• Creating a web application and deploying my project on it, basically interacting on the web application for input and output. A user will be giving the test case on the website e.g., giving some meme and the model has to tell that whether it is positive or negative.

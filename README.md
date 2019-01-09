# Data-Agnostic Classifier for Dummies

This is a ML application used to classify any text dataset into different labels. The app uses many text classifiers built into it and chooses the best classifier with the highest accuracy to train the models to predict the labels for the unlabelled data(user case).

## Getting started
Clone the whole repository with the html templates as it is and run it with any python IDE

## Prerequisites
Follow this guide to install required libraries for running this application.

### Windows:
Download Microsoft visual studio build tools, this would be required to install libraries needed.
[Download link](http://download.microsoft.com/download/5/F/7/5F7ACAEB-8363-451F-9425-68A90F98B238/visualcppbuildtools_full.exe)

Next install all your libraries by running cmd with administrator rights and running the following command in that folder:

```
pip install -r requirements.txt 
```

Install spacy language lib by executing this command in adminstrator mode

```
python -m spacy download en
```

To complete installing nltk execute these commands one by one in adminstrator mode

```
python
```
```
import nltk
```
```
nltk.download()
```

A window will pop up, download and install all the files and close it

### Linux:
Run the following commands to get your linux system to run our program:

```
sudo python3 -m pip install -r requirements.txt
python3 -m spacy download en
```

## How to use?
1. Open a terminal in the local repository and run the fit_dataset.py program.

	- For Windows:
	```
	python fit_dataset.py
	```
	- For Linux:
	```
	python3 fit_dataset.py
	```
	
2. After the program is successfully running, go to ```http://localhost:5000/``` in your preferred browser.

### Sample Run

Let us consider a sample dataset and use it in our application. The data set consists of Consumer Finance Complaints with 4 predefined categories namely - ‘Credit reporting’,  ‘Debt collection’, ‘Mortgage’, ‘Student loan’.
(Source: [data.gov](https://catalog.data.gov/dataset/consumer-complaint-database))

The input data to our application should contain a feature name and label fields.
The sameple dataset looks like the following:

||Consumer_complaint_narrative|Product|
|---|:---|---:|
|0|This company refuses to provide me verification and validation of debt per ...|Debt collection|
|1|Started the refinance of home mortgage process with cash out option on XX/ ...|Mortgage|
|2|I was dropped from my income based repayment plan by FedLoan servicing for ...|Student loan|
|3|The first communication that I received from the debt collector was a court ...|Debt collection|
|...|...|...|

Now assuming the application is running, let us proceed:

***Step 1:*** Upload labelled dataset in **.csv** format to train the model. Enter the feature name and label (Here, "Consumer_complaint_narrative" and "Product" respectively). Enter the name by which your trained model should be saved. Finally, Click the ```Train``` button.

![Screenshot1.png](https://user-images.githubusercontent.com/23413000/50923779-d3e05600-1473-11e9-8def-ec472a9f8e92.png)
The loading screen appears till the models are trained.

***Step 2:*** The accuracy table of the classifiers. The classifier with best accuracy is automatically chosen and is saved as a model with the name given in previous step.

![Screenshot2.png](https://user-images.githubusercontent.com/23413000/50923889-068a4e80-1474-11e9-86fe-aaca89d4c455.png)

***Step 3:*** Upload unlabelled dataset in **.csv** format to predict the labels. Enter the feature name in the unlabelled file whose label have to be predicted. Next, input the filename of the saved model. Then enter the filename of file in which results are going to be saved. Click OK. 

![Screenshot3.png](https://user-images.githubusercontent.com/23413000/50923891-08541200-1474-11e9-8cc6-d7d8c3118615.png)

***Step 4:*** The results are displayed in the following format:

![Screenshot4.png](https://user-images.githubusercontent.com/23413000/50923892-09853f00-1474-11e9-9246-8edf553f392e.png)

(***Note:*** If you have already trained your dataset and don't want to train it again, you can jump directly to step 3 by clicking "Go to Predict Page" in step 1)

## Built With
- [Flask](http://flask.pocoo.org/) - A web microframework for Python
- [scikit-learn](http://scikit-learn.org/stable/index.html) - Machine learning library in Python
- [spaCy](https://spacy.io/) - Industrial-Strength Natural Language Processing
- [NLTK](https://www.nltk.org/) - NLP tasks
- [pandas](https://pandas.pydata.org/) data structures and data analysis tools for the Python

## Authors
* **Allu Praveen**
* **Deepak Divya Tejaswi**
* **Nimit Kanani**
* **Nitish Kumar Naineni**

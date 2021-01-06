# Emotional-Based-Song-Retrival

This is the reposirtory for a emotional based song retrival system using LSTM and LSI to rank songs for a user input

# Getting Started
The instruction below will enable you to download this project locally and run this app in your computer so you can test or play with it.

## Prerequiste
Python3

### Installing 
Please follow the instruction to set up prerequsite 
1. Clone the Repository to your computer

2. Set up the virtual environment by doing the following:

- Create a new virtual environment:

'''virtualenv myenv'''

- Activate the virtual environment

If you are in MAC, run '''source myenv/bin/activate'''

If you are in Windows, run '''activate.bat'''

- Verify that there are no modules installed by pip, and then do a pip install from requirements.txt. You should see the following list of modules:
(myenv) $ pip freeze
(myenv) $ pip install -r requirements.txt
(myenv) $ pip freeze

click==7.1.2
gensim==3.8.3
joblib==1.0.0
langdetect==1.0.8
nltk==3.5
numpy==1.19.5
pandas==1.2.0
python-dateutil==2.8.1
pytz==2020.5
rank-bm25==0.2.1
regex==2020.11.13
scipy==1.6.0
six==1.15.0
smart-open==4.1.0
tqdm==4.55.1

To run this model, simple run
``` app.py ``` in Windows Terminal
``` python app.py ``` in Mac Terminal

## Loom Video Link:
https://www.loom.com/share/043efb844bc746bda2e3a8cfea2cda79?sid=8f53b520-b6d3-4ca0-b539-86a652754e21

## Project Overview:-
This project provides a containerized API for filtering Application-to-Person (A2P) SMS messages. 
It uses a machine learning model - Multinomial Naive Bayes,  to accurately filters A2P SMS, while whitelisting ensures that trusted messages are always delivered. 
The entire application is packaged within a single Docker container, for ease of deployment and use.

-> app.py serves as the main entry point of the project.

-> The data/ directory stores raw and processed datasets.

-> dataCleanup.py includes preprocessing utilities such as cleaning, transforming, and preparing data for model training or inference.

-> The model/ directory contains machine learning model.

-> whitelisting.py manages whitelisting logic.

-> Application logs are stored in app.log for performance monitoring.

-> All dependencies are listed in requirements.txt.

-> A virtual environment (venv/) is recommended for local development.

## Instructions to train and run the model:-
1.Clone the repository:
git clone https://github.com/lavenya0204/A2P-SMS-Spam-Filter.git
cd A2P-SMS-Spam-Filter

2.Create and activate a virtual environment (optional):
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3.Install dependencies:
pip install -r requirements.txt

4. Clean the dataset and train the model:
python dataCleanup.py

5. Run the Application:
uvicorn app:app --host 0.0.0.0 --port 8000
and follow the link
6. Read the Docs
http://localhost:8000/docs

-> Then try out the Example API Usage


## How To Add whitelist entries?
-> Go to config.json file 

-> Add the whitelist domain or phases you wish to add, to trusted_domains or trusted_phrases list

## Example API Usage:-
#Using Postman:

-> Select POST method and enter this url - http://127.0.0.1:8000/check_sms

-> In Body section , select raw and JSON

-> Give Input as follows with any message you want:

{
    "message": "Your OTP is 294647. Do not share it with anyone."
}

-> Click send and you will recieve the output . For the above input , the Output is:

{
    "verdict": "allowed",
    "reason": "whitelisted"
}
Because the message starts with a whitelisted phrase.

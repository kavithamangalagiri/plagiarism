import pickle 
import re
import nltk
from pathlib import Path
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.classify.scikitlearn import SklearnClassifier

__version__ = "0.1.0"

Base_Dir = Path(__file__).resolve(strict=True).parent

with open(f"{Base_Dir}/student_auth.pkl","rb") as f:
    model = pickle.load(f)

stop_words = set(stopwords.words("english"))
punctuations = set(string.punctuation)

# Define a function to preprocess the text data
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    # Remove stop words and punctuation marks
    tokens = [token for token in tokens if token not in stop_words and token not in punctuations]
    # Compute the frequency distribution of the tokens
    freq_dist = FreqDist(tokens)
    # Return the 100 most common tokens
    return " ".join([token for token, freq in freq_dist.most_common(100)])

def predict_pipeline(text):
    p_text = preprocess_text(text)
    pred = model.predict_proba([p_text])
    return pred #{"The predicted author is " : str(pred)}


#print(predict_pipeline("But remember that owing to my 'BS busting' activities, I have had the fortune (or misfortune) to become an expert on smear campaigns and harassment thanks to 1) Monsanto (for my precautionary principle work), 2) The eugenists 'race realists' alt-right (IQ work), 3) Covid and vaccine deniers (pandemics tail risk work), and now 4) Bitcoiners who strangely overlap with Covid Deniers. I also learned from my friend Ralph Nader (smeared by GM in the 1970s) who guided me through the steps.How do you smear someone? Along with the nasty mob harassment, you make sure to demonize and prevent the person from being able to shout that he or she is being harassed [See fig.1]. Snowden called the harassers my 'victims', a category to which he subsequently identified himself, simply because I responded to him.Historically, abusers trying to pass for victim receive extra punishment.That episode of smear campaigning confirms that Snowden is neither bright nor distinctly innovative (now explain what do the Slavonicophones have to gain by having to carry a humorless and not particularly sharp guest who will stick around for the next few decades?) Snowden simply took my Twitter posts, showed my (colorful) responses, while hiding the identity of the people I was responding to. Elementary one-sided evidence, selective referencing, the cherry picking one gets in Charlatan-Training-101. Needless to say he did, in the process, some basement level virtue signaling and some bungled bigotteering. So, that total imbecile, Ed Snowden, didn't realize that I came to Twitter to get corrected and collect free comments on my books and papers as evidenced below thanks to a Finnish engineer, Paavo Hietanen, who came first to bust Snowden."))


'''
### This code works for predicting the probabilites for a file using its url ####

import requests

url = "https://42lms.s3-accelerate.amazonaws.com/c_2.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230503T075012Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAS2KBMDAVRCT2SNOV%2F20230503%2Fap-northeast-1%2Fs3%2Faws4_request&X-Amz-Signature=ed10474c286ce727907760ef3b6857db3d0577cb8b0c84b7bd0a3d312d281ad5"
response = requests.get(url)

if response.status_code == 200:
    # File contents
    file_contents = response.content
    
    # File size in bytes
    file_size = len(file_contents)
    
    # File name from URL
    file_name = url.split("/")[-1]
    
    # File type from content type header
    content_type = response.headers.get("Content-Type")
    file_type = content_type.split("/")[-1]
    
    # Prediction of the text document
    file_prediction = predict_pipeline(file_contents.decode('utf-8'))
    
    # Print file details
    print(f"File Name: {file_name}")
    print(f"File Size: {file_size} bytes")
    print(f"File Type: {file_type}")
    print(file_contents.decode('utf-8'))
    #print(f"File Contents: {file_contents.decode('utf-8')}")
    print(f"File Prediction: {file_prediction}")
else:
    print("Error downloading file")
    
'''
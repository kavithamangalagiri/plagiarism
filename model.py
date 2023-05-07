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

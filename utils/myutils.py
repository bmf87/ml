import numpy as np
import pandas as pd
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def rm_punctuation(text):
  if(type(text)==float):
    return text
  
  parsed = ""  
  for t in text:     
    if t not in string.punctuation:
      parsed+=t

  # Items not in string.punctuation
  replacements = [("'", ''), ('"', '')]

  for char, replacement in replacements:
    if char in text:
      parsed = parsed.replace(char, replacement)
  return parsed
  

# 
# Remove English Stop Words
# Caution: May have lengthy runtime (~12mins)
#
def rm_stop_words(text):
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Convert back to string
    processed_text = ' '.join(filtered_tokens)

    return processed_text



# Create Lemmas
def lemmatize_text(text):
    tokens = word_tokenize(text)
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemma_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Convert back to string
    processed_text = ' '.join(lemma_tokens)

    return processed_text


def create_posTags(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
       
    return pos_tags

# Filter text based on POS tag input
def filter_pos(text, pos_tags):
   output = ""
   tmp = ""
   for sentence in text:
        #print(f'sentence: {sentence}')
        for word in sentence:
            #print(f'len(word): {len(word)}')      
            if word in pos_tags:
               #print(f'word match on {word}: --> {sentence[0]}')
               tmp += sentence[0] + " "
                        
        output += tmp
        tmp = ""
   return output
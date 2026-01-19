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


# Missing Values Table
def missing_values_table(df, summary=True):
    # Total missing values
    mis_val = df.isnull().sum()
    rows_with_missing_values = df.isna().any(axis=1).sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Create table with results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename columns
    mis_val_table_ren_columns = mis_val_table.rename(
      columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort by percent missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
      mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
      '% of Total Values', ascending=False).round(1)

    # Print some summary information
    if summary:
      print(f"""
          Row Total in dataframe: {len(df)}
          Rows with missing values: {rows_with_missing_values}
          Dataframe Columns: {(df.shape[1])}.
          Dataframe Column with missing values: {mis_val_table_ren_columns.shape[0]}
          
        """)

    # Return df missing info
    return mis_val_table_ren_columns
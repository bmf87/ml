import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys, os


# Add the parent directory to sys.path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock nltk dependencies if they don't exist
try:
    import nltk
except ImportError:
    mock_nltk = MagicMock()
    sys.modules['nltk'] = mock_nltk
    sys.modules['nltk.corpus'] = MagicMock()
    sys.modules['nltk.tokenize'] = MagicMock()
    sys.modules['nltk.stem'] = MagicMock()

from utils import dataprep_utils

class TestDataprepUtils(unittest.TestCase):

    def test_rm_punctuation_string(self):
        text = "Hello, world! This is a test."
        expected = "Hello world This is a test"
        self.assertEqual(dataprep_utils.rm_punctuation(text), expected)

    def test_rm_punctuation_with_quotes(self):
        text = "He said, 'Hello'."
        expected = "He said Hello"
        self.assertEqual(dataprep_utils.rm_punctuation(text), expected)

    def test_rm_punctuation_float(self):
        # The function explicitly checks for float and returns it
        val = 3.14
        self.assertEqual(dataprep_utils.rm_punctuation(val), val)

    @patch('utils.dataprep_utils.word_tokenize')
    @patch('utils.dataprep_utils.stopwords')
    def test_rm_stop_words(self, mock_stopwords, mock_word_tokenize):
        mock_word_tokenize.return_value = ['This', 'is', 'a', 'test']
        mock_stopwords.words.return_value = ['is', 'a']
        
        text = "This is a test"
        expected = "This test"
        
        result = dataprep_utils.rm_stop_words(text)
        self.assertEqual(result, expected)

    @patch('utils.dataprep_utils.word_tokenize')
    @patch('utils.dataprep_utils.WordNetLemmatizer')
    def test_lemmatize_text(self, mock_lemmatizer_class, mock_word_tokenize):
        mock_word_tokenize.return_value = ['cats', 'running']
        mock_lemmatizer_instance = mock_lemmatizer_class.return_value
        # Configure the mock to return specific values for specific inputs
        def side_effect(token):
            if token == 'cats': return 'cat'
            if token == 'running': return 'run'
            return token
        mock_lemmatizer_instance.lemmatize.side_effect = side_effect
        
        text = "cats running"
        expected = "cat run"
        
        result = dataprep_utils.lemmatize_text(text)
        self.assertEqual(result, expected)

    @patch('utils.dataprep_utils.word_tokenize')
    @patch('utils.dataprep_utils.pos_tag')
    def test_create_posTags(self, mock_pos_tag, mock_word_tokenize):
        mock_word_tokenize.return_value = ["Hello", "world"]
        mock_pos_tag.return_value = [("Hello", "NNP"), ("world", "NN")]
        
        text = "Hello world"
        result = dataprep_utils.create_posTags(text)
        self.assertEqual(result, [("Hello", "NNP"), ("world", "NN")])

    def test_filter_pos(self):
        # filter_pos iterates over sentences, then words.
        # It expects text to be a list of lists (sentences of words) or similar iterable structure?
        # Looking at code:
        # for sentence in text:
        #    for word in sentence:
        #        if word in pos_tags: ...
        # But wait, create_posTags returns a list of tuples (word, tag).
        # The function signature is filter_pos(text, pos_tags). 
        # The logic: if word in pos_tags. 
        # If pos_tags is the output of create_posTags, it's a list of tuples.
        # So `word in pos_tags` will check if the word (string) is inside the list of tuples? No, that won't work effectively if `word` is just a string.
        # Let's re-read filter_pos logic in source.
        # if word in pos_tags:
        #    tmp += sentence[0] + " "
        
        # It seems `pos_tags` is expected to be a list of allowed words? Or maybe the argument name is misleading and it expects a list of words to keep?
        # Or maybe it expects `pos_tags` to be a dictionary or set of allowable items?
        # Unclear from just looking at it. But let's assume it filters words based on input.
        # Wait, `tmp += sentence[0] + " "` -> it always adds `sentence[0]`. This looks buggy or I misunderstand `sentence`.
        # logic: for sentence in text: for word in sentence: if word in pos_tags: ...
        # If text is [['word1', 'word2']], sentence is ['word1', 'word2'].
        # word is 'word1'.
        # if 'word1' in pos_tags...
        # tmp += sentence[0] + " " -> Adds 'word1' + " ".
        
        # NOTE: The code seems to construct a string from `sentence[0]`?
        # If `word` matches, it adds `sentence[0]`. 
        # If `text` is a list of tuples (word, tag), then `sentence` is (word, tag).
        # `sentence[0]` is the word.
        # `word` in loop `for word in sentence`... `sentence` is a tuple?
        # If sentence is ('Hello', 'NNP').
        # word will be 'Hello', then 'NNP'.
        # If 'NNP' in pos_tags (where pos_tags is maybe a list of allowed tags like ['NN', 'NNP']?)
        # Then it adds sentence[0] (which is 'Hello') to output.
        
        # This strongly suggests `text` input to `filter_pos` is expected to be a list of (word, tag) tuples (like output of pos_tag), 
        # and `pos_tags` argument is a list of TAGS to keep (e.g. ['NN', 'VB']).
        
        # Let's test this hypothesis.
        # text = [('Hello', 'NNP'), ('world', 'NN')]
        # pos_tags = ['NN']
        # Loop 1: sentence = ('Hello', 'NNP'). 
        #   Inner loop: word='Hello'. 'Hello' in ['NN']? No.
        #   Inner loop: word='NNP'. 'NNP' in ['NN']? No.
        # Loop 2: sentence = ('world', 'NN').
        #   Inner loop: word='world'. No.
        #   Inner loop: word='NN'. Yes!
        #   tmp += sentence[0] ("world") + " "
        # output += "world "
        
        # This seems to be the logic.
        
        data = [('Hello', 'NNP'), ('is', 'VBZ'), ('test', 'NN')]
        tags_to_keep = ['NN', 'NNP']
        
        # Expect 'Hello' (matched NNP) and 'test' (matched NN)
        # 'is' has VBZ, not in list.
        
        expected = "Hello test " 
        
        # Let's verify what it actually does.
        # sentence=('Hello', 'NNP'). word='Hello' (no), word='NNP' (yes). Adds 'Hello '.
        # sentence=('is', 'VBZ'). word='is' (no), word='VBZ' (no).
        # sentence=('test', 'NN'). word='test' (no), word='NN' (yes). Adds 'test '.
        
        result = dataprep_utils.filter_pos(data, tags_to_keep)
        self.assertEqual(result, expected)

    def test_missing_values_table(self):
        df = pd.DataFrame({
            'A': [1, 2, np.nan],
            'B': [4, np.nan, np.nan],
            'C': [7, 8, 9]
        })
        
        # Capture stdout to verify print
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            result = dataprep_utils.missing_values_table(df, summary=True)
        finally:
            sys.stdout = sys.__stdout__
            
        # Check result dataframe
        # A: 1 missing (33.3%)
        # B: 2 missing (66.7%)
        # C: 0 missing
        # Result should only contain columns with missing values, sorted by % desc.
        # Row order: B then A.
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result.index[0], 'B')
        self.assertEqual(result.index[1], 'A')
        self.assertEqual(result.loc['B', 'Missing Values'], 2)
        self.assertEqual(result.loc['A', 'Missing Values'], 1)
        
        # Check summary print
        output = captured_output.getvalue()
        self.assertIn("columns with missing values: 2", output.lower())

if __name__ == '__main__':
   unittest.main()

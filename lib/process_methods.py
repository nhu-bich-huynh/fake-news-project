import re
import pandas as pd
from cleantext import clean
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

nltk.download('punkt')
nltk.download("stopwords")


def clean_text(text: str) -> str:
    """
    Clean the data
    """

    # Replace dates with <DATE> (assuming date format MM DD,YYYY (fx Jan. 8, 2017 or april 4, 1999))
    text = re.sub(r'(?:[a-zA-Z]+)\.?\s+[0-9]{1,2},\s+[0-9]{4}', '_DATE_', text)

    # Remove multiple white spaces, tabs, new lines
    text = re.sub(r' +|\n+|\t+',' ', text)
    
    text = clean(text,
        lower=True,                    # lowercase text
        no_urls=True,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_numbers=True,               # replace all numbers with a special token      
        replace_with_url="_URL_",
        replace_with_email="_EMAIL_",
        replace_with_number="_NUM_"            
    )

    return text


def remove_stopwords(text: str) -> str:
    """
    Tokenize text and remove stopwords
    """

    stop_words = set(stopwords.words('english'))

    # Tokenize the input text
    word_tokens = nltk.word_tokenize(text)

    # Filter out stopwords
    filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]

    # Join the filtered words back into a single string
    nostop_text = ' '.join(filtered_sentence)

    return nostop_text


def remove_word_variations(text: str) -> str:
    """
    Remove word variations, stemming
    """

    # Initialize a SnowballStemmer with English language
    stemmer = SnowballStemmer("english")
    
    # Tokenize the input text
    word_tokens = nltk.word_tokenize(text)
    
    # Stem each word token
    stemmed_words = [stemmer.stem(word) for word in word_tokens]
    
    # Join the stemmed words back into a single string
    stemmed_text = ' '.join(stemmed_words)
    
    return stemmed_text


def get_word_freq(text: str) -> int:
    """
    Count unique words in text (word frequency of content_clean)
    """
    tokens = nltk.word_tokenize(text)
    return len(set(tokens))


def reduction_rate(df: pd.DataFrame, src_col_a: str, src_col_b: str):
    """
    Compute reduction rate and store result in new column in dataframe
    """
    col_a = df[src_col_a]
    col_b = df[src_col_b]
    return round(((col_a - col_b)/col_a) * 100, 3)


def train_valid_test(df: pd.DataFrame) -> None:
    """
    Splitting a pandas dataframe into 80/10/10 split: 80% traning data, 10% validation data, and 10% test data.
    The splitting is done by uniformly selecting random rows. Each split is saved as new csv files in 'data' subfolder.
    """
    
    # initialize dataframes
    training_data = pd.DataFrame()
    validation_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # training data split into 80/20
    # random_state saves the state of the random selection. This means it can be reproduced.
    training_data = df.sample(frac=0.8, random_state=0)

    # remaining data
    remaining_data = df.drop(training_data.index)

    # validation data
    validation_data = remaining_data.sample(frac=0.5, random_state=1)

    # test data
    test_data = remaining_data.drop(validation_data.index)

    # save as csv files
    training_data.to_csv('data/training_data.csv')
    validation_data.to_csv('data/validation_data.csv')
    test_data.to_csv('data/test_data.csv')
    
    return None


def group_data(df: pd.DataFrame, omitted_types: set, fake_types: set) -> pd.DataFrame:
    """
    Removed omitted types from dataset and group fake_types as 'fake'
    """

    # make copy
    df_out = df.copy(deep=True)

    # drop omitted types
    drop_indexes = df_out[ (df_out['type'].isin(omitted_types))].index
    df_out.drop(drop_indexes, inplace=True)

    # group fake types
    def change_to_fake(type: str) -> str:
        if type in fake_types:
            return 'fake'
        else:
            return type

    df_out['type'] = df_out['type'].apply(change_to_fake)

    # return dataframe
    return df_out
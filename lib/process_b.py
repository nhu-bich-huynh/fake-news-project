import lib.process_methods as pm
import pandas as pd
import swifter

def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:

    """
    Preprocessing pipeline for cleaning the '995,000_rows.csv' dataset
    """

    # deep copy
    clean_data = dataframe.copy(deep=True)

    # remove empty 'content' rows
    clean_data.dropna(subset=['content'], inplace=True)

    # remove unused columns
    clean_data.drop(['id',
                    'domain',
                    'url',
                    'scraped_at',
                    'inserted_at',
                    'updated_at',
                    'keywords',
                    'meta_keywords',
                    'meta_description',
                    'tags',
                    'summary',
                    'Unnamed: 0',
                    'source'
                    ], axis=1, inplace=True)

    # remove rows without type labels
    drop_null_types = clean_data[ (clean_data['type'].isnull())].index
    clean_data.drop(drop_null_types, inplace=True)

    # cleanup text on 'content' column and add into new column 'content_clean'
    clean_data['content_clean'] = clean_data['content'].swifter.apply(pm.clean_text)

    # Apply remove_stopwords to 'content_clean' column and create 'content_stopword' column
    clean_data['content_stopword'] = clean_data['content_clean'].swifter.apply(pm.remove_stopwords)

    # stemming
    clean_data['content_stem'] = clean_data['content_stopword'].swifter.apply(pm.remove_word_variations)

    return clean_data

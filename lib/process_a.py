import lib.process_methods as pm
import pandas as pd
import swifter


def preprocess(dataframe) -> None:
    """
    Preprocessing pipeline for cleaning up the 'news_sample.csv' dataset
    """

     # deep copy
    clean_data = dataframe.copy(deep=True)

    # remove empty content
    clean_data.dropna(subset=['content'], inplace=True)

    # cleanup text on 'content' column and add into new column 'content_clean'
    clean_data['content_clean'] = clean_data['content'].swifter.apply(pm.clean_text)

    # Apply remove_stopwords to 'content_clean' column and create 'content_stopword' column
    clean_data['content_stopword'] = clean_data['content_clean'].swifter.apply(pm.remove_stopwords)

    # stemming
    clean_data['content_stem'] = clean_data['content_stopword'].swifter.apply(pm.remove_word_variations)

    # get vocabulary sizes (word frequencies)
    clean_data['content_word_freq'] = clean_data['content_clean'].swifter.apply(pm.get_word_freq)
    clean_data['stop_word_freq'] = clean_data['content_stopword'].swifter.apply(pm.get_word_freq)
    clean_data['stem_word_freq'] = clean_data['content_stem'].swifter.apply(pm.get_word_freq)

    # get reduction rates
    clean_data['stop_reduction_rate'] = pm.reduction_rate(clean_data, 
                                                        'content_word_freq', 
                                                        'stop_word_freq',
                                                        )
    
    clean_data['stem_reduction_rate'] = pm.reduction_rate(clean_data, 
                                                        'content_word_freq', 
                                                        'stem_word_freq',
                                                        )
    

    return clean_data
# fake-news-project

### About

For this project, we have made several jupyter notebook files to structure the code for our models and processes. We did this to avoid having a single large file with all our implementations. This gave us a better overview as we progressed with our project and made it easier to collaborate in Git and GitHub. 

### Requirements

To install the package requirements for the miniconda enviroment, use the following command:

    ```bash
    conda create -y -n "fake-news-proj" python=3.12
    conda activate fake-news-proj
    pip install -r requirements.txt
    ```

### Data Files

- 'news_sample.csv' file is available at: https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv
- '995,000_rows.csv' file is available at: https://absalon.ku.dk/courses/72550/files/8102667?wrap=1. Download the file and add it to the 'data' folder.
- 'articles.csv' is available in the 'data' folder
- LIAR dataset files are available in the 'data' folder

### Reproduction Pipeline for Model Results

Please follow the pipeline below to reproduce all data files (~35 GB storage) required for the models.

    ```
    Pipline:
        1_preprocessing.ipynb
        2_process_baseline_features.ipynb
        3_process_bbc_articles.ipynb
        4_process_liar_data.ipynb
        5_sentence_transformer.ipynb
    ```
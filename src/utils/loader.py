"""Data files loader.
"""

import os
import random
import tarfile
import urllib.request

import nalp.utils.loader as l
import nalp.utils.preprocess as p

# Constants
DATA_FOLDER = 'data/'
DATA_FILES = ['amazon_customer_reviews', 'coco_image_captions',
              'wmt_emnlp17_news', 'yelp_reviews']
TAR_FILE = 'language_modeling.tar.gz'
TAR_PATH = DATA_FOLDER + TAR_FILE
TAR_URL = 'https://recogna.tech/files/datasets/' + TAR_FILE


def download_file(url, output_path):
    """Downloads a file given its URL and the output path to be saved.

    Args:
        url (str): URL to download the file.
        output_path (str): Path to save the downloaded file.

    """

    # Checks if file exists
    file_exists = os.path.exists(output_path)

    # If file does not exist
    if not file_exists:
        print(f'Downloading file: {url}')

        # Checks if data folder exists
        folder_exists = os.path.exists(DATA_FOLDER)

        # If data folder does not exist
        if not folder_exists:
            # Creates the folder
            os.mkdir(DATA_FOLDER)

        # Downloads the file
        urllib.request.urlretrieve(url, output_path)

        print('File downloaded.')


def untar_file(file_path):
    """Decompress a file with .tar.gz.

    Args:
        file_path (str): Path of the file to be decompressed.

    Returns:
        The folder that has been decompressed.

    """

    # Opens a .tar.gz file with `file_path`
    with tarfile.open(file_path, "r:gz") as tar:
        # Defines the path to the folder and check if it exists
        folder_path = file_path.split('.tar.gz')[0]
        folder_path_exists = os.path.exists(folder_path)

        # If path does not exists
        if not folder_path_exists:
            print(f'Decompressing file: {file_path}')

            # Extracts all files
            tar.extractall(path=folder_path)

            print('File decompressed.')

    return folder_path


def load_data(file_name):
    """Loads data from pre-defined dataset files.

    Args:
        file_name (str): Name of data file to be loaded (without extension).

    Returns:
        List of sentences from a text-based data file.

    """

    # Checks if file's name really exists
    if file_name not in DATA_FILES:
        # If not, raises an exception
        raise Exception('Data not supported yet.')

    # Downloads the file
    download_file(TAR_URL, TAR_PATH)

    # Decompresses the file
    folder_path = untar_file(TAR_PATH)

    # Loads the data and split into sentences
    sentences = l.load_txt(f'{folder_path}/{file_name}.txt').splitlines()

    return sentences


def split_data(sentences, train_split=0.8, val_split=0.1, test_split=0.1, seed=0):
    """Splits the data according to provided percentages.

    Args:
        sentences (list): List of sentences to be splitted.
        train_split (float): Training set split.
        val_split (float): Validation set split.
        test_split (float): Testing set split.
        seed (int): Random seed.

    Returns:
        Lists holding the training, validation and testing sets.

    """

    # Defines the random seed
    random.seed(seed)

    # Checks if supplied splits sum to one
    if train_split + val_split + test_split != 1:
        # If not, raises an exception
        raise Exception('Splits do not sum to 1.')

    # Calculates the number of input samples
    n_samples = len(sentences)

    # Calculates the number of samples per set
    # Note we don't need the number of test samples as it will be the remaining samples
    train_samples = round(n_samples * train_split)
    val_samples = round(n_samples * val_split)

    # Generates a shuffled list
    random.shuffle(sentences)

    # Splices the array into sets
    train = sentences[:train_samples]
    val = sentences[train_samples:train_samples+val_samples]
    test = sentences[train_samples+val_samples:]

    return train, val, test


def tokenize_data(sentences):
    """Tokenizes the data according to a pre-defined pipeline.

    Args:
        sentences (list): List of sentences to be tokenized.

    Returns:
        Tokenized sentences.

    """

    # Defines the tokenization pipeline
    pipe = p.pipeline(p.lower_case, p.valid_char, p.tokenize_to_word)

    # Tokenizes the data
    tokens = [pipe(sentence) for sentence in sentences]

    return tokens

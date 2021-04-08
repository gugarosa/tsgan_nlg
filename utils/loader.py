"""Data files loader.
"""

import os
import tarfile
import urllib.request

import nalp.utils.loader as l

# Constants
DATA_FOLDER = 'data/'
DATA_FILES = ['amazon_customer_reviews', 'coco_image_captions', 'google_one_billion_words', 'wmt_emnlp17_news']
TAR_FILE = 'language_modelling.tar.gz'
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
        Text-based data from a possible file.

    """

    # Checks if file's name really exists
    if file_name not in DATA_FILES:
        # If not, raises an exception
        raise Exception(f'Data not supported yet.')

    # Downloads the file
    download_file(TAR_URL, TAR_PATH)

    # Decompresses the file
    folder_path = untar_file(TAR_PATH)

    # Loads the auxiliary data
    data = l.load_txt(f'{folder_path}/{file_name}.txt')

    return data

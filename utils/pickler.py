"""Save and load custom objects using Pickle.
"""

import pickle


def save_to_file(output_path, **kwargs):
    """Saves an object into a pickle file.

    Args:
        output_path (str): Path to save the object.

    """

    # Iterates through all keyword arguments
    for k, v in kwargs.items():
        # Opens a write binary file
        with open(f'{output_path}_{k}.pkl', 'wb') as f:
            # Dumps the object
            pickle.dump(v, f)


def load_from_file(input_file):
    """Loads an object from a pickle file.

    Args:
        input_file (str): File to be loaded.

    Returns:
        The loaded object from the pickle file.

    """

    # Opens a read-only binary file
    with open(f'{input_file}', 'rb') as f:
        # Loads the object
        obj = pickle.load(f)

    return obj

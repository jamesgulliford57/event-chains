import numpy as np

def write_npy(output_directory, **data_arrays):
        """
        Write simulation data with a timestamp for uniqueness.
        Parameters:
        folder_name: provide name of folder in format f"...{}..."
        **data_arrays: keyword arguments
            Each keyword corresponds to the name of the array, and the value is the array to be saved.
        """
        import os

        # Save each array to its own file in the directory
        for array_name, array_data in data_arrays.items():
            file_path = os.path.join(output_directory, f"{array_name}.npy")
            np.save(file_path, array_data)

        print(f"Data saved in {output_directory}")

def write_json(output_directory, **data_arrays):
        """
        Write simulation data with a timestamp for uniqueness.
        Parameters:
        folder_name: provide name of folder in format f"...{}..."
        **data_arrays: keyword arguments
            Each keyword corresponds to the name of the array, and the value is the array to be saved.
        """
        import os
        import json

        # Save each array to its own file in the directory
        for array_name, array_data in data_arrays.items():
            file_path = os.path.join(output_directory, f"{array_name}.json")
            with open(file_path, "w") as f:
                  json.dump(array_data, f, indent=4)

        print(f"Data saved in {output_directory}")

def print_section(message, empty_lines_before=1, empty_lines_after=1, separator_length=30):
    """
    Print a formatted section with separators and optional empty lines.
    """
    from colorama import Fore, Style
    print("\n" * empty_lines_before, end="")  # Add empty lines before the section
    print(Fore.GREEN + "=" * separator_length)
    print(Fore.GREEN + message)
    print(Fore.GREEN + "=" * separator_length + Style.RESET_ALL)
    print("\n" * empty_lines_after, end="")  # Add empty lines after the section


def timer(func):
    from functools import wraps
    import time
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper
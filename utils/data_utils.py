def write_npy(output_directory, **data_arrays):
        """
        Write simulation data to npy files to provided output directory.

        Parameters
        ---
        output_directory : str
            Path to directory where data will be saved.
        **data_arrays: keyword arguments
            Each keyword corresponds to the name of the array, and the value is the array to be saved.
        """
        from os.path import join
        from numpy import save

        # Save each array to its own file in the directory
        for array_name, array_data in data_arrays.items():
            file_path = join(output_directory, f"{array_name}.npy")
            save(file_path, array_data)

        print(f"\nData saved in {output_directory}")

def write_json(output_directory, **data_arrays):
        """
        Write simulation data with a timestamp for uniqueness.

        Parameters
        ---
        output_directory : str
            Path to directory where data will be saved.
        **data_arrays: keyword arguments
            Each keyword corresponds to the name of the array, and the value is the array to be saved.
        """
        from os.path import join
        from json import dump

        # Save each array to its own file in the directory
        for array_name, array_data in data_arrays.items():
            file_path = join(output_directory, f"{array_name}.json")
            with open(file_path, "w") as f:
                  dump(array_data, f, indent=4)

        print(f"\nData saved in {output_directory}")  

def update_json(json_path, **items):
    from os import path 
    import json 

    if path.exists(json_path):
        with open(json_path, 'r+', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            for key, value in items.items():
                data[key] = value
            
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    else:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=4)

    return data
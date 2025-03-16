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

        print(f"\n{tuple(data_arrays.keys())} arrays written to {output_directory}")

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
        import json

        # Save each array to its own file in the directory
        for array_name, array_data in data_arrays.items():
            file_path = join(output_directory, f"{array_name}.json")
            with open(file_path, "w") as f:
                  json.dump(array_data, f, indent=4)

        print(f"\nData saved in {output_directory}")  

def read_json(json_path):
    """
    Read data from a json file.

    Parameters
    ---
    json_path : str
        Path to the json file.
    """
    import json
    import os

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                output = json.load(f)
        except json.JSONDecodeError:
            output = {}
    else:
        output = {}

    return output

def update_json(json_path, **items):
    """
    
    Parameters
    ---
    json_path : str
        Path to the json file.
    **items : keyword arguments
        Key-value pairs to be added to the json file.
    """
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

def set_colors(dim):
    """
    Create a list of colors for plotting based on the number of dimensions.

    Parameters
    ---
    dim : int
        Number of dimensions.
    """
    import matplotlib.colors as mcolors

    colors = ["firebrick", "black", "dimgray", "darkred", "brown", "maroon", "gray", "darkslategray"]
    n_colors = len(colors)
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=n_colors)
    if dim == 1:
        cpt_colors = ["firebrick"]  
    else:
        cpt_colors = [custom_cmap(i / (dim - 1)) for i in range(dim)]
    return cpt_colors
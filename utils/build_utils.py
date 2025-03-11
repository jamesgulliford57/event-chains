def parse_value(value):
    """
    Attempt to convert string value to corresponding Python literal.
    If the conversion fails the original string is returned.
    """
    import ast
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    
def parse_possible_list(value):
    """
    Parse input that is either a list or a single int or float.
    """
    import ast
    if not isinstance(value, str):
        return [value]

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except Exception:
        pass

    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        items = inner.split(',')
        parsed_items = []
        for item in items:
            item = item.strip()
            if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                item = item[1:-1].strip()
                parsed_items.append(item)
            else:
                try:
                    parsed_items.append(float(item))
                except ValueError:
                    parsed_items.append(item)
        return parsed_items

    return [value]

def to_camel_case(s):
    """
    Convert a snake_case string to CamelCase.
    """
    return ''.join(word.capitalize() for word in s.split('_'))

def to_snake_case(s):
    """
    Convert a CamelCase string to snake_case.
    """
    import re
    # Insert an underscore between a lowercase letter or digit and an uppercase letter followed by lowercase letters.
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
    # Insert an underscore between a lowercase letter/digit and an uppercase letter.
    snake = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return snake.lower()

def create_directory(output_dir, model_name, scheme, final_time, n, init_value, num_paths, do_timestamp=False):
    """
    Create output directory for simulation results.

    Parameters
    ---
    output_dir : str
        Path to highest leveldirectory where output will be saved.
    model_name : str
        Name of the model being simualted.
    scheme : str
        Numerical scheme used to simulate model.
    final_time : float
        Time horizon of simulation.
    n : int
        Discretisation parameter, number of intervals T is divided into.
    init_value : float
        Initial condition for random solution trajectories.
    num_paths : int
        Number of trajectories to be simulated.
    do_timestamp : bool
        Flag to include timestamp in output directory name.
    """

    import datetime
    import os 

    if do_timestamp:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        output_directory = os.path.join(output_dir, model_name, scheme.lower(), f'final_time={final_time}_n={n}_init={init_value}_num_paths={num_paths}', timestamp)
    else:
        output_directory = os.path.join(output_dir, model_name, scheme.lower(), f'final_time={final_time}_n={n}_init={init_value}_num_paths={num_paths}')
    os.makedirs(output_directory, exist_ok=True)

    return output_directory

def load_class(model_name):
    """
    Dynamically load class based on provided model name.
    
    Parameters
    ---
    model_name : str
        The name of the model.
    """
    import importlib

    module_name = model_name.lower()  
    class_name = to_camel_case(model_name) 
    module = importlib.import_module(f"models.{module_name}")
    return getattr(module, class_name)
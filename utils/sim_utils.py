def print_section(message, empty_lines_before=1, empty_lines_after=1):
    """
    Print a formatted section with separators and optional empty lines.

    Parameters
    ---
    message : str
        Message to be displayed in the section.
    empty_lines_before : int
        Number of empty lines to be printed before the section.
    empty_lines_after : int
        Number of empty lines to be printed after the section.
    """
    from colorama import Fore, Style
    
    separator_length=len(message)
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

def progress_bar_decorator(desc="Simulating", unit="step"):
    """
    Decorator that provides tqdm progress bar.

    Parameters
    ---
    desc : str
        Description of the progress bar.
    unit : str  
        Unit of the progress bar.
    """
    from tqdm import tqdm
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to grab 'N' from kwargs
            N = kwargs.get('N')
            if N is None:
                raise ValueError("The wrapped function must have an 'N' keyword argument.")
            
            # Create the progress bar
            with tqdm(total=N-1, desc=desc, unit=unit, ncols=60, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                # Insert the progress bar into the function kwargs
                kwargs['pbar'] = pbar
                # Call the actual function
                result = func(*args, **kwargs)
            
            return result
        return wrapper
    return decorator


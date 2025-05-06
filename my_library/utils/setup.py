import os
import sys
from typing import Tuple


def setup_project_path(root_dir_names: Tuple[str, ...] = ("ML_competitions", "workspace")) -> str:
    """
    Adds the project root directory to sys.path if any of the specified directory names
    are found in the current working directory path.
    
    This is useful when using different project root directory names in local and Docker
    container environments.
    
    Args:
        root_dir_names: A tuple of directory names to be recognized as project roots
                       Default is ("ML_competitions", "workspace")
    
    Returns:
        str: The configured project root path
        
    Raises:
        RuntimeError: If none of the specified directory names are found in the current path
    """
    cwd = os.getcwd()
    project_root = None
    
    # Check if any of the specified directory names are in the current path
    for dir_name in root_dir_names:
        if dir_name in cwd:
            project_root = cwd[:cwd.index(dir_name) + len(dir_name)]
            break
    
    if project_root:
        if project_root not in sys.path:
            sys.path.append(project_root)
            print(f"Added to sys.path: {project_root}")
        return project_root
    else:
        raise RuntimeError(f"None of {root_dir_names} found in current path: {cwd}")


# Usage example
if __name__ == "__main__":
    try:
        project_path = setup_project_path()
        print(f"Project root: {project_path}")
    except RuntimeError as e:
        print(f"Error: {e}")

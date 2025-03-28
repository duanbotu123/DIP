import numpy as np
import os
import argparse

def load_npz(file_path):
    """
    Load and print information about an npz file
    
    Args:
        file_path (str): Path to the npz file
        
    Returns:
        dict: Dictionary containing the npz data
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    # Load the npz file
    data = np.load(file_path)
    
    # Print information about the npz file
    print(f"NPZ file: {file_path}")
    print(f"Keys: {data.files}")
    
    # Print shape and type information for each array
    for key in data.files:
        array = data[key]
        print(f"- {key}: shape={array.shape}, dtype={array.dtype}")
    
    return dict(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and inspect an NPZ file")
    parser.add_argument("file_path", help="Path to the npz file")
    
    args = parser.parse_args()
    data = load_npz(args.file_path)
    
    # Example of accessing a specific array if needed
    # (Uncomment and modify as needed)
    # if data and 'specific_key' in data:
    #     specific_array = data['specific_key']
    #     print(f"Value of specific_key: {specific_array}")
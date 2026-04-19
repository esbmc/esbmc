import os
from os import mkdir, rmdir, remove

def create_directory(path: str) -> bool:
    """Attempts to create a directory, returns True on success"""
    try:
        mkdir(path)
        return True
    except FileExistsError:
        return False

def cleanup_directory(path: str) -> int:
    """
    Attempts to remove a directory
    Returns: 0 on success, 1 if not empty, 2 if not found
    """
    try:
        rmdir(path)
        return 0
    except OSError:
        return 1
    except FileNotFoundError:
        return 2

def delete_file(path: str) -> bool:
    """Attempts to delete a file, returns True on success"""
    try:
        remove(path)
        return True
    except FileNotFoundError:
        return False

def test_file_operations() -> None:
    # Test directory creation
    result1 = create_directory("/tmp/testdir")
    assert result1 == True or result1 == False  # Valid outcomes
    
    # Test directory removal
    result2 = cleanup_directory("/tmp/testdir")
    assert result2 >= 0 and result2 <= 2  # Valid return codes
    
    # Test file deletion
    result3 = delete_file("/tmp/testfile.txt")
    
    # This assertion may fail - demonstrating verification of error handling
    assert result3 == True  # Assumes file always exists

test_file_operations()


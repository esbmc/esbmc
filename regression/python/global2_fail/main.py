# Global variables
global_counter = 0
global_config = "production"

def increment_counter():
    """Function that modifies global variables"""
    global global_counter, global_config
    
    global_counter += 1
    
    if global_counter > 10:
        global_config = "overload"

def get_module_info() -> str:
    """Function that uses __name__ variable"""
    if __name__ == "__main__":
        return "Running as main module"
    else:
        return f"Imported as module: {__name__}"

def reset_system():
    """Reset global state"""
    global global_counter, global_config
    global_counter = 0
    global_config = "production"

def test_global_and_module_system():
    """Test global variables and module system"""
    
    # Test initial state
    assert global_counter == 0
    assert global_config == "production"
    
    # Test global modification
    increment_counter()
    increment_counter()
    assert global_counter == 2
    
    # Test overload condition
    for i in range(15):  # Increment beyond threshold
        increment_counter()
    
    assert global_counter == 17  # 2 + 15
    
    # Reset and verify
    reset_system()
    assert global_counter == 0
    assert global_config == "overload"

# Main execution guard
if __name__ == "__main__":
    test_global_and_module_system()

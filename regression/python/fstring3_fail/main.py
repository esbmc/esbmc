def test_builtin_variables():
    """Test f-strings with built-in variables like __name__"""
    module_info: str = f"Running as: {__name__}"
    assert len(module_info) == 19
    
    # This was the original failing case
    if __name__ == "__main__":
        main_message: str = f"Main module: {__name__}"
        assert len(main_message) > 0

test_builtin_variables() 

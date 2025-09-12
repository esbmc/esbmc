def validate_user_input() -> bool:
    """Validate user input with nondeterministic modeling"""
    
    # ESBMC models input() as nondeterministic string up to 256 chars
    user_input = input("Enter a number: ")
    
    # Input validation
    if len(user_input) == 0:
        return False
    
    # Check if input contains only digits
    for i in range(len(user_input)):
        if i < len(user_input):  # Bounds check
            char = user_input[i]
            if char < '0' or char > '9':
                return False
    
    # Convert to integer and validate range
    try:
        number = int(user_input)
        return 1 <= number <= 100
    except ValueError:
        return False

def process_input_scenarios():
    """Process various input scenarios"""
    
    # Test with nondeterministic input
    is_valid = validate_user_input()
    
    if is_valid:
        print("Input is valid")
        # Additional processing for valid input
        user_data = input("Enter additional data: ")
        assert len(user_data) <= 256  # ESBMC's input limit
    else:
        print("Input is invalid")

def test_input_assumptions():
    """Test input with specific assumptions"""
    
    # Using assume to specify input constraints
    user_age_str = input("Enter age: ")
    
    # Assume input is not empty (for this test path)
    __ESBMC_assume(len(user_age_str) > 0)
    __ESBMC_assume(len(user_age_str) <= 3)  # Reasonable age input
    
    # Process assuming constraints are met
    try:
        age = int(user_age_str)
        __ESBMC_assume(age >= 0)  # Non-negative age
        
        if age < 18:
            category = "minor"
        elif age < 65:
            category = "adult"
        else:
            category = "senior"
        
    except ValueError:
        # This path should not be reachable given our assumptions
        assert False, "Conversion should succeed with our assumptions"

# Note: assume() function used for verification assumptions
def assume(condition: bool):
    """Assumption function for verification"""
    if not condition:
        # In ESBMC, this would prune the execution path
        pass

if __name__ == "__main__":
    process_input_scenarios()
    test_input_assumptions()


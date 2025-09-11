def validate_user_input() -> None:
    user_input = input()  # ESBMC models this as nondeterministic string

    # Convert to integer (may raise ValueError)
    try:
        number: int = int(user_input)

        # Use assumption to constrain the verification domain
        __ESBMC_assume(number >= 0)
        __ESBMC_assume(number <= 100)

        # Verify that our processing logic is correct for valid inputs
        if number < 50:
            result: int = number * 2
            assert result < 100
        else:
            result: int = number + 10
            assert result <= 110

    except ValueError:
        # Handle invalid input
        assert False  # Should not reach here with our assumptions


validate_user_input()

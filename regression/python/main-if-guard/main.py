def main():
    """Main function that only runs if __name__ == '__main__'."""
    if __name__ == "__main__":
        assert True
    else:
        # This should never happen in the main module
        assert False

if __name__ == "__main__":
    main()

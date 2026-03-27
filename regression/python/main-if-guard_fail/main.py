def main():
    """Main function that will fail if __name__ == '__main__'."""
    if __name__ == "__main__":
        # This should trigger a verification failure
        assert False
    else:
        assert True

if __name__ == "__main__":
    main()

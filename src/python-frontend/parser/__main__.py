"""CLI entry point for the python-frontend parser package."""

try:
    from .parser import main
except ImportError:
    # Support direct execution: ``python parser/__main__.py ...``
    from parser import main

if __name__ == "__main__":
    main()

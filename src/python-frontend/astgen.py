import ast
import sys
import importlib.util
import json

def import_module_by_name(module_name):
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        print(f"Error: Module '{module_name}' not found.")
        print(f"Please install it with: pip3 install {module_name}")
        sys.exit(1)


def check_usage():
    if len(sys.argv) != 2:
        print("Usage: python astgen.py <file path>")
        sys.exit(2)


def main():
    ast2json_module = import_module_by_name("ast2json")

    check_usage()

    filename = sys.argv[1]

    with open(filename, "r") as source:
        tree = ast.parse(source.read())


    ast_json = ast2json_module.ast2json(tree)
    json_filename = "/tmp/ast.json"

    with open(json_filename, "w") as json_file:
        json.dump(ast_json, json_file, indent=4)


if __name__ == "__main__":
    main()

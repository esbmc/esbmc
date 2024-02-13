import ast
import sys
import importlib.util
import json
import os

def check_usage():
    if len(sys.argv) != 3:
        print("Usage: python astgen.py <file path> <output directory>")
        sys.exit(2)


def import_module_by_name(module_name):
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        print(f"Error: Module '{module_name}' not found.")
        print(f"Please install it with: pip3 install {module_name}")
        sys.exit(1)


def process_imports(node, output_dir):
    module_name = node.module
    module = import_module_by_name(module_name)
    filename = module.__file__
    if filename.endswith('.pyc'):
        filename = filename[:-1]  # Remove the 'c' at the end for .pyc files
    generate_ast_json(filename, output_dir)


def generate_ast_json(filename, output_dir):
    with open(filename, "r") as source:
        tree = ast.parse(source.read())

    ast2json_module = import_module_by_name("ast2json")
    ast_json = ast2json_module.ast2json(tree)
    ast_json["filename"] = filename

    json_filename = os.path.join(output_dir, f"{os.path.basename(filename)}.json")

    with open(json_filename, "w") as json_file:
        json.dump(ast_json, json_file, indent=4)


def main():
    check_usage()
    filename = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(filename, "r") as source:
        tree = ast.parse(source.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            process_imports(node, output_dir)

    generate_ast_json(filename, output_dir)

if __name__ == "__main__":
    main()


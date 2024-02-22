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
    
    # Check if module is available/installed
    module = import_module_by_name(module_name)
    filename = module.__file__

    # Remove the 'c' at the end for .pyc files, if present
    if filename.endswith('.pyc'):
        filename = filename[:-1]

    # Add the full path recovered from importlib to the import node
    node.full_path = filename

    # Generate JSON file for each import
    for name_node in node.names:
        with open(filename, "r") as source:
          try:
            tree = ast.parse(source.read())
          except UnicodeDecodeError:
              continue
        generate_ast_json(tree, filename, name_node.name, output_dir, None)


def generate_ast_json(tree, python_filename, element_to_import, output_dir, output_file):
    # Filter elements to be imported from the module
    filtered_nodes = []
    if element_to_import:
        for node in tree.body:
            if (isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef)) and node.name == element_to_import:
                  filtered_nodes.append(node)
                  break

    ast2json_module = import_module_by_name("ast2json")
    ast_json = ast2json_module.ast2json(ast.Module(body=filtered_nodes) if filtered_nodes else tree)
    ast_json["filename"] = python_filename
    ast_json["ast_output_dir"] = output_dir

    if output_file:
        json_filename = os.path.join(output_dir, output_file)
    else:
        python_filename = python_filename[:-3]
        json_filename = os.path.join(output_dir, f"{os.path.basename(python_filename)}.json")

    with open(json_filename, "w") as json_file:
        json.dump(ast_json, json_file, indent=4)


def main():
    check_usage()
    filename = sys.argv[1]
    output_dir = sys.argv[2]

    sys.path.append(os.path.dirname(filename))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(filename, "r") as source:
        tree = ast.parse(source.read())
        
    # Handle imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            process_imports(node, output_dir)

    # Generate json for main file from AST
    generate_ast_json(tree, filename, None, output_dir, "ast.json")


if __name__ == "__main__":
    main()


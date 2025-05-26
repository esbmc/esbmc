import sys

# Detect the Python version
PY3 = sys.version_info[0] == 3

if not PY3:
    print("Python version: {}.{}.{}".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))
    print("ERROR: Please ensure Python 3 is available in your environment.")
    sys.exit(1)


import ast
import importlib.util
import json
import os
import glob
import base64
from preprocessor import Preprocessor


def check_usage():
    if len(sys.argv) != 3:
        print("Usage: python astgen.py <file path> <output directory>")
        sys.exit(2)

def is_imported_model(module_name):
    models = ["math", "os", "numpy"]
    return module_name in models

def is_unsupported_module(module_name):
    unsuported_modules = ["blah"]
    return module_name in unsuported_modules


def import_module_by_name(module_name, output_dir):
    if is_unsupported_module(module_name):
        print("ERROR: \"import {}\" is not supported".format(module_name))
        sys.exit(3)

    base_module = module_name.split(".")[0]

    if is_imported_model(base_module):
        parts = module_name.split(".")
        model_dir = os.path.join(output_dir, "models")
        path = os.path.join(model_dir, *parts) + ".py"

        if not os.path.exists(path):
            path = os.path.join(model_dir, *parts, "__init__.py")

        return os.path.abspath(path)

    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        print("ERROR: Module '{}' not found.".format(module_name))
        print("Please install it with: pip3 install {}".format(module_name))
        sys.exit(4)


    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        print("ERROR: Module '{}' not found.".format(module_name))
        print("Please install it with: pip3 install {}".format(module_name))
        sys.exit(4)


def encode_bytes(value):
    return base64.b64encode(value).decode('ascii')

def add_type_annotation(node):
    value_node = node.value
    if isinstance(value_node, ast.Str):
        value_node.esbmc_type_annotation = "str"
    elif isinstance(value_node, ast.Bytes):
        value_node.esbmc_type_annotation = "bytes"
        value_node.encoded_bytes = encode_bytes(value_node.value)


import_aliases = {}

def process_imports(node, output_dir):
    """
    Process import statements in the AST node.

    Parameters:
        - node: The import node to process.
        - output_dir: The directory to save the generated JSON files.
    """

    global import_aliases

    if isinstance(node, (ast.Import)):
        for alias_node in node.names:
            module_name = alias_node.name
            alias = alias_node.asname or module_name
            import_aliases[alias] = module_name
        imported_elements = None
    else: #ImportFrom
        module_name = node.module
        imported_elements = node.names
        if module_name:
            import_aliases[module_name] = module_name

    # Check if module is available/installed
    if is_imported_model(module_name):
        models_dir = os.path.join(output_dir, "models")
        filename = os.path.join(models_dir, module_name + ".py")
    else:
        module = import_module_by_name(module_name, output_dir)
        filename = module.__file__

    # Add the full path recovered from importlib to the import node
    node.full_path = filename

    # Generate JSON file for imported elements
    try:
        with open(filename, "r") as source:
            tree = ast.parse(source.read())
            generate_ast_json(tree, filename, imported_elements, output_dir)
    except UnicodeDecodeError:
        pass


def generate_ast_json(tree, python_filename, elements_to_import, output_dir):
    """
    Generate AST JSON from the given Python AST tree.

    Parameters:
        - tree: The Python AST tree.
        - python_filename: The filename of the Python source file.
        - elements_to_import: The elements (classes or functions) to be imported from the module.
        - output_dir: The directory to save the generated JSON file.
    """

    # Filter elements to be imported from the module
    filtered_nodes = []
    if elements_to_import is not None and elements_to_import:
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                for elem_info in elements_to_import:
                    if node.name == elem_info.name:
                        filtered_nodes.append(node)


    # Convert AST to JSON
    ast2json_module = import_module_by_name("ast2json", "")
    ast_json = ast2json_module.ast2json(ast.Module(body=filtered_nodes) if filtered_nodes else tree)
    ast_json["filename"] = python_filename
    ast_json["ast_output_dir"] = output_dir

    # Construct JSON filename
    if python_filename.endswith('__init__.py'):
        # Use the parent directory name instead of the filename
        dir_name = os.path.basename(os.path.dirname(python_filename))
        json_filename = os.path.join(output_dir, f"{dir_name}.json")
    else:
        # Otherwise, use the filename without the '.py' extension
        json_filename = os.path.join(output_dir, f"{os.path.basename(python_filename[:-3])}.json")

    json_dir = os.path.dirname(json_filename)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # Write AST JSON to file
    try:
        with open(json_filename, "w") as json_file:
            json.dump(ast_json, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print("Error writing JSON file: {}".format(e))


def detect_and_process_submodules(node, processed_submodules, output_dir):
    """
    Detects the usage of submodules in the AST and processes them.

    Parameters:
        - node: The AST node to process for submodules.
        - import_aliases: Dict mapping aliases to actual module names (e.g., 'np' → 'numpy').
        - processed_submodules: Set to avoid reprocessing submodules.
        - output_dir: The directory to save the generated JSON files.

    """
    global import_aliases

    if isinstance(node, ast.Attribute):
        value = node.value
        if isinstance(value, ast.Name):
            alias = value.id
            base_module = import_aliases.get(alias)

            # Only process submodules of supported model modules
            if not base_module or not is_imported_model(base_module):
                return

            full_module = f"{base_module}.{node.attr}"

            # Avoid reprocessing the same submodule
            if full_module in processed_submodules:
                return
            processed_submodules.add(full_module)

            try:
                module = import_module_by_name(full_module, output_dir)
            except SystemExit:
                return

            if isinstance(module, str):
                file_path = module
            else:
                file_path = module.__file__

            if file_path.endswith('__init__.py') or os.path.isdir(file_path):
                module_dir = os.path.dirname(file_path)
            else:
                module_dir = os.path.dirname(file_path)

            for root, dirs, files in os.walk(module_dir):
                for file in files:
                    if file.endswith('.py'):
                        full_path = os.path.join(root, file)
                        try:
                            with open(full_path, "r") as f:
                                tree = ast.parse(f.read())
                                generate_ast_json(tree, full_path, None, output_dir + "/" + base_module)
                        except UnicodeDecodeError:
                            continue

def main():
    check_usage()
    filename = sys.argv[1]
    output_dir = sys.argv[2]

    # Add the script directory to the import search path
    sys.path.append(os.path.dirname(filename))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process and convert AST for main file
    with open(filename, "r") as source:
        tree = ast.parse(source.read())

    # Apply AST transformations
    preprocessor = Preprocessor(filename)
    tree = preprocessor.visit(tree)

    # Tracking of imported modules and aliases
    processed_submodules = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Handle imports
            process_imports(node, output_dir)
        elif isinstance(node, ast.Assign):
            # Add type annotation on assignments
            add_type_annotation(node)
        elif isinstance(node, ast.Attribute):
            # Detect and process submodule usage
            detect_and_process_submodules(node, processed_submodules, output_dir)

    # Generate JSON from AST for the main file.
    generate_ast_json(tree, filename, None, output_dir)

    # Process and convert AST for memory models
    models_dir = os.path.join(output_dir, "models")

    # Iterate over all .py files in the directory
    for python_file in glob.glob(os.path.join(models_dir, "*.py")):
        filename = os.path.basename(python_file)
        module_name = filename[:-3]

        if is_imported_model(module_name):
            continue;

        with open(python_file) as model:
            model_tree = ast.parse(model.read())
            # Generate JSON from AST for the memory models.
            generate_ast_json(model_tree, filename, None, output_dir)


if __name__ == "__main__":
    main()

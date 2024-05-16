import ast
import sys
import importlib.util
import json
import os
import glob
import base64

class ForRangeToWhileTransformer(ast.NodeTransformer):
    def __init__(self):
        self.target_name = ""

    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            start = node.iter.args[0]
            end = node.iter.args[1]
            if len(node.iter.args) > 2:
                step = node.iter.args[2]
            else:
                step = ast.Constant(value=1)

            start_assign = ast.AnnAssign(target=ast.Name(id='start', ctx=ast.Store()), annotation=ast.Name(id='int', ctx=ast.Load()), value=start, simple=1)
            has_next_call = ast.Call(func=ast.Name(id='ESBMC_range_has_next_', ctx=ast.Load()), args=[start, end, step], keywords=[])
            has_next_assign = ast.AnnAssign(target=ast.Name(id='has_next', ctx=ast.Store()), annotation=ast.Name(id='bool', ctx=ast.Load()), value=has_next_call, simple=1)
            has_next_name = ast.Name(id='has_next', ctx=ast.Load())
            while_cond = ast.Compare(left=has_next_name, ops=[ast.Eq()], comparators=[ast.Constant(value=True)])
            transformed_body = []
            self.target_name = node.target.id
            for statement in node.body:
                transformed_body.append(self.visit(statement))
            while_body = transformed_body + [ast.Assign(targets=[ast.Name(id='start', ctx=ast.Store())], value=ast.Call(func=ast.Name(id='ESBMC_range_next_', ctx=ast.Load()), args=[ast.Name(id='start', ctx=ast.Load()), step], keywords=[])), ast.Assign(targets=[has_next_name], value=ast.Call(func=ast.Name(id='ESBMC_range_has_next_', ctx=ast.Load()), args=[ast.Name(id='start', ctx=ast.Load()), end, step], keywords=[]))]
            while_stmt = ast.While(test=while_cond, body=while_body, orelse=[])
            return [start_assign, has_next_assign, while_stmt]
        return node

    def visit_Name(self, node):
        if node.id == self.target_name:
            node.id = 'start'
        return node

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

def encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode('ascii')

def add_type_annotation(node):
    value_node = node.value
    if isinstance(value_node, ast.Str):
        value_node.esbmc_type_annotation = "str"
    elif isinstance(value_node, ast.Bytes):
        value_node.esbmc_type_annotation = "bytes"
        value_node.encoded_bytes = encode_bytes(value_node.value)


def process_imports(node, output_dir):
    """
    Process import statements in the AST node.

    Parameters:
        - node: The import node to process.
        - output_dir: The directory to save the generated JSON files.
    """

    if isinstance(node, (ast.Import)):
        module_name = node.names[0].name
        imported_elements = None
    else: #ImportFrom
        module_name = node.module
        imported_elements = node.names

    # Check if module is available/installed
    module = import_module_by_name(module_name)
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
    ast2json_module = import_module_by_name("ast2json")
    ast_json = ast2json_module.ast2json(ast.Module(body=filtered_nodes) if filtered_nodes else tree)
    ast_json["filename"] = python_filename
    ast_json["ast_output_dir"] = output_dir

    # Construct JSON filename
    json_filename = os.path.join(output_dir, f"{os.path.basename(python_filename[:-3])}.json")

    # Write AST JSON to file
    try:
        with open(json_filename, "w") as json_file:
            json.dump(ast_json, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing JSON file: {e}")


def main():
    check_usage()
    filename = sys.argv[1]
    output_dir = sys.argv[2]

    # Include the current directory for import search
    sys.path.append(os.path.dirname(filename))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process and convert AST for main file
    with open(filename, "r") as source:
        tree = ast.parse(source.read())

    transformer = ForRangeToWhileTransformer()
    tree = transformer.visit(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Handle imports
            process_imports(node, output_dir)
        elif isinstance(node, ast.Assign):
            # Add type annotation on assignments
            add_type_annotation(node)

    # Generate JSON from AST for the main file.
    generate_ast_json(tree, filename, None, output_dir)

    # Process and convert AST for memory models
    memory_models_dir = os.path.join(output_dir, "memory-models")

    # Iterate over all .py files in the directory
    for python_file in glob.glob(os.path.join(memory_models_dir, "*.py")):
        with open(python_file) as memory_model:
            mm_tree = ast.parse(memory_model.read())
            filename = os.path.basename(python_file)
            # Generate JSON from AST for the memory models.
            generate_ast_json(mm_tree, filename, None, output_dir)


if __name__ == "__main__":
    main()


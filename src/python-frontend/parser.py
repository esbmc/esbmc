import sys
import subprocess

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

def is_testing_framework(module_name):
    # Check if module is a testing framework that should be skipped.
    testing_frameworks = [
        "pytest",
    ]
    return module_name in testing_frameworks


def import_module_by_name(module_name, output_dir):
    if is_unsupported_module(module_name):
        print("ERROR: \"import {}\" is not supported".format(module_name))
        sys.exit(3)

    base_module = module_name.split(".")[0]

    # Skip typing module - it's for type annotations only and doesn't need AST processing.
    if base_module == "typing":
        return None

    # Skip testing frameworks - they don't contain logic to verify
    if is_testing_framework(base_module):
        return None

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
        # Try importing the parent module if this looks like a class/attribute reference
        if "." in module_name:
            parent = ".".join(module_name.split(".")[:-1])
            try:
                return importlib.import_module(parent)
            except ImportError:
                pass

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


def is_standard_library_file(filename):
    stdlib_paths = [
        '/usr/lib/python',
        '/usr/local/lib/python',
        '/Library/Frameworks/Python.framework',
        '/opt/homebrew/Cellar/python',  # Homebrew Python on macOS (Apple Silicon)
        '/usr/local/Cellar/python',     # Homebrew Python on macOS (Intel)
        '/opt/conda/lib/python',       # Conda standard installation path
    ]
    # Check fixed paths first (no expanduser needed)
    if any(filename.startswith(path) for path in stdlib_paths):
        return True
    # Check pyenv paths
    pyenv_root = os.environ.get('PYENV_ROOT', os.path.expanduser('~/.pyenv'))
    if pyenv_root and filename.startswith(pyenv_root):
        # Check if it's in the versions directory (standard library location)
        if '/versions/' in filename and '/lib/python' in filename:
            return True
    # Check conda paths (including user installations)
    if filename.startswith(os.path.expanduser('~/miniconda3/lib/python')) or \
       filename.startswith(os.path.expanduser('~/anaconda3/lib/python')):
        return True
    return False


def expand_star_import(module) -> list[str] | None:
    names = getattr(module, '__all__', None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith('_')]
    return names


def get_referenced_names(node):
    """
    Find all functions and classes referenced in a function or class definition.
    Returns a set of names that are called as functions or used in type annotations.
    """
    referenced = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            # Check if it's a direct function/class call (simple Name node)
            if isinstance(child.func, ast.Name):
                referenced.add(child.func.id)

        # Check for names in type annotations (return types, argument types, etc.)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Return type annotation
            if child.returns and isinstance(child.returns, ast.Name):
                referenced.add(child.returns.id)
            # Argument type annotations
            for arg in child.args.args:
                if arg.annotation and isinstance(arg.annotation, ast.Name):
                    referenced.add(arg.annotation.id)

        # Variable annotations (e.g., x: Foo = ...)
        elif isinstance(child, ast.AnnAssign):
            if isinstance(child.annotation, ast.Name):
                referenced.add(child.annotation.id)

    return referenced

import_aliases = {}
# Track all imports per module to combine them
module_imports = {}

def process_imports(node, output_dir):
    """
    Process import statements in the AST node.

    Parameters:
        - node: The import node to process.
        - output_dir: The directory to save the generated JSON files.
    """


    if isinstance(node, (ast.Import)):
        for alias_node in node.names:
            module_name = alias_node.name
            alias = alias_node.asname or module_name
            import_aliases[alias] = module_name
        imported_elements = None
    elif isinstance(node, ast.ImportFrom):
        module_name = node.module
        # If it's a star import, set the list to None to import everything
        if any(a.name == '*' for a in node.names):
            imported_elements = None
        else:
            imported_elements = node.names
        if module_name:
            import_aliases[module_name] = module_name

    # Track imports for this module
    if module_name not in module_imports:
        module_imports[module_name] = {'import_all': False, 'specific_names': set()}

    if imported_elements is None:
        # This is an "import module" or "from module import *"; mark to import everything
        module_imports[module_name]['import_all'] = True
    else:
        # Add specific names to the set
        for elem in imported_elements:
            module_imports[module_name]['specific_names'].add(elem.name)

    # Check if module is available/installed
    if is_imported_model(module_name):
        models_dir = os.path.join(output_dir, "models")
        filename = os.path.join(models_dir, module_name + ".py")
    else:
        module = import_module_by_name(module_name, output_dir)
        if module is None:
            # Module doesn't need processing (e.g., typing) - don't set full_path
            return

        # Check if module has __file__ attribute (built-in C extensions don't)
        if not hasattr(module, '__file__') or module.__file__ is None:
            # Skip built-in C extension modules (e.g., _sre, _socket, etc.)
            return

        filename = module.__file__

    # Don't process the file here; we'll do it once after collecting all imports
    node.full_path = filename


def resolve_module_file(module_qualname: str, output_dir: str) -> str | None:
    """Return file path for module qualname (or None if stdlib/missing)."""
    try:
        mod = import_module_by_name(module_qualname, output_dir)
    except SystemExit:
        return None
    filename = mod if isinstance(mod, str) else getattr(mod, "__file__", None)
    if not filename or is_standard_library_file(filename):
        return None
    if not os.path.exists(filename):  # e.g. math.pi is not a submodule
        return None
    return filename


def filter_imports(tree: ast.AST) -> ast.AST:
    """Remove import statements for verification-agnostic testing frameworks(import pytest) from the AST.
    This prevents the C++ backend from trying to open JSON files for
    imported testing frameworks that we intentionally skip.
    """
    filtered_body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            # Filter out frameworks
            filtered_names = []
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if not is_testing_framework(base_module):
                    filtered_names.append(alias)
            # If all imports were testing frameworks, skip the entire import statement
            if filtered_names:
                node.names = filtered_names
                filtered_body.append(node)

        elif isinstance(node, ast.ImportFrom):
            # Filter out "from testing_framework import ..." statements
            if node.module:
                base_module = node.module.split(".")[0]
                if not is_testing_framework(base_module):
                    filtered_body.append(node)
            else:
                # Relative import without module (from . import x)
                filtered_body.append(node)
        else:
            filtered_body.append(node)

    tree.body = filtered_body
    return tree

def parse_file(filename: str) -> ast.AST:
    """Open, parse, and run Preprocessor on a Python source file."""
    with open(filename, "r", encoding="utf-8") as src:
        tree = ast.parse(src.read())
    return Preprocessor(filename).visit(tree)


def emit_file_as_json(
    filename: str,
    output_dir: str,
    module_qualname: str | None = None,
    elements_to_import=None,
) -> None:
    """Generate AST JSON for a file."""
    tree = parse_file(filename)
    generate_ast_json(
        tree,
        filename,
        elements_to_import,
        output_dir,
        module_qualname=module_qualname,
    )


def emit_module_json(
    module_qualname: str,
    output_dir: str,
    elements_to_import=None,
) -> None:
    """Resolve module to file and emit AST JSON."""
    filename = resolve_module_file(module_qualname, output_dir)
    if filename:
        emit_file_as_json(filename, output_dir, module_qualname, elements_to_import)


def process_collected_imports(output_dir):
    # Keep processing until no new modules are discovered
    processed_modules = set()

    while True:
        # Get modules that haven't been processed yet
        modules_to_process = set(module_imports.keys()) - processed_modules

        if not modules_to_process:
            break

        for module_name in modules_to_process:
            import_info = module_imports[module_name]
            processed_modules.add(module_name)

            imported_elements = None if import_info['import_all'] \
                else [ast.alias(name, None) for name in import_info['specific_names']]

            # Attempt to resolve and emit JSON for imported submodules (e.g., "pkg.sub")
            if import_info['specific_names']:
                for name in list(import_info['specific_names']):
                    emit_module_json(f"{module_name}.{name}", output_dir)

            # Emit the module itself
            filename = resolve_module_file(module_name, output_dir)
            if not filename:
                continue

            # Parse once, fix relative imports, collect nested imports
            tree = parse_file(filename)
            for subnode in ast.walk(tree):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    rewrite_relative_import(subnode, module_name)
                    process_imports(subnode, output_dir)

            generate_ast_json(tree, filename, imported_elements, output_dir, module_qualname=module_name)


def rewrite_relative_import(node, parent_module: str | None):
    """
    Rewrite a relative ImportFrom node to be absolute.

    Example node structure for "from .x import y":
      node.module = "x"
      node.level = 1

    We need to compute the absolute module path based on
    parent_module and node.level, then set node.module to
    the absolute path and reset node.level to 0.
    """
    # node.level indicates the number of leading dots:
    #   from .x import y  → level = 1
    #   from ..x import y → level = 2
    #   from math import y → level = 0 (not relative)
    lvl = getattr(node, "level", 0)
    if lvl <= 0 or not parent_module:
        # Nothing to fix if it's already absolute or we don't know the parent module
        return

    # Split the parent module name into parts, e.g., "pkg.sub" → ["pkg", "sub"]
    parts = parent_module.split(".")

    # Move up "lvl" levels in the module hierarchy.
    # Example:
    #   parent_module = "l.ks", level = 1 → base = "l"
    #   parent_module = "l", level = 1 → base = "l" (fallback if index <= 0)
    idx = len(parts) - lvl
    base = parent_module if idx <= 0 else ".".join(parts[:idx])

    # Rebuild the full absolute module path.
    # Example:  from .ks import foo  →  from l.ks import foo
    node.module = f"{base}.{node.module}" if node.module else base

    # Reset level to 0 since it's no longer a relative import.
    node.level = 0


def generate_ast_json(tree, python_filename, elements_to_import, output_dir, module_qualname=None):
    """
    Generate AST JSON from the given Python AST tree.

    Parameters:
        - tree: The Python AST tree.
        - python_filename: The filename of the Python source file.
        - elements_to_import: The elements (classes or functions) to be imported from the module.
        - output_dir: The directory to save the generated JSON file.
    """

    # Remove verification-agnostic testing framework imports
    tree = filter_imports(tree)

    # Filter elements to be imported from the module
    filtered_nodes = []
    if elements_to_import is not None and elements_to_import:
        # First pass: collect explicitly imported element names
        explicitly_imported = {elem_info.name for elem_info in elements_to_import}

        # Collect all referenced names (functions and classes) from explicitly imported functions/classes
        referenced_names = set()
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                if node.name in explicitly_imported:
                    referenced_names.update(get_referenced_names(node))

        # Second pass: include explicitly imported items and their referenced functions/classes
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                # Always include ESBMC helper functions
                if node.name in ['ESBMC_range_has_next_', 'ESBMC_range_next_']:
                    filtered_nodes.append(node)
                # Include explicitly imported items
                elif node.name in explicitly_imported:
                    filtered_nodes.append(node)
                # Include functions/classes referenced by imported items
                elif node.name in referenced_names:
                    filtered_nodes.append(node)

            # Include annotated assignments (e.g., x: int = 42)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in explicitly_imported:
                    filtered_nodes.append(node)

    # Convert AST to JSON
    ast2json_module = import_module_by_name("ast2json", "")
    ast_json = ast2json_module.ast2json(ast.Module(body=filtered_nodes) if filtered_nodes else tree)
    ast_json["filename"] = python_filename
    ast_json["ast_output_dir"] = output_dir

     # Build JSON path
    if module_qualname:
        parts = module_qualname.split(".")
        json_dir = os.path.join(output_dir, *parts[:-1])  # package subdirs
        json_filename = os.path.join(json_dir, f"{parts[-1]}.json")
    else:
        if python_filename.endswith('__init__.py'):
            dir_name = os.path.basename(os.path.dirname(python_filename))
            json_filename = os.path.join(output_dir, f"{dir_name}.json")
        else:
            json_filename = os.path.join(
                output_dir, f"{os.path.basename(python_filename[:-3])}.json"
            )

    os.makedirs(os.path.dirname(json_filename), exist_ok=True)

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
                            with open(full_path, "r", encoding="utf-8") as f:
                                tree = ast.parse(f.read())
                                generate_ast_json(tree, full_path, None, output_dir + "/" + base_module)
                        except UnicodeDecodeError:
                            continue

def main():
    check_usage()
    filename = sys.argv[1]
    output_dir = sys.argv[2]

    # Type checking input program with mypy
    result = subprocess.run(
    ["mypy", "--strict", filename],
    capture_output=True,
    text=True)

    if result.returncode != 0:
        print("\033[93m\nType checking warning:\033[0m")
        print(result.stdout)

    # Add the script directory to the front of the import search path
    script_dir = os.path.dirname(os.path.abspath(filename))
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

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
            # Collect import information
            process_imports(node, output_dir)
        elif isinstance(node, ast.Assign):
            # Add type annotation on assignments
            add_type_annotation(node)
        elif isinstance(node, ast.Attribute):
            # Detect and process submodule usage
            detect_and_process_submodules(node, processed_submodules, output_dir)

    # Now process all collected imports once
    process_collected_imports(output_dir)

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
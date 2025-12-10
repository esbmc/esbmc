import ast
from preprocessor import Preprocessor
import sys

code = """
def main() -> None:
    count: int = 10
    count = "wrong"
"""

tree = ast.parse(code)
print("Original AST:")
print(ast.dump(tree, indent=2))

preprocessor = Preprocessor("test_module")
tree = preprocessor.visit(tree)

print("\nProcessed AST:")
print(ast.dump(tree, indent=2))

print("\nNode Traversal:")
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        print(f"Found Assign node: targets={[t.id for t in node.targets if isinstance(t, ast.Name)]}")
    if isinstance(node, ast.AnnAssign):
        print(f"Found AnnAssign node: target={node.target.id if isinstance(node.target, ast.Name) else 'complex'}")

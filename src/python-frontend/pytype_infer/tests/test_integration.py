import ast, os
from pytype_infer import annotate_tree
p = os.path.join(os.path.dirname(__file__), 'example_src.py')
src = open(p).read()
tree = ast.parse(src)
ann = annotate_tree(tree)
try:
    print(ast.unparse(ann))
except Exception:
    import astor
    print(astor.to_source(ann))

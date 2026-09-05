import ast, argparse
from .annotate_ast import annotate_ast

def annotate_tree(tree: ast.Module, filename: str | None = None) -> ast.Module:
    """
    Runs the inference algorithm and injects AnnAssign / FunctionDef.arg.annotation / FunctionDef.returns nodes etc.
    Returns the AST Module.
    """
    return annotate_ast(tree)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', required=True)
    p.add_argument('--out', dest='outfile', required=True)
    args = p.parse_args()
    src = open(args.infile,'r',encoding='utf-8').read()
    tree = ast.parse(src)
    annotated = annotate_tree(tree, filename=args.infile)
    s = ast.unparse(annotated)
    open(args.outfile,'w',encoding='utf-8').write(s)
    print('Wrote', args.outfile)

if __name__ == '__main__':
    main()

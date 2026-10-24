"""Gives `import pkg.sub` an alias the converter resolves (#8975)."""

import ast


def _dotted(node):
    """`a.b.c` for a chain of attribute reads ending in a name, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _binds_otherwise(module, root):
    """Whether anything but an import binds `root`, or a star import may."""
    for node in ast.walk(module):
        if isinstance(node, ast.Name) and node.id == root and not isinstance(node.ctx, ast.Load):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and \
                node.name == root:
            return True
        if isinstance(node, ast.arg) and node.arg == root:
            return True
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" or (a.asname or a.name) == root
                                                    for a in node.names):
            return True
    return False


class _Rename(ast.NodeTransformer):

    def __init__(self, aliases):
        self.aliases = aliases

    def visit_Attribute(self, node):
        alias = self.aliases.get(_dotted(node))
        if alias and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Name(id=alias, ctx=ast.Load()), node)
        return self.generic_visit(node)


def alias_dotted_imports(module):
    """Rewrite a top-level `import pkg.sub` as `import pkg` and
    `import pkg.sub as ESBMC_import_pkg_sub`, reading `pkg.sub` through the
    alias. Python binds `pkg` there too, so the package stays bound."""
    aliases = {}
    body = []
    for stmt in module.body:
        if not isinstance(stmt, ast.Import):
            body.append(stmt)
            continue
        # One alias per statement: the converter misreads `import a, b.c as d`.
        for a in stmt.names:
            root = a.name.split(".")[0]
            if a.asname or "." not in a.name or _binds_otherwise(module, root):
                names = [a]
            else:
                aliases[a.name] = "ESBMC_import_" + a.name.replace(".", "_")
                names = [ast.alias(name=root), ast.alias(name=a.name, asname=aliases[a.name])]
            body += [ast.copy_location(ast.Import(names=[n]), stmt) for n in names]
    if aliases:
        module.body = body
        _Rename(aliases).visit(module)
        ast.fix_missing_locations(module)

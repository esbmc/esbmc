"""Count the places a module binds each name."""

import ast
from collections import Counter


def binding_sites(module_node):
    """How many places anywhere in *module_node* bind or declare each name."""
    sites = Counter()
    for n in ast.walk(module_node):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            sites[n.id] += 1
        elif isinstance(n, ast.alias):
            sites[n.asname or n.name.split(".")[0]] += 1
        elif isinstance(n, ast.arg):
            sites[n.arg] += 1
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            sites.update(n.names)
        elif isinstance(getattr(n, "name", None), str):
            sites[n.name] += 1
        elif isinstance(n, ast.MatchMapping) and n.rest:
            sites[n.rest] += 1
    return sites

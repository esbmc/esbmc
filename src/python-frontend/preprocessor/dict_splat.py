"""Spell a `**` splat of a dict whose keys are known as the keywords it packs
(#9329).

Call binding reads only named keywords, so `inner(**d)` left `inner`'s
parameter unfilled and raised a TypeError CPython does not. A splatted dict
literal with constant str keys becomes those keywords. A name qualifies when
each of its bindings in the scope is such a literal with the same keys and
each of its reads there is a splat, so its keys cannot change; the splat
becomes `key=name["key"]` for each key. Only a call by name of an undecorated
top-level def is rewritten, in the module or a def; a splat elsewhere, as in a
method call, a class body or a lambda, is left to the converter.
"""

import ast

from preprocessor.binding_sites import binding_sites

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _literal_keys(node):
    """The keys of dict literal `node`, when all are distinct constant strs."""
    if not isinstance(node, ast.Dict):
        return None
    keys = [
        k.value if isinstance(k, ast.Constant) and isinstance(k.value, str) else None
        for k in node.keys
    ]
    if None in keys or len(set(keys)) != len(keys):
        return None
    return keys


def _own_nodes(body):
    """The nodes of a scope, without the bodies of the scopes nested in it."""
    stack = list(body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _nested_names(body):
    """The names read or bound in the scopes nested in this one."""
    return {
        n.id
        for node in _own_nodes(body) if isinstance(node, _SCOPES) for n in ast.walk(node)
        if isinstance(n, ast.Name)
    }


def _parameters(scope):
    if isinstance(scope, ast.Module):
        return set()
    args = scope.args
    return {
        a.arg
        for a in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]
        if a is not None
    }


def _other_bindings(body):
    """The names this scope binds other than by assigning to a Name: a def, a
    class, an `except ... as`, a match capture or an import."""
    names = set()
    for node in _own_nodes(body):
        if isinstance(getattr(node, "name", None), str):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


def _assigned_literals(body):
    """Per Name a single-target assignment in this scope stores to, the keys
    of the dict literal it stores, or None."""
    return {
        id(node.targets[0]): _literal_keys(node.value)
        for node in _own_nodes(body) if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


def _dict_names(scope, declared):
    """Per name the keys its bindings in `scope` give it, for the names that
    qualify."""
    body = scope.body
    literals = _assigned_literals(body)
    splats = {
        id(node.value)
        for node in _own_nodes(body) if isinstance(node, ast.keyword) and node.arg is None
    }
    refused = _parameters(scope) | declared | _nested_names(body) | _other_bindings(body)
    keys = {}
    for node in _own_nodes(body):
        if not isinstance(node, ast.Name) or node.id in refused:
            continue
        if isinstance(node.ctx, ast.Load):
            qualifies = id(node) in splats
        else:
            found = literals.get(id(node))
            qualifies = found is not None and keys.setdefault(node.id, found) == found
        if not qualifies:
            refused.add(node.id)
    return {name: found for name, found in keys.items() if name not in refused}


def _accepts(func_def, keys):
    """Whether `func_def` takes each of `keys` as a keyword. Another key is a
    TypeError the converter does not raise for a spelled-out keyword."""
    args = func_def.args
    if set(keys) & {a.arg for a in args.posonlyargs}:
        return False
    return args.kwarg is not None or set(keys) <= {a.arg for a in args.args + args.kwonlyargs}


def _keywords(kw, names, func_def):
    """The keywords spelling out splat `kw`, or None to keep it."""
    value = kw.value
    if isinstance(value, ast.Dict):
        keys = _literal_keys(value)
        if keys is None or not _accepts(func_def, keys):
            return None
        pairs = zip(keys, value.values)
    elif (isinstance(value, ast.Name) and value.id in names
          and _accepts(func_def, names[value.id])):
        pairs = ((key,
                  ast.Subscript(value=ast.Name(id=value.id, ctx=ast.Load()),
                                slice=ast.Constant(value=key),
                                ctx=ast.Load())) for key in names[value.id])
    else:
        return None
    spelled = []
    for key, item in pairs:
        new = ast.keyword(arg=key, value=item)
        ast.copy_location(new, kw)
        ast.fix_missing_locations(new)
        spelled.append(new)
    return spelled


def _spell_scope(scope, declared, defs):
    names = _dict_names(scope, declared)
    for node in _own_nodes(scope.body):
        if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
                or node.func.id not in defs or all(kw.arg is not None for kw in node.keywords)):
            continue
        keywords = []
        for kw in node.keywords:
            spelled = _keywords(kw, names, defs[node.func.id]) if kw.arg is None else None
            keywords.extend([kw] if spelled is None else spelled)
        node.keywords = keywords


def _module_functions(module):
    """The undecorated top-level defs whose name nothing else binds."""
    sites = binding_sites(module)
    return {
        s.name: s
        for s in module.body
        if isinstance(s, ast.FunctionDef) and not s.decorator_list and sites[s.name] == 1
    }


def spell_out_dict_splats(module):
    """Spell out each qualifying `**` splat in `module`."""
    declared = {
        name
        for node in ast.walk(module) if isinstance(node, (ast.Global, ast.Nonlocal))
        for name in node.names
    }
    defs = _module_functions(module)
    for scope in [module] + [
            n for n in ast.walk(module) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]:
        _spell_scope(scope, declared, defs)

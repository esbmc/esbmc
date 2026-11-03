"""Answer `getattr` with a default and `hasattr` on an attribute that a scope
adds to its own instance on only some paths (#8964).

The converter refuses such a store: the instance's struct cannot tell a path
that set the attribute from one that did not. When the attribute is stored
and read nowhere else, and only through `getattr` with a default or
`hasattr`, its value and whether it is set are two local variables instead:

    b = B()                      b = B()
                                 esbmc_has_b_code = False
    if c:                        if c:
        b.code = 1                   esbmc_b_code = 1
                                     esbmc_has_b_code = True
    getattr(b, "code", 0)        esbmc_b_code if esbmc_has_b_code else 0
    hasattr(b, "code")           esbmc_has_b_code

The receiver must be a name bound once, by constructing a class of the
module, in the one scope holding every use of the attribute. Anything that
could reach the attribute another way (a hook, an import, a dynamic name)
leaves the program unchanged.
"""

import ast
import builtins

_HOOKS = {
    "setattr", "delattr", "vars", "dir", "globals", "locals", "eval", "exec", "__dict__",
    "__class__", "__slots__", "__getattr__", "__getattribute__", "__setattr__", "__delattr__"
}
_SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
           ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _scopes(module):
    """The innermost scope of every node in `module`, by node id."""
    owner = {}
    stack = [(module, module)]
    while stack:
        node, scope = stack.pop()
        owner[id(node)] = scope
        inner = node if isinstance(node, _SCOPES) else scope
        stack.extend((child, inner) for child in ast.iter_child_nodes(node))
    return owner


def _bindings(module, name):
    """The nodes of `module` that bind `name`."""
    found = []
    for n in ast.walk(module):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load) and n.id == name:
            found.append(n)
        elif isinstance(n,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            found.append(n)
        elif isinstance(n, ast.arg) and n.arg == name:
            found.append(n)
        elif isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
            found.append(n)
    return found


def _builtin_lacks(base, attr):
    cls = getattr(builtins, base, None)
    return (base == "object"
            or isinstance(cls, type) and issubclass(cls, BaseException)) and not hasattr(cls, attr)


def _class_lacks(classes, name, attr):
    """Whether class `name` of the module, and each class it derives from,
    leaves `attr` to its instances."""
    cls = classes.get(name)
    if cls is None or cls.decorator_list or cls.keywords:
        return False
    if any(
            isinstance(n, ast.Name) and n.id == attr or isinstance(n, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == attr
            for n in ast.walk(cls)):
        return False
    return all(
        isinstance(b, ast.Name) and
        (_builtin_lacks(b.id, attr) if b.id not in classes else _class_lacks(classes, b.id, attr))
        for b in cls.bases)


class _Uses:
    """Every use of one attribute name, or `ok` False when one is unsupported."""

    def __init__(self):
        self.ok = True
        self.receivers = set()
        self.nodes = []
        self.stores = []
        self.calls = []

    def add(self, receiver, node):
        if not isinstance(receiver, ast.Name):
            self.ok = False
            return
        self.receivers.add(receiver.id)
        self.nodes.append(node)


def _store_target(node):
    """The attribute a plain `obj.attr = value` statement stores, or None."""
    targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
    if (isinstance(node, (ast.Assign, ast.AnnAssign)) and len(targets) == 1
            and isinstance(targets[0], ast.Attribute) and node.value is not None):
        return targets[0]
    return None


def _call_attribute(node):
    """The attribute a `getattr` or `hasattr` call names: None when `node` is
    no such call, "" when the call does not name it by a string literal."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("getattr", "hasattr")):
        return None
    name = node.args[1] if len(node.args) > 1 else None
    return name.value if isinstance(name, ast.Constant) and isinstance(name.value, str) else ""


def _supported_call(node):
    """Whether `node` is `getattr(obj, name, default)` with a default that
    cannot fail to evaluate, or `hasattr(obj, name)`."""
    if node.keywords:
        return False
    if node.func.id == "hasattr":
        return len(node.args) == 2
    return len(node.args) == 3 and isinstance(node.args[2], (ast.Constant, ast.Name))


def _uses_by_attribute(module):
    """Each attribute name's uses, or None when a use names it dynamically."""
    uses = {}
    stores = set()
    for n in ast.walk(module):
        target = _store_target(n)
        if target is not None:
            stores.add(id(target))
            uses.setdefault(target.attr, _Uses()).stores.append(n)
    for n in ast.walk(module):
        attr = _call_attribute(n)
        if attr == "":
            return None
        if isinstance(n, ast.Attribute):
            use = uses.setdefault(n.attr, _Uses())
            use.ok &= id(n) in stores
            use.add(n.value, n)
        elif attr is not None:
            use = uses.setdefault(attr, _Uses())
            use.ok &= _supported_call(n)
            use.add(n.args[0], n)
            use.calls.append(n)
    return uses


class _Rewrite(ast.NodeTransformer):

    def __init__(self, calls, has, value):
        self.calls = {id(c) for c in calls}
        self.has = has
        self.value = value

    def visit_Call(self, node):
        self.generic_visit(node)
        if id(node) not in self.calls:
            return node
        has = ast.Name(id=self.has, ctx=ast.Load())
        if node.func.id == "hasattr":
            return ast.copy_location(has, node)
        value = ast.Name(id=self.value, ctx=ast.Load())
        return ast.copy_location(ast.IfExp(test=has, body=value, orelse=node.args[2]), node)


def _set(name, value, at):
    return ast.copy_location(
        ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Constant(value=value)),
        at)


def _insert_after(module, stmt, extra):
    for node in ast.walk(module):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and any(s is stmt for s in value):
                index = next(i for i, s in enumerate(value) if s is stmt)
                value[index + 1:index + 1] = [extra]
                return


def _lower(module, use, binding, has, value):
    for store in use.stores:
        name = ast.Name(id=value, ctx=ast.Store())
        if isinstance(store, ast.Assign):
            store.targets = [ast.copy_location(name, store.targets[0])]
        else:
            store.target = ast.copy_location(name, store.target)
            store.simple = 1
        _insert_after(module, store, _set(has, True, store))
    _insert_after(module, binding, _set(has, False, binding))
    _Rewrite(use.calls, has, value).visit(module)


def _mentioned(module):
    """Every name and attribute name `module` mentions."""
    return {
        s
        for n in ast.walk(module)
        for s in (getattr(n, "id", None), getattr(n, "attr", None)) if isinstance(s, str)
    }


def _reachable_otherwise(module, mentioned):
    """Whether something other than the nodes this pass rewrites may reach an
    attribute: another module, a hook, a class pattern, or a rebound getattr
    or hasattr."""
    return (any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Match)) for n in ast.walk(module))
            or bool(mentioned & _HOOKS) or bool(_bindings(module, "getattr"))
            or bool(_bindings(module, "hasattr")))


def _binding(module, use, scopes):
    """The statement binding the one receiver of `use`, when one statement
    binds it, in the function or module scope of every use, and a store of
    the attribute sits in a block nested in that scope."""
    if not use.ok or not use.stores or not use.calls or len(use.receivers) != 1:
        return None
    scope = {scopes[id(n)] for n in use.nodes}
    bound = _bindings(module, next(iter(use.receivers)))
    if len(scope) != 1 or len(bound) != 1 or scopes[id(bound[0])] not in scope:
        return None
    scope = scope.pop()
    if not isinstance(scope,
                      (ast.Module, ast.FunctionDef)) or all(s in scope.body for s in use.stores):
        return None
    return next(
        (s for s in ast.walk(scope) if isinstance(s, ast.Assign) and s.targets == [bound[0]]), None)


def lower_path_dependent_attributes(module):
    """Rewrite each attribute that qualifies, as the module docstring shows."""
    mentioned = _mentioned(module)
    if _reachable_otherwise(module, mentioned):
        return
    uses = _uses_by_attribute(module)
    if uses is None:
        return
    classes = {s.name: s for s in module.body if isinstance(s, ast.ClassDef)}
    scopes = _scopes(module)
    for attr, use in uses.items():
        binding = _binding(module, use, scopes)
        if binding is None:
            continue
        receiver = binding.targets[0].id
        has, value = f"esbmc_has_{receiver}_{attr}", f"esbmc_{receiver}_{attr}"
        call = binding.value
        cls = call.func.id if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) else ""
        if (len(_bindings(module, cls)) != 1 or not _class_lacks(classes, cls, attr)
                or {has, value} & mentioned):
            continue
        _lower(module, use, binding, has, value)

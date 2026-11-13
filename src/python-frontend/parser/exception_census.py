"""Exception-site census for the Python frontend (`--python-exception-census`).

Question answered: can any construct of this program, or of a module it imports, raise an
exception of the checked family?  If not, the property "no uncaught exception of the family"
holds whatever else the program does.

Everything is an allowlist.  A node type, builtin, attribute name, import or dunder that is not
known safe for the family is reported, and a single report withholds the verdict.  Reports carry
a class:

  family    the construct may raise the family
  unknown   the construct is outside what the census can reason about
  overflow  OverflowError-class site.  OverflowError is outside both families; these are
            reported separately so that their cost can be read off.

Nothing here executes the program.  Any failure of the census answers "no proof": see
`census_summary`.
"""
# pylint: disable=missing-function-docstring,invalid-name,too-many-lines
from __future__ import annotations

import ast
import importlib
import os
import sys

if __package__:
    T = importlib.import_module(f"{__package__}.exception_census_tables")
else:
    T = importlib.import_module("exception_census_tables")

BUILTIN_NAMES = frozenset(dir(__import__("builtins")))
BIG = 2**63


class Unfoldable(Exception):
    """A constant expression that is too large to compute or that raises."""

    def __init__(self, huge=False):
        super().__init__()
        self.huge = huge


class Scope:
    """The names bound in one function, class, comprehension or module scope."""

    __slots__ = ("kind", "local", "globals", "nonlocals")

    def __init__(self, kind, local=(), globals_=(), nonlocals=()):
        self.kind = kind
        self.local = set(local)
        self.globals = set(globals_)
        self.nonlocals = set(nonlocals)


class Binder(ast.NodeVisitor):
    """Names bound directly in one scope, without entering nested scopes."""

    def __init__(self):
        self.names, self.globals, self.nonlocals, self.stars = set(), set(), set(), []

    def visit_Name(self, n):
        if isinstance(n.ctx, (ast.Store, ast.Del)):
            self.names.add(n.id)

    def visit_FunctionDef(self, n):
        self.names.add(n.name)

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_ClassDef(self, n):
        self.names.add(n.name)

    def visit_Lambda(self, n):
        pass

    def _comp(self, n):
        """Walrus targets bind in the enclosing scope, except inside a nested lambda."""
        todo = [n]
        while todo:
            sub = todo.pop()
            if isinstance(sub, ast.Lambda):
                continue
            if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                self.names.add(sub.target.id)
            todo.extend(ast.iter_child_nodes(sub))

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp  # noqa: N815

    def visit_Import(self, n):
        for a in n.names:
            self.names.add(a.asname or a.name.split(".")[0])

    def visit_ImportFrom(self, n):
        for a in n.names:
            if a.name == "*":
                self.stars.append(n)
            else:
                self.names.add(a.asname or a.name)

    def visit_ExceptHandler(self, n):
        if n.name:
            self.names.add(n.name)
        self.generic_visit(n)

    def visit_Global(self, n):
        self.globals.update(n.names)

    def visit_Nonlocal(self, n):
        self.nonlocals.update(n.names)

    def visit_MatchAs(self, n):
        if getattr(n, "name", None):
            self.names.add(n.name)
        self.generic_visit(n)

    visit_MatchStar = visit_MatchAs  # noqa: N815


def bound_in(nodes):
    b = Binder()
    for n in nodes:
        b.visit(n)
    return b


def param_names(args):
    out = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    if args.vararg:
        out.append(args.vararg.arg)
    if args.kwarg:
        out.append(args.kwarg.arg)
    return out


_FOLD = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
    ast.LShift: lambda a, b: a << b,
}


def _too_big(op, a, b):
    """Whether folding `a op b` would build an integer too large to be worth computing."""
    if not (isinstance(a, int) and isinstance(b, int)):
        return isinstance(op, ast.LShift)
    if isinstance(op, ast.Pow):
        return abs(a) > 1 and b > 4096
    if isinstance(op, ast.LShift):
        return b > 4096
    return isinstance(op, ast.Mult) and a.bit_length() + b.bit_length() > 100000


def bound_names(node):
    """The names `node` binds in its own scope, in the order they are written."""
    if isinstance(node, ast.Name):
        return [node.id] if isinstance(node.ctx, (ast.Store, ast.Del)) else []
    if isinstance(node, ast.FunctionDef):
        return [node.name] + param_names(node.args)
    if isinstance(node, ast.ClassDef):
        return [node.name]
    if isinstance(node, ast.Lambda):
        return param_names(node.args)
    if isinstance(node, ast.ExceptHandler):
        return [node.name] if node.name else []
    if isinstance(node, ast.Import):
        return [a.asname or a.name.split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [a.asname or a.name for a in node.names]
    return []


def class_attr_names(node):
    """Attribute names a class body defines: methods, nested classes and class variables."""
    names = set()
    for st in node.body:
        if isinstance(st, (ast.FunctionDef, ast.ClassDef)):
            names.add(st.name)
        elif isinstance(st, ast.Assign):
            names.update(sub.id for t in st.targets for sub in ast.walk(t)
                         if isinstance(sub, ast.Name))
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            names.add(st.target.id)
    return names


class Module:
    """One parsed source file the program executes."""

    def __init__(self, path, tree, audited=False):
        self.path, self.tree, self.audited = path, tree, audited
        self.bound = set()  # module-level bindings
        self.module_names = set()  # names that hold a module object
        self.aliases = {}  # name -> local Module bound by a module-level import
        self.name_counts = {}


class Census:
    """Walks a program and the local modules it imports, collecting reports."""

    def __init__(self, family, main, roots=None, roots_mode="env"):
        self.family = family
        self.main = os.path.realpath(main)
        self.roots = roots or self.default_roots(roots_mode)
        self.mods = {}
        self.reports = []
        self.safe_attrs = T.SAFE_ATTRS[family]
        self.safe_builtins = T.SAFE_BUILTINS[family]
        self.arith = family == "arithmetic"
        self.user_attrs = set()
        self.user_classes = set()
        self.lib_class_aliases = set()
        self.from_imports = {}  # local name -> [(module, imported name)]
        self.module_imports = {}  # local name -> module it names, for `import M [as N]`
        self.user_attr_stores = set()
        self.nonimport_bound = set()
        self.typing_aliases = set()
        self.bind_count = {}
        self.const_names = {}
        self.resolved = {}

    def report(self, cls, path, line, key):
        if len(self.reports) < 200:
            rel = os.path.relpath(path, os.path.dirname(self.main)) if path else ""
            self.reports.append((cls, rel, line, key))

    # ---- module discovery ----
    def default_roots(self, mode):
        """Directories CPython searches for the program's imports, in order.

        mode "env": the script directory, then PYTHONPATH (the SV-COMP wrapper
        puts the task directory, its parent and the directory holding
        _sv_verifier.py there).  mode "dir": the script directory alone.
        """
        roots = [os.path.dirname(self.main)]
        if mode == "env":
            # An empty entry of a non-empty PYTHONPATH is the working directory.
            raw = os.environ.get("PYTHONPATH", "")
            roots += [os.path.abspath(p) for p in raw.split(os.pathsep)] if raw else []
        return list(dict.fromkeys(roots))

    def shadow_check(self):
        """A local file named like a stdlib module is imported ahead of it, by the stdlib itself
        as well, so nothing the stdlib does can be trusted.  Compiled files count too: an
        extension module or a .pyc with an unchecked hash does not have to match any source."""
        watched = T.STDLIB_NAMES | T.HIJACK_NAMES | {"_sv_verifier"}
        for r in self.roots:
            if os.path.exists(r) and not os.path.isdir(r):
                # An archive on the path (zipimport) holds modules the census cannot list.
                self.report("unknown", r, 0, "path-entry-not-a-directory")
            for sub in ("", "__pycache__"):
                for n in sorted(self.listing(os.path.join(r, sub))):
                    stem = n.split(".")[0]
                    if stem in watched and self.shadows(r, sub, n, stem):
                        self.report("unknown", os.path.join(r, sub, n), 0, "shadows-stdlib:" + stem)

    @staticmethod
    def shadows(root, sub, name, stem):
        if name.endswith((".pyc", ".so", ".pyd")):
            return True
        if sub or stem == "_sv_verifier":
            return False
        return name.endswith(".py") or os.path.isfile(os.path.join(root, name, "__init__.py"))

    @staticmethod
    def listing(directory):
        try:
            return set(os.listdir(directory))
        except OSError:
            return set()

    @staticmethod
    def compiled_variants(directory, stem):
        """Compiled forms of module `stem` in `directory`, `__pycache__` included."""
        found = []
        for sub in ("", "__pycache__"):
            path = os.path.join(directory, sub)
            found += [
                os.path.join(sub, f) for f in sorted(Census.listing(path))
                if f.startswith(stem + ".") and f.endswith((".pyc", ".so", ".pyd"))
            ]
        return found

    def check_compiled(self, directory, stem):
        for f in self.compiled_variants(directory, stem):
            self.report("unknown", os.path.join(directory, f), 0, "compiled-module:" + stem)

    @staticmethod
    def find_in_dir(d, name):
        p = os.path.join(d, name)
        if os.path.isfile(os.path.join(p, "__init__.py")):
            return ("pkg", p)
        if os.path.isfile(p + ".py"):
            return ("mod", p + ".py")
        if os.path.isdir(p):
            return ("ns", p)
        return None

    def find_top(self, name, dirs):
        ns = None
        for r in dirs:
            f = self.find_in_dir(r, name)
            if f and f[0] != "ns":
                return f
            ns = ns or f
        return ns

    def local_chain(self, parts, dirs):
        """(files executed, last directory) for a dotted import from local
        sources, or None when the first part is not local."""
        cur = self.find_top(parts[0], dirs)
        # A directory without __init__.py loses to any regular module of that name later on
        # the path, which the census cannot see; only a relative import is certain of it.
        if cur is None or (cur[0] == "ns" and dirs is self.roots):
            return None
        files, curdir = [], None
        for i, part in enumerate(parts):
            if i:
                cur = self.find_in_dir(curdir, part)
                if cur is None:
                    break
            kind, p = cur
            if kind == "pkg":
                files.append(os.path.join(p, "__init__.py"))
                self.check_compiled(p, "__init__")
                curdir = p
            elif kind == "mod":
                files.append(p)
                self.check_compiled(os.path.dirname(p), os.path.basename(p)[:-3])
                break
            else:
                curdir = p
        return files, curdir

    def import_targets(self, mod, node):
        """Local files an import statement executes; [] for a library import.
        For `from M import *` the last entry is M itself."""
        key = (mod.path, id(node))
        if key not in self.resolved:
            if isinstance(node, ast.Import):
                self.resolved[key] = self.import_files(node)
            else:
                self.resolved[key] = self.from_import_files(mod, node)
        return self.resolved[key]

    def import_files(self, node):
        out = []
        for alias in node.names:
            found = self.local_chain(alias.name.split("."), self.roots)
            if found:
                out += found[0]
        return out

    def from_import_files(self, mod, node):
        dirs = self.roots
        if node.level:
            base = os.path.dirname(mod.path)
            for _ in range(node.level - 1):
                base = os.path.dirname(base)
            dirs = [base]
        out, curdir = [], dirs[0]
        if node.module:
            found = self.local_chain(node.module.split("."), dirs)
            curdir = found[1] if found else None
            out += found[0] if found else []
        for alias in node.names if curdir else ():
            found = self.find_in_dir(curdir, alias.name) if alias.name != "*" else None
            if found and found[0] == "pkg":
                out.append(os.path.join(found[1], "__init__.py"))
            elif found and found[0] == "mod":
                out.append(found[1])
        return out

    def load(self, path):
        if path in self.mods:
            return self.mods[path]
        try:
            with open(path, "rb") as fh:
                data = fh.read()
            tree = ast.parse(data, filename=path)
        except (SyntaxError, ValueError, RecursionError, MemoryError, OSError) as e:
            self.report("unknown", path,
                        getattr(e, "lineno", 0) or 0, "unparsable:" + type(e).__name__)
            self.mods[path] = None
            return None
        audited = False
        if os.path.basename(path) == "_sv_verifier.py":
            # Imported here: shadow_check has by now declined if a file next to the program
            # would stand in for hashlib.
            import hashlib  # pylint: disable=import-outside-toplevel
            audited = hashlib.sha256(data).hexdigest() in T.AUDITED_SV_VERIFIER
            if not audited:
                self.report("unknown", path, 0, "sv-verifier-not-audited")
        m = self.mods[path] = Module(path, tree, audited)
        return m

    def discover(self):
        todo, seen = [self.main], set()
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            m = self.load(p)
            if m is None or m.audited:
                continue
            for n in ast.walk(m.tree):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    targets = self.import_targets(m, n)
                    if self.main in targets:
                        # The program's own module, a second time and under another name.
                        self.report("unknown", m.path, n.lineno, "self-import")
                    todo += targets

    # ---- program-wide facts ----
    def count(self, name):
        self.bind_count[name] = self.bind_count.get(name, 0) + 1

    def prepass_node(self, m, node):
        names = bound_names(node)
        for name in names:
            self.count(name)
            m.name_counts[name] = m.name_counts.get(name, 0) + 1
        if isinstance(node, ast.Import):
            self.record_import(m, node, names)
        elif isinstance(node, ast.ImportFrom):
            self.record_from_import(node, names)
        else:
            self.nonimport_bound.update(names)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            self.user_attrs.add(node.attr)
            self.user_attr_stores.add(node.attr)
        if isinstance(node, ast.ClassDef):
            self.user_attrs |= class_attr_names(node)
            self.user_classes.add(node.name)

    def record_import(self, m, node, names):
        m.module_names.update(names)
        for alias, name in zip(node.names, names):
            self.module_imports[name] = alias.name if alias.asname else name
            if alias.name in T.TYPING_MODULES:
                self.typing_aliases.add(name)
            else:
                self.nonimport_bound.add(name)

    def record_from_import(self, node, names):
        for alias in node.names:
            if not node.level:
                entry = (node.module or "", alias.name)
                self.from_imports.setdefault(alias.asname or alias.name, []).append(entry)
        self.lib_class_aliases.update(a.asname for a in node.names
                                      if a.asname and a.name in T.LIBRARY_PY_CLASSES | T.ABC_MIXINS)
        if (node.module or "") not in T.TYPING_MODULES:
            self.nonimport_bound.update(names)

    def link_aliases(self, m):
        """Names bound by module-level imports of local modules."""
        for node in m.tree.body:
            if isinstance(node, ast.Import):
                self.link_import(m, node)
            elif isinstance(node, ast.ImportFrom):
                self.link_from_import(m, node)

    def link_import(self, m, node):
        for alias in node.names:
            found = self.local_chain(alias.name.split("."), self.roots)
            if found and found[0]:
                target = self.mods.get(found[0][-1] if alias.asname else found[0][0])
                if target is not None:
                    m.aliases[alias.asname or alias.name.split(".")[0]] = target

    def link_from_import(self, m, node):
        targets = self.import_targets(m, node)
        for alias in node.names:
            if self.package_binds(targets, alias.name):
                continue
            for path in targets:
                stem = os.path.splitext(os.path.basename(path))[0]
                if stem == "__init__":
                    stem = os.path.basename(os.path.dirname(path))
                if stem == alias.name and self.mods.get(path) is not None:
                    m.aliases[alias.asname or alias.name] = self.mods[path]
                    m.module_names.add(alias.asname or alias.name)

    def check_init_submodules(self, m):
        """A package's __init__ that binds the name of one of its submodules makes
        `pkg.sub` the binding, not the file."""
        if os.path.basename(m.path) != "__init__.py":
            return
        folder = os.path.dirname(m.path)
        for entry in sorted(self.listing(folder)):
            stem = entry[:-3] if entry.endswith(".py") else entry
            found = self.find_in_dir(folder, stem) if stem != "__init__" else None
            if found and found[0] != "ns" and stem in m.bound:
                self.report("unknown", m.path, 0, "init-binds-submodule:" + stem)

    def package_binds(self, targets, name):
        """Whether a package's __init__ binds `name`, which then beats a submodule of that name."""
        return any(
            self.mods.get(f) is not None and name in self.mods[f].bound for f in targets
            if os.path.basename(f) == "__init__.py")

    def prepass(self):
        live = [m for m in self.mods.values() if m is not None and not m.audited]
        for m in live:
            b = bound_in(m.tree.body)
            m.bound = set(b.names)
            for node in ast.walk(m.tree):
                if isinstance(node, ast.Global):
                    m.bound.update(node.names)
            for st in b.stars:
                # Names it binds are not known to the census, nor which of them it rebinds.
                self.report("unknown", m.path, st.lineno, "star-import")
            self.check_init_submodules(m)
            self.user_attrs |= m.bound
            for node in ast.walk(m.tree):
                self.prepass_node(m, node)
        for m in live:
            self.link_aliases(m)
        for m in live:
            self.find_consts(m)

    def find_consts(self, m):
        for st in m.tree.body:
            if (isinstance(st, ast.Assign) and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Name)
                    and self.bind_count.get(st.targets[0].id) == 1
                    and st.targets[0].id not in self.user_attr_stores):
                try:
                    v = self.numconst(st.value, names=False)
                except Unfoldable:
                    continue
                if v is not None:
                    self.const_names[(m.path, st.targets[0].id)] = v

    # ---- constant folding ----
    def numconst(self, n, names=True, mod=None):
        """Value of a numeric constant expression, None if not constant.
        Raises Unfoldable when the value is too large to compute."""
        if isinstance(n, ast.Constant):
            return n.value if isinstance(n.value, (bool, int, float, complex)) else None
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = self.numconst(n.operand, names, mod)
            if v is None:
                return None
            return -v if isinstance(n.op, ast.USub) else +v
        if isinstance(n, ast.BinOp):
            left = self.numconst(n.left, names, mod)
            right = self.numconst(n.right, names, mod) if left is not None else None
            return self.fold(n.op, left, right)
        if names and isinstance(n, ast.Name) and mod is not None:
            return self.const_names.get((mod, n.id))
        return None

    @staticmethod
    def fold(op, a, b):
        """a op b for numeric constants, None when either is unknown."""
        if a is None or b is None or type(op) not in _FOLD:
            return None
        if _too_big(op, a, b):
            raise Unfoldable(huge=True)
        try:
            return _FOLD[type(op)](a, b)
        except OverflowError as exc:
            raise Unfoldable(huge=True) from exc
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise Unfoldable() from exc

    # ---- driver ----
    def run(self):
        """Walk the program; reports are left in self.reports."""
        self.shadow_check()
        if self.reports:
            return False
        self.discover()
        self.prepass()
        for m in list(self.mods.values()):
            if m is not None and not m.audited:
                Checker(self, m).check()
        return self.verdict()

    def verdict(self, waive_overflow=False):
        return not [r for r in self.reports if not (waive_overflow and r[0] == "overflow")]


class Checker(ast.NodeVisitor):
    """Applies the allowlists to one module; every node type outside them is reported."""

    def __init__(self, census, mod):
        self.c, self.m = census, mod
        self.scopes = [Scope("module")]
        self.in_annotation = 0
        self.type_ok = set()
        self.class_ok = set()
        self.attr_base = set()

    def rej(self, cls, node, key):
        self.c.report(cls, self.m.path, getattr(node, "lineno", 0), key)

    def check(self):
        self.visit(self.m.tree)

    def visit(self, node):
        t = type(node).__name__
        if t not in T.ALLOWED_NODES:
            self.rej("family" if t == "Raise" else "unknown", node, "node:" + t)
            return None
        return super().visit(node)

    # ---- scopes ----
    def fast_bound(self, name):
        """Bound in an enclosing function or comprehension: such a load never
        falls back to builtins, unlike a module or class level load."""
        top = self.scopes[-1]
        if top.kind == "class" and (name in top.local or name in top.globals):
            return False
        for s in reversed(self.scopes):
            if s.kind in ("func", "comp"):
                if name in s.globals:
                    return False
                if name in s.local or name in s.nonlocals:
                    return True
        return False

    def push_func(self, params, body_nodes):
        b = bound_in(body_nodes)
        self.scopes.append(
            Scope("func", (b.names | set(params)) - b.globals - b.nonlocals, b.globals,
                  b.nonlocals))

    # ---- names and attributes ----
    def library_class_use(self, n, name):
        """A library class written in Python is only constructed, inherited from or named in a
        type: used as a value, `Counter.update(foreign, ...)` runs its code on an object of the
        program's choosing."""
        named = name in T.LIBRARY_PY_CLASSES or name in self.c.lib_class_aliases
        if named and not self.in_annotation and id(n) not in self.class_ok:
            self.rej("unknown", n, "library-class-value:" + name)

    def module_value_use(self, n, is_module):
        """A module object is only ever the base of an attribute read.  Held in a variable or
        passed on, it could be stored into from code the census does not trace to it, and a
        store is how a name it trusts gets rebound."""
        if is_module and id(n) not in self.attr_base:
            self.rej("unknown", n, "module-value")

    def reads_module_name(self, n):
        """Whether `n` reads, from a followed local module, a name that holds a module."""
        base = self.module_of(n.value)
        return base is not None and n.attr in base.module_names

    def module_store(self, n):
        """An attribute store, or del, whose base chain starts at a module name."""
        base = n.value
        while isinstance(base, ast.Attribute):
            base = base.value
        if (isinstance(base, ast.Name) and base.id in self.m.module_names
                and not self.fast_bound(base.id)):
            self.rej("unknown", n, "module-attribute-store")

    def abc_mixin_use(self, n, name):
        """The collections.abc mixins hand the program's object to C code through unbound calls."""
        if (name in T.ABC_MIXINS or name in self.c.lib_class_aliases) and not self.in_annotation:
            self.rej("family", n, "abc-mixin:" + name)

    def visit_Name(self, n):
        name = n.id
        if isinstance(n.ctx, (ast.Store, ast.Del)):
            if name.startswith("__") and name.endswith("__") and name not in T.SAFE_DUNDER_STORES:
                self.rej("unknown", n, "store-dunder:" + name)
        elif not self.fast_bound(name):
            self.module_value_use(n, name in self.m.module_names)
            self.library_class_use(n, name)
            self.abc_mixin_use(n, name)
            self.resolve_load(n, name)

    def resolve_load(self, n, name):
        """A load that no function or comprehension scope binds: a builtin, a module-level
        name, a class-body name, or nothing the census knows."""
        if name in BUILTIN_NAMES:
            if name not in self.c.safe_builtins:
                self.rej("family" if name in ("pow", "divmod") else "unknown", n, "builtin:" + name)
            return
        in_class_body = self.scopes[-1].kind == "class" and name in self.scopes[-1].local
        if not (name in self.m.bound or name in T.INTRINSICS or in_class_body):
            self.rej("unknown", n, "unresolved-name:" + name)

    def module_of(self, e):
        if isinstance(e, ast.Name):
            if self.fast_bound(e.id) or self.m.name_counts.get(e.id) != 1:
                return None
            return self.m.aliases.get(e.id)
        if isinstance(e, ast.Attribute) and isinstance(e.ctx, ast.Load):
            base = self.module_of(e.value)
            if (base is not None and e.attr not in self.c.user_attr_stores
                    and os.path.basename(base.path) == "__init__.py"):
                f = Census.find_in_dir(os.path.dirname(base.path), e.attr)
                if f and f[0] == "pkg":
                    return self.c.mods.get(os.path.join(f[1], "__init__.py"))
                if f and f[0] == "mod":
                    return self.c.mods.get(f[1])
        return None

    def local_module_attribute(self, n):
        """Whether `n` reads a name a followed local module defines."""
        base = self.module_of(n.value) if isinstance(n.ctx, ast.Load) else None
        if base is None or (n.attr.startswith("__") and n.attr.endswith("__")):
            return False
        names = T.SV_VERIFIER_EXPORTS if base.audited else base.bound
        is_package = os.path.basename(base.path) == "__init__.py"
        return n.attr in names or bool(is_package
                                       and Census.find_in_dir(os.path.dirname(base.path), n.attr))

    @staticmethod
    def dunder_report(n):
        allowed = isinstance(n.ctx, ast.Load) and n.attr in ("__init__", "__name__")
        return None if allowed else ("unknown", "dunder-attr:" + n.attr)

    @staticmethod
    def store_report(a):
        """A name the library knows may be a library module's or class's own."""
        known = a in T.UNIVERSE or a in T.STDLIB_NAMES
        return ("unknown", "store-universe-attr:" + a) if known else None

    def attribute_report(self, n):
        """(class, key) of the reason to decline attribute `n`, None when it is allowed."""
        a = n.attr
        if a.startswith("__") and a.endswith("__"):
            return self.dunder_report(n)
        if isinstance(n.ctx, ast.Store):
            return self.store_report(a)
        if a in self.c.safe_attrs:
            flagged = self.c.arith and a in T.OVERFLOW_ATTRS
            return ("overflow", "overflow-attr:" + a) if flagged else None
        if a in self.c.user_attrs and a not in T.UNIVERSE and not a.startswith("_"):
            return None
        lookup = a in T.DATA_UNSAFE_ATTRS or a in T.LOOKUP_ATTRS
        return ("family" if lookup else "unknown", "attr:" + a)

    def visit_Attribute(self, n):
        self.attr_base.add(id(n.value))
        if not isinstance(n.ctx, ast.Load):
            self.module_store(n)
        elif self.module_of(n) is not None or self.reads_module_name(n):
            self.module_value_use(n, True)
        if n.attr == "__name__":
            self.type_ok.add(id(n.value))
        self.library_class_use(n, n.attr)
        if n.attr in T.ABC_MIXINS and not self.in_annotation:
            self.rej("family", n, "abc-mixin:" + n.attr)
        if self.local_module_attribute(n):
            return
        reason = self.attribute_report(n)
        if reason:
            self.rej(reason[0], n, reason[1])
        self.visit(n.value)

    # ---- annotations and type subscripts ----
    def generic_name_ok(self, v):
        if isinstance(v, ast.Name):
            return (v.id in T.TYPE_GENERICS and v.id not in self.c.nonimport_bound
                    and not self.fast_bound(v.id))
        if isinstance(v, ast.Attribute):
            return (isinstance(v.value, ast.Name) and v.value.id in self.c.typing_aliases
                    and v.value.id not in self.c.nonimport_bound and v.attr in T.TYPE_GENERICS)
        return False

    def annotation(self, n):
        self.in_annotation += 1
        try:
            self.annotation_inner(n)
        finally:
            self.in_annotation -= 1

    def annotation_inner(self, n):
        if n is None or isinstance(n, ast.Constant):
            return
        if isinstance(n, (ast.Name, ast.Attribute)):
            self.visit(n)
        elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load) and self.generic_name_ok(
                n.value):
            self.visit(n.value)
            self.annotation(n.slice)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
            self.annotation(n.left)
            self.annotation(n.right)
        elif isinstance(n, (ast.Tuple, ast.List)):
            for e in n.elts:
                self.annotation(e)
        else:
            self.rej("unknown", n, "annotation:" + type(n).__name__)

    def visit_Subscript(self, n):
        if isinstance(n.ctx, ast.Load) and self.generic_name_ok(n.value):
            self.visit(n.value)
            self.annotation(n.slice)
            return
        if self.c.family == "data-lookup":
            self.rej("family", n, "subscript")
        self.generic_visit(n)

    # ---- definitions ----
    def visit_defaults_and_annotations(self, args):
        for d in args.defaults + [d for d in args.kw_defaults if d is not None]:
            self.visit(d)
        for a in args.posonlyargs + args.args + args.kwonlyargs + [
                x for x in (args.vararg, args.kwarg) if x
        ]:
            self.annotation(a.annotation)

    def visit_FunctionDef(self, n):
        if getattr(n, "type_params", None):
            self.rej("unknown", n, "type-params")
        nm = n.name
        if nm.startswith("__") and nm.endswith("__") and nm not in T.SAFE_DUNDER_DEFS:
            self.rej("unknown", n, "dunder-def:" + nm)
        for d in n.decorator_list:
            self.visit(d)
        self.visit_defaults_and_annotations(n.args)
        self.annotation(n.returns)
        self.push_func(param_names(n.args), n.body)
        for st in n.body:
            self.visit(st)
        self.scopes.pop()

    def visit_Lambda(self, n):
        self.visit_defaults_and_annotations(n.args)
        self.push_func(param_names(n.args), [])
        self.visit(n.body)
        self.scopes.pop()

    def visit_ClassDef(self, n):
        if getattr(n, "type_params", None):
            self.rej("unknown", n, "type-params")
        for base in n.bases:
            self.class_ok.add(id(base))
            if isinstance(base, ast.Subscript):
                self.class_ok.add(id(base.value))
        for d in n.decorator_list + n.bases:
            self.visit(d)
        for base in n.bases:
            if not self.base_ok(base):
                self.rej("unknown", base, "class-base")
        for k in n.keywords:
            self.visit(k.value)
        body = bound_in(n.body)
        self.scopes.append(Scope("class", body.names, body.globals))
        for st in n.body:
            self.visit(st)
        self.scopes.pop()

    def base_ok(self, base):
        """Whether a class may inherit from `base`: nothing a C method of a builtin container
        or a library class could reach back into the program through.  A base is judged by what
        its name is bound to, never by how it is spelled."""
        if isinstance(base, ast.Name):
            return self.name_base_ok(base.id)
        if isinstance(base, ast.Attribute):
            return self.module_base_ok(base)
        if isinstance(base, ast.Subscript):
            return (isinstance(base.value, (ast.Name, ast.Attribute)) and self.base_ok(base.value)
                    and getattr(base.value, "id", getattr(base.value, "attr",
                                                          "")) in T.GENERIC_BASES)
        return False

    def name_base_ok(self, name):
        if self.fast_bound(name):
            return False
        bindings = self.c.bind_count.get(name, 0)
        if name in self.c.user_classes and name not in BUILTIN_NAMES:
            return bindings == 1
        if name in T.LIB_BASE_MODULES:
            exact = [(mod, name) for mod in T.LIB_BASE_MODULES[name]]
            return bindings == 1 and self.c.from_imports.get(name) in ([e] for e in exact)
        return (name == "object" or name in T.EXC_NAMES) and bindings == 0

    def module_base_ok(self, base):
        """`M.Name`, where M is a module-level `import` of the module that defines Name."""
        root = base.value
        if not isinstance(root, ast.Name) or self.fast_bound(root.id):
            return False
        module = self.c.module_imports.get(root.id)
        return (self.c.bind_count.get(root.id) == 1
                and module in T.LIB_BASE_MODULES.get(base.attr, ()))

    def _comp(self, n, elts):
        first = n.generators[0]
        self.visit(first.iter)
        names = set()
        for g in n.generators:
            for sub in ast.walk(g.target):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    names.add(sub.id)
        self.scopes.append(Scope("comp", names))
        self.visit(first.target)
        for i in first.ifs:
            self.visit(i)
        for g in n.generators[1:]:
            self.visit(g.iter)
            self.visit(g.target)
            for i in g.ifs:
                self.visit(i)
        for e in elts:
            self.visit(e)
        self.scopes.pop()

    def visit_ListComp(self, n):
        self._comp(n, [n.elt])

    visit_SetComp = visit_GeneratorExp = visit_ListComp  # noqa: N815

    def visit_DictComp(self, n):
        self._comp(n, [n.key, n.value])

    def visit_AnnAssign(self, n):
        self.annotation(n.annotation)
        self.visit(n.target)
        if n.value is not None:
            self.visit(n.value)

    # ---- imports ----
    def nested_import(self, n):
        """Only module-level imports: a module object bound in a function or a class is
        beyond the module-value and module-store rules."""
        if any(scope.kind != "module" for scope in self.scopes):
            self.rej("unknown", n, "nested-import")

    def visit_Import(self, n):
        self.nested_import(n)
        for a in n.names:
            if not self.c.local_chain(a.name.split("."), self.c.roots):
                self.lib_import(n, a.name, None)

    def visit_ImportFrom(self, n):
        self.nested_import(n)
        if n.level or self.c.import_targets(self.m, n):
            return
        for a in n.names:
            if a.name != "*":
                self.lib_import(n, n.module or "", a.name)

    def lib_import(self, node, modname, name):
        names = T.MODULE_NAMES[self.c.family].get(modname)
        if names is None:
            self.rej("unknown", node, "import:" + modname)
        elif name is not None and name not in names:
            self.rej("unknown", node, f"import-name:{modname}.{name}")
        elif name is not None and self.c.arith and name in T.OVERFLOW_ATTRS:
            self.rej("overflow", node, f"overflow-import:{modname}.{name}")

    # ---- arithmetic sites ----
    def const(self, n):
        if isinstance(n, ast.Name) and self.fast_bound(n.id):
            return None
        try:
            return self.c.numconst(n, True, self.m.path)
        except Unfoldable:
            return None

    def check_folded(self, node):
        """OverflowError-class: a constant expression that is, or builds, a huge integer."""
        try:
            folded = self.c.numconst(node, True, self.m.path)
        except Unfoldable as exc:
            folded = None
            if exc.huge:
                self.rej("overflow", node, "huge-constant")
        if isinstance(folded, int) and not isinstance(folded, bool) and abs(folded) >= BIG:
            self.rej("overflow", node, "huge-constant")

    def check_division(self, node, op, left, right):
        if self.c.arith:
            divisor = self.const(right)
            if divisor is not None and divisor:
                return
            formatting = isinstance(left, ast.Constant) and isinstance(left.value, (str, bytes))
            if not (isinstance(op, ast.Mod) and formatting):
                self.rej("family", node, "divisor:" + type(op).__name__)
        elif isinstance(op, ast.Mod) and self.const(left) is None and self.const(right) is None:
            self.rej("family", node, "percent-format")

    def check_power(self, node, left, right):
        exponent, base = self.const(right), self.const(left)
        if exponent is not None and not isinstance(exponent, complex) and exponent >= 0:
            if exponent > 64:
                self.rej("overflow", node, "pow-large-exponent")
        elif not (base is not None and base and exponent is not None):
            self.rej("family", node, "pow-exponent")

    def binop(self, node, op, left, right):
        if self.c.arith and isinstance(node, ast.BinOp):
            self.check_folded(node)
        if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)):
            self.check_division(node, op, left, right)
        elif isinstance(op, ast.Pow) and self.c.arith:
            self.check_power(node, left, right)

    def visit_BinOp(self, n):
        self.binop(n, n.op, n.left, n.right)
        self.generic_visit(n)

    def visit_AugAssign(self, n):
        self.binop(n, n.op, n.target, n.value)
        self.generic_visit(n)

    def visit_Constant(self, node):
        v = node.value
        if self.c.arith and isinstance(v, int) and not isinstance(v, bool) and abs(v) >= BIG:
            self.rej("overflow", node, "huge-constant")

    def ovf_safe_arg(self, a):
        if isinstance(a, ast.Constant):
            v = a.value
            return isinstance(v, str) or (isinstance(v, (int, float)) and abs(v) < 2**53)
        return (isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
                and a.func.id in ("len", "ord", "nondet_int", "nondet_bool", "nondet_str")
                and not self.fast_bound(a.func.id))

    def visit_Compare(self, n):
        for operand in [n.left] + n.comparators:
            self.type_ok.add(id(operand))
        self.generic_visit(n)

    def visit_Call(self, n):
        f = n.func
        self.class_ok.add(id(f))
        if isinstance(f, ast.Name) and f.id in ("isinstance", "issubclass"):
            self.class_ok.update(id(a) for a in n.args[1:])
        if self.is_compared_type_call(n):
            self.visit(n.args[0])
            return
        self.check_overflow_call(n)
        self.generic_visit(n)

    def is_compared_type_call(self, n):
        """`type(x)` as an operand of a comparison or before `.__name__`, the only reads of
        the type builtin: three arguments build a class, and a class is a value to call."""
        return (id(n) in self.type_ok and isinstance(n.func, ast.Name) and n.func.id == "type"
                and len(n.args) == 1 and not n.keywords and not self.fast_bound("type"))

    def check_overflow_call(self, n):
        f = n.func
        if (self.c.arith and self._names_overflow_builtin(f)
                and (n.keywords or not all(self.ovf_safe_arg(a) for a in n.args))):
            self.rej("overflow", n, "overflow-call:" + f.id)

    def _names_overflow_builtin(self, f):
        return (isinstance(f, ast.Name) and f.id in T.OVERFLOW_CALLS and not self.fast_bound(f.id))


def run_one(path, family, roots_mode="env"):
    """Census of one program for one family; the Census object."""
    census = Census(family, path, roots_mode=roots_mode)
    census.run()
    return census


def running_python():
    """(major, minor) of the interpreter the frontend runs, the one whose behaviour matters."""
    return tuple(sys.version_info[:2])


def family_of(types):
    """The census family whose exceptions include all of `types`, or None."""
    wanted = set(types)
    for name, members in T.FAMILIES.items():
        if wanted and wanted <= set(members):
            return name
    return None


def census_summary(path, types):
    """One line saying why `path` cannot raise any of `types`, else None.

    Fail-open: every failure of the census itself, whatever its cause,
    answers None, so the caller converts the program as it would without a
    census.  Only a complete walk that found nothing returns a line.
    """
    try:
        if running_python() != T.SUPPORTED_PYTHON:
            return None
        family = family_of(types)
        if family is None:
            return None
        census = run_one(path, family)
        if census.reports:
            return None
        modules = sum(1 for m in census.mods.values() if m is not None) - 1
        listed = ", ".join(sorted(types))
        return (f"Exception census: no construct of {listed} in {os.path.basename(path)} "
                f"and {modules} other module(s)")
    except Exception:  # pylint: disable=broad-except
        return None

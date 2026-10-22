import ast
import builtins
import copy
from collections import Counter, defaultdict
# pylint: disable=too-many-boolean-expressions


def _class_scope_nodes(node):
    """The nodes of a class-body statement that are evaluated in the class
    scope: not function or lambda bodies, nested class bodies, or the parts of
    a comprehension past its first iterable."""
    yield node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        children = [*node.decorator_list, *args.defaults, *filter(None, args.kw_defaults)]
        children += [
            a.annotation
            for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
            if a and a.annotation
        ]
        children += [node.returns] if node.returns else []
    elif isinstance(node, ast.Lambda):
        children = [*node.args.defaults, *filter(None, node.args.kw_defaults)]
    elif isinstance(node, ast.ClassDef):
        children = [*node.decorator_list, *node.bases, *node.keywords]
    elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        children = [node.generators[0].iter]
    else:
        children = list(ast.iter_child_nodes(node))
    for child in children:
        yield from _class_scope_nodes(child)


def _def_scope_nodes(scope):
    """The nodes under `scope` outside any nested def, which the visit tracks
    as a scope of its own; a nested def is yielded but not entered."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, ast.FunctionDef):
            stack.extend(ast.iter_child_nodes(node))


def _runs_call(node):
    """Whether evaluating `node` in a class body may call something."""
    for n in _class_scope_nodes(node):
        if isinstance(n, ast.Call):
            return True
        if isinstance(n, ast.ClassDef) and any(_runs_call(s) for s in n.body):
            return True
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) and any(
                isinstance(m, ast.Call) for m in ast.walk(n)):
            return True
    return False


def _identifiers(node):
    """The strings in the fields of `node`."""
    for _, value in ast.iter_fields(node):
        yield from (v for v in (value if isinstance(value, list) else [value])
                    if isinstance(v, str))


def _only_called_as_method(module, target):
    """Whether the name `target` binds occurs in `module` only as the
    attribute called in `x.name(...)`, or not at all, as for a dunder that only
    an operator reaches. Any other use, such as comparing the bound method or
    naming it to getattr, can tell the copy from the original (#8356)."""
    called = {
        id(n.func)
        for n in ast.walk(module) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    return all(n is target or id(n) in called or target.id not in _identifiers(n)
               for n in ast.walk(module))


class _QualifyNestedClasses(ast.NodeTransformer):
    """Rewrite `Outer.Inner` to the hoisted class's module-level name."""

    def __init__(self, renames):
        self.renames = renames

    def visit_Attribute(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Load):
            hoisted = self.renames.get((node.value.id, node.attr))
            if hoisted:
                return ast.copy_location(ast.Name(id=hoisted, ctx=ast.Load()), node)
        return node


class ClassContextMixin:

    def hoist_nested_classes(self, module):
        """Lift each class nested in a module-level class body to module level
        as `Outer__Inner`, since the converter only builds module-level classes
        (#8256). `Outer.Inner` is rewritten everywhere, and so is a bare `Inner`
        that the outer class body's scope resolves to the nested class.

        The rewrite is purely by name, so an outer class whose name is bound
        anywhere else in the module is left alone."""
        bound = [name for n in ast.walk(module) for name in self._names_bound_by(n)]
        taken = set(bound) | {n.id for n in ast.walk(module) if isinstance(n, ast.Name)}
        renames = {}
        body = []
        for stmt in module.body:
            if isinstance(stmt, ast.ClassDef) and bound.count(stmt.name) == 1:
                body.extend(self._hoist_from_class(stmt, taken, renames))
            else:
                body.append(stmt)
        self._hoisted_classes = renames
        if renames:
            module.body = body
            _QualifyNestedClasses(renames).visit(module)

    def _hoist_from_class(self, cls, taken, renames):
        """`cls` preceded by the classes hoisted out of it, recursively."""
        hoisted = []
        local = {}
        kept = []
        for index, stmt in enumerate(cls.body):
            for n in _class_scope_nodes(stmt):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in local:
                    n.id = local[n.id]
            if self._can_hoist(cls, index, taken):
                qualified = f"{cls.name}__{stmt.name}"
                taken.add(qualified)
                renames[(cls.name, stmt.name)] = qualified
                local[stmt.name] = qualified
                stmt.name = qualified
                hoisted.extend(self._hoist_from_class(stmt, taken, renames))
            else:
                kept.append(stmt)
        cls.body = kept or [ast.copy_location(ast.Pass(), cls)]
        return [*hoisted, cls]

    def _can_hoist(self, cls, index, taken):
        """Whether `cls.body[index]` is a nested class that keeps its meaning
        at module level: its name is bound by nothing else in the class scope,
        and moving its definition ahead of the outer class does not reorder
        two calls."""
        stmt = cls.body[index]
        if not isinstance(stmt, ast.ClassDef) or f"{cls.name}__{stmt.name}" in taken:
            return False
        if any(n is not stmt and stmt.name in self._names_bound_by(n) for other in cls.body
               for n in _class_scope_nodes(other)):
            return False
        earlier = [*cls.decorator_list, *cls.bases, *cls.keywords, *cls.body[:index]]
        return not (_runs_call(stmt) and any(_runs_call(e) for e in earlier))

    def copy_aliased_methods(self, module):
        """Replace each class-body `alias = name` that names an earlier def of
        the same class body with a copy of that def called `alias` (#8356).
        Python binds the alias to the def's function object, so the copy runs
        the same body even when a subclass overrides `name`."""
        for cls in (n for n in ast.walk(module) if isinstance(n, ast.ClassDef)):
            for index, stmt in enumerate(cls.body):
                method = self._aliased_method(cls, index)
                if method and _only_called_as_method(module, stmt.targets[0]):
                    cls.body[index] = copy.deepcopy(method)
                    cls.body[index].name = stmt.targets[0].id

    def _aliased_method(self, cls, index):
        """The def that `cls.body[index]` aliases, when a renamed copy of it is
        exact, else None."""
        stmt = cls.body[index]
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Name)):
            return None
        method = next((s for s in cls.body[:index]
                       if isinstance(s, ast.FunctionDef) and s.name == stmt.value.id), None)
        if method is None or not self._copy_is_exact(cls, method, stmt.targets[0].id):
            return None
        return method

    def _copy_is_exact(self, cls, method, alias):
        """Whether a copy of `method` named `alias` behaves as the alias: the
        def is undecorated, its defaults are constants, since a copy evaluates
        them again, and both names are bound once in the class scope."""
        bound = [
            b for s in cls.body for n in _class_scope_nodes(s) for b in self._names_bound_by(n)
        ]
        defaults = [*method.args.defaults, *filter(None, method.args.kw_defaults)]
        return (not method.decorator_list and bound.count(alias) == 1
                and bound.count(method.name) == 1
                and all(isinstance(d, ast.Constant) for d in defaults))

    def visit_ClassDef(self, node):
        """Track class context for method definitions."""
        old_class_name = getattr(self, "current_class_name", None)
        self.current_class_name = node.name
        inits_before = len(self._pending_method_default_inits)

        node = self.expand_dataclass(node)
        self._collect_class_attr_annotations(node)
        self._record_exit_suppresses_all(node)
        self._record_class_with_exit(node)
        self._class_stack.append(node)
        self.generic_visit(node)
        self._class_stack.pop()

        self.current_class_name = old_class_name

        # Hoist method default-helper assignments past the outermost ClassDef
        # so they become module-level statements visible at call sites. For
        # nested classes, keep them pending until the outer ClassDef exits.
        if old_class_name is None and len(self._pending_method_default_inits) > inits_before:
            hoisted = self._pending_method_default_inits[inits_before:]
            del self._pending_method_default_inits[inits_before:]
            return [node, *hoisted]
        return node

    def _class_body_binds(self, stmt, name):
        """Whether a statement directly in a class body binds `name` there."""
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return stmt.name == name
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AugAssign) or (isinstance(stmt, ast.AnnAssign)
                                                 and stmt.value is not None):
            targets = [stmt.target]
        else:
            targets = []
        return any(name in self._names_bound_by(n) for t in targets for n in ast.walk(t))

    def class_scope_default(self, default_node, method):
        """A method default naming an attribute its top-level class bound before
        the method, as `Class.name`: the default is hoisted past the class, where
        the bare name no longer resolves (#8200)."""
        if len(self._class_stack) != 1 or not isinstance(default_node, ast.Name):
            return default_node
        cls = self._class_stack[0]
        for stmt in cls.body:
            if stmt is method:
                break
            if self._class_body_binds(stmt, default_node.id):
                qualified = ast.Attribute(value=ast.Name(id=cls.name, ctx=ast.Load()),
                                          attr=default_node.id,
                                          ctx=ast.Load())
                return ast.copy_location(qualified, default_node)
        return default_node

    @staticmethod
    def _names_bound_by(node):
        """The names a single AST node binds in its scope."""
        if isinstance(node, ast.arg):
            return [node.arg]
        if isinstance(node, ast.Name):
            return [] if isinstance(node.ctx, ast.Load) else [node.id]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler,
                             ast.MatchAs, ast.MatchStar)):
            return [node.name]
        if isinstance(node, ast.alias):
            return [node.asname or node.name.split(".")[0]]
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return node.names
        return []

    def _binds_name(self, tree, name, skip=None):
        """Whether any node in `tree` other than `skip` binds `name`."""
        return any(n is not skip and name in self._names_bound_by(n) for n in ast.walk(tree))

    def _cls_constructor_calls(self, member):
        """The `cls(...)` calls of a @classmethod whose `cls` is never rebound, else []."""
        decorators = member.decorator_list if isinstance(member, ast.FunctionDef) else []
        if not (len(decorators) == 1 and isinstance(decorators[0], ast.Name)
                and decorators[0].id == "classmethod"):
            return []
        params = member.args.posonlyargs + member.args.args
        if not params or params[0].arg != "cls" or self._binds_name(member, "cls", params[0]):
            return []
        return [
            n for n in ast.walk(member)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "cls"
        ]

    def _rebind_cls_calls(self, method, class_name):
        for call in self._cls_constructor_calls(method):
            call.func = ast.copy_location(ast.Name(id=class_name, ctx=ast.Load()), call.func)

    @staticmethod
    def _class_ancestry(classes, node):
        """`node` and its single-inheritance ancestors in `classes`, nearest first."""
        chain = []
        while node is not None and node not in chain:
            chain.append(node)
            base = node.bases[0] if len(node.bases) == 1 else None
            node = classes.get(base.id) if isinstance(base, ast.Name) else None
        return chain

    @staticmethod
    def _class_defines(node, name):
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == name:
                return True
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            else:
                targets = [getattr(stmt, "target", None)]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return True
        return False

    def _method_owner(self, classes, node, name):
        """The nearest class in `node`'s ancestry that defines `name`, or None."""
        return next(
            (a for a in self._class_ancestry(classes, node) if self._class_defines(a, name)), None)

    def _unbindable_classes(self, classes):
        """Classes reached through multiple inheritance or a metaclass keyword."""
        unsafe = set()
        for node in classes.values():
            if len(node.bases) > 1 or node.keywords:
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id in classes:
                        unsafe.update(a.name
                                      for a in self._class_ancestry(classes, classes[base.id]))
        return unsafe

    @staticmethod
    def _indirect_attributes(module, classes):
        """Attribute names used anywhere other than as a direct `Class.name(...)` call."""
        direct = {
            id(n.func)
            for n in ast.walk(module)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id in classes
        }
        return {
            n.attr
            for n in ast.walk(module) if isinstance(n, ast.Attribute) and id(n) not in direct
        }

    @staticmethod
    def _top_level_classes(module):
        """Top-level classes by name, or None when a name is defined twice."""
        classes = {}
        for stmt in module.body:
            if isinstance(stmt, ast.ClassDef):
                if stmt.name in classes:
                    return None
                classes[stmt.name] = stmt
        return classes

    def _classmethod_heirs(self, classes, owner, name):
        """Classes other than `owner` whose `name` resolves to `owner`'s method."""
        return [
            c for c in classes.values()
            if c is not owner and self._method_owner(classes, c, name) is owner
        ]

    def _can_bind(self, member, targets):
        """Whether `member` can be bound to each class in `targets`, copying it if needed."""
        args = member.args
        if len(targets) > 1 and (args.defaults or any(d is not None for d in args.kw_defaults)):
            return False
        return not any(self._binds_name(member, t.name) for t in targets)

    def _bind_classmethod_constructors(self, module):
        """Rewrite `cls(...)` in a @classmethod to the class the method runs on.

        Only in the entry module, and only when that class is known statically: every
        reference to the method is a direct `Class.method(...)` call and the classes
        involved use single inheritance. A subclass inheriting the method gets its own
        copy. Anything else keeps `cls(...)`, which the converter rejects.
        """
        classes = self._top_level_classes(module)
        if not self.is_entry_module or classes is None:
            return
        unsafe = self._unbindable_classes(classes)
        indirect = self._indirect_attributes(module, classes)
        for owner in classes.values():
            if owner.name in unsafe:
                continue
            for member in list(owner.body):
                if not self._cls_constructor_calls(member) or member.name in indirect:
                    continue
                heirs = self._classmethod_heirs(classes, owner, member.name)
                if not self._can_bind(member, [owner] + heirs):
                    continue
                template = copy.deepcopy(member)
                self._rebind_cls_calls(member, owner.name)
                for heir in heirs:
                    method = copy.deepcopy(template)
                    self._rebind_cls_calls(method, heir.name)
                    heir.body.append(method)

    def _self_called_methods(self, member):
        """The names `member` calls on its receiver, when a copy of `member` is
        exact: it is undecorated, never rebinds its receiver, does not use
        `super` or `__class__`, and has constant defaults."""
        receiver = self._receiver(member)
        if receiver is None or member.name.startswith("__"):
            return set()
        defaults = [*member.args.defaults, *filter(None, member.args.kw_defaults)]
        if (not all(isinstance(d, ast.Constant) for d in defaults) or any(
                isinstance(n, ast.Name) and n.id in ("super", "__class__")
                for n in ast.walk(member))):
            return set()
        return {
            n.func.attr
            for n in ast.walk(member)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == receiver
        }

    def _bind_overridden_method_calls(self, module):
        """Give a subclass its own copy of an inherited method that calls
        `self.m()` where the subclass resolves `m` to another definition than
        the method's class does, so the copy types `self` as the subclass and
        the call reaches the override (#9035).

        Only in the entry module, with single inheritance and no metaclass."""
        classes = self._top_level_classes(module)
        if not self.is_entry_module or classes is None:
            return
        unsafe = self._unbindable_classes(classes)
        copies = []
        for owner in classes.values():
            if owner.name in unsafe:
                continue
            for member in owner.body:
                called = self._self_called_methods(member)
                if not called:
                    continue
                copies += [(heir, member)
                           for heir in self._classmethod_heirs(classes, owner, member.name) if any(
                               self._method_owner(classes, heir, name) is not self._method_owner(
                                   classes, owner, name) for name in called)]
        for heir, member in copies:
            heir.body.append(copy.deepcopy(member))

    @staticmethod
    def _names_mentioned(module):
        """Every identifier or string `module` mentions other than by loading an attribute."""
        mentioned = set()
        for n in ast.walk(module):
            for field, value in ast.iter_fields(n):
                if not (isinstance(n, ast.Attribute) and field == "attr"
                        and isinstance(n.ctx, ast.Load)):
                    mentioned.update(v for v in (value if isinstance(value, list) else [value])
                                     if isinstance(v, str))
        return mentioned

    def _receiver(self, method):
        """The first parameter of an undecorated method that never rebinds it, else None."""
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) or method.decorator_list:
            return None
        params = method.args.posonlyargs + method.args.args
        if not params or self._binds_name(method, params[0].arg, params[0]):
            return None
        return params[0].arg

    def _inherited_nested_classes(self, classes, cls, excluded):
        """The classes hoisted out of `cls` or its ancestors, by attribute name,
        leaving out the names in `excluded`."""
        ancestry = {a.name for a in self._class_ancestry(classes, cls)}
        return {
            attr: name
            for (outer, attr), name in self._hoisted_classes.items()
            if outer in ancestry and attr not in excluded
        }

    @staticmethod
    def _recovers_classes(module):
        """Whether `module` can reach a class from an instance or by name. A
        `type(x)` that is only an operand of `is` or `is not` reaches no method."""
        compared = {
            id(operand)
            for n in ast.walk(module) if isinstance(n, ast.Compare) and all(
                isinstance(op, (ast.Is, ast.IsNot)) for op in n.ops)
            for operand in (n.left, *n.comparators)
        }
        return any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "type"
            and len(n.args) == 1 and id(n) not in compared
            or isinstance(n, ast.Attribute) and n.attr == "__class__"
            or isinstance(n, ast.Name) and n.id in ("getattr", "vars", "globals")
            for n in ast.walk(module))

    def _escaped_classes(self, module, classes):
        """Classes whose object may reach a method with a receiver of another
        class: all of them when the module can recover a class, else those
        named other than as a constructor call, a base, or the class tested by
        isinstance or issubclass."""
        if self._recovers_classes(module):
            return set(classes)
        allowed = set()
        for n in ast.walk(module):
            if isinstance(n, ast.ClassDef):
                allowed.update(map(id, n.bases))
            elif isinstance(n, ast.Call):
                allowed.add(id(n.func))
                if isinstance(n.func, ast.Name) and n.func.id in ("isinstance", "issubclass"):
                    allowed.update(map(id, n.args[1:2]))
        return {
            n.id
            for n in ast.walk(module)
            if isinstance(n, ast.Name) and n.id in classes and id(n) not in allowed
        }

    def _inherits_from(self, classes, cls, names):
        """Whether a class named in `names` is `cls` or a subclass of it."""
        return any(c.name in names and cls in self._class_ancestry(classes, c)
                   for c in classes.values())

    def _bind_nested_classes_via_self(self, module):
        """Rewrite `self.Inner` in a method to the class hoisted out of the
        method's class or one of its ancestors (#8357).

        Only in the entry module, where every subclass is visible, and for a
        class that it and its subclasses only construct, subclass or test with
        isinstance, with single inheritance and no metaclass: otherwise a method
        may run on an instance of another class. And only for a name hoisted out
        of one class that the module mentions nowhere but in attribute loads:
        any other mention could bind it on an instance or a subclass, as could
        overriding `__getattribute__`."""
        classes = self._top_level_classes(module)
        if not self.is_entry_module or classes is None:
            return
        mentioned = self._names_mentioned(module)
        if "__getattribute__" in mentioned:
            return
        owners = Counter(attr for _, attr in self._hoisted_classes)
        mentioned.update(attr for attr, count in owners.items() if count > 1)
        escaped = self._escaped_classes(module, classes) | self._unbindable_classes(classes)
        for cls in classes.values():
            if self._inherits_from(classes, cls, escaped):
                continue
            hoisted = self._inherited_nested_classes(classes, cls, mentioned)
            for method in cls.body:
                receiver = self._receiver(method)
                if receiver is not None:
                    renames = {(receiver, attr): name for attr, name in hoisted.items()}
                    for stmt in method.body:
                        _QualifyNestedClasses(renames).visit(stmt)

    def _calls_unresolved_attribute(self, classes, cls, method):
        """Whether `method` of `cls` calls `self.x(...)` with an `x` that no
        class in the ancestry of `cls` defines, so only an attribute an
        instance gets elsewhere can supply it."""
        receiver = self._receiver(method)
        return receiver is not None and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == receiver
            and self._method_owner(classes, cls, n.func.attr) is None for n in ast.walk(method))

    def _heirs_needing_copy(self, classes, owner, member):
        """The subclasses of `owner` to give their own copy of `member`, or
        none when one overrides it or it uses `super()`, which would bind
        another class."""
        if (not self._calls_unresolved_attribute(classes, owner, member)
                or {"super", "__class__"} & self._names_mentioned(member)):
            return []
        family = [c for c in classes.values() if owner in self._class_ancestry(classes, c)]
        heirs = [c for c in family if self._method_owner(classes, c, member.name) is owner]
        if len(heirs) != len(family) or not self._can_bind(member, heirs):
            return []
        return [c for c in heirs if c is not owner]

    def _runs_only_on_subclasses(self, module, classes, owner, member):
        """Whether `member` of `owner`, which every subclass copies, runs on
        no object: `owner` is never constructed, nothing in `owner` names the
        method, and no class deriving from it names `super`, which could
        reach the original."""
        family = [c for c in classes.values() if owner in self._class_ancestry(classes, c)]
        constructed = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == owner.name
            for n in ast.walk(module))
        named = any(isinstance(n, ast.Attribute) and n.attr == member.name for n in ast.walk(owner))
        return not (constructed or named or any("super" in self._names_mentioned(c)
                                                for c in family))

    def _copy_methods_into_heirs(self, module):
        """Give each subclass inheriting a method that calls `self.x(...)`,
        with `x` defined by no ancestor, its own copy (#8956).

        The copy types its receiver as the subclass, which may give `x`. Only
        in the entry module, for a class that no other class's object reaches
        (see _bind_nested_classes_via_self). The original is dropped when it
        can run on no object; otherwise the converter refuses its call."""
        classes = self._top_level_classes(module)
        if not self.is_entry_module or classes is None:
            return
        escaped = self._escaped_classes(module, classes) | self._unbindable_classes(classes)
        for owner in classes.values():
            if self._inherits_from(classes, owner, escaped):
                continue
            for member in list(owner.body):
                heirs = self._heirs_needing_copy(classes, owner, member)
                for heir in heirs:
                    heir.body.append(copy.deepcopy(member))
                if heirs and self._runs_only_on_subclasses(module, classes, owner, member):
                    owner.body.remove(member)
            if not owner.body:
                owner.body.append(ast.Pass())

    @staticmethod
    def _is_name_copy(node):
        """Whether `node` is a plain `x = y` assignment."""
        return (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Name))

    @staticmethod
    def _module_level_nodes(module):
        """The nodes of `module` outside any function or lambda body."""
        stack = [module]
        while stack:
            node = stack.pop()
            yield node
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                stack.extend(ast.iter_child_nodes(node))

    def _bind_class_alias_calls(self, module):
        """Rewrite `alias(...)` to `Class(...)` where a top-level `alias = Class`
        statement is the only binding of `alias`, to a top-level class the
        module never rebinds: the converter recognises a constructor call only
        by the class's own name (#8352). A module-level call before the binding
        stays as written, since CPython raises NameError there."""
        counts = Counter(name for n in ast.walk(module) for name in self._names_bound_by(n))
        classes = {stmt.name for stmt in module.body if isinstance(stmt, ast.ClassDef)}
        bindings = {
            n.targets[0].id: n
            for n in filter(self._is_name_copy, module.body)
            if n.value.id in classes and counts[n.targets[0].id] == counts[n.value.id] == 1
        }
        module_level = set(map(id, self._module_level_nodes(module)))
        for n in ast.walk(module):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
                continue
            binding = bindings.get(n.func.id)
            if binding is None or (id(n) in module_level and
                                   (n.lineno, n.col_offset) < (binding.lineno, binding.col_offset)):
                continue
            n.func = ast.copy_location(ast.Name(id=binding.value.id, ctx=ast.Load()), n.func)

    def _bind_type_values(self, module):
        """Replace a parameter or local that only ever holds one known class by
        that class's name, which the converter resolves statically (#8389).

        A known class is a top-level class or undecorated function bound only
        by its definition, or a builtin class the module never binds, so its
        name denotes the same object wherever it is read. A local qualifies
        when a statement `t = C` directly in its function's body is its only
        binding: the reads after that statement become `C`. A parameter
        qualifies when, in the entry module, every reference to its
        undecorated top-level function is a call passing a known class for it
        and nothing rebinds it; each distinct combination of such classes gets
        its own copy of the function, and the calls keep their arguments, so
        they are still evaluated."""
        while True:
            counts = Counter(name for n in ast.walk(module) for name in self._names_bound_by(n))
            if counts["*"]:
                return
            if self.is_entry_module and self._hoist_lambda_arguments(module, counts):
                continue
            known = self._known_classes(module, counts)
            self._bind_local_type_values(module, known)
            if not self.is_entry_module:
                self._unroll_class_tuples(module, known, counts)
                return
            self._lower_called_lambdas(module, known, counts)
            if not self._specialise_one_function(module, known, counts):
                self._unroll_class_tuples(module, known, counts)
                return

    def _unroll_class_tuples(self, module, known, counts):
        """Unroll a `for` loop, and an `any`/`all` over a generator, that runs
        over a tuple of known classes, so each copy reads one class by name
        (#8765). A loop qualifies when its target is a name and it has no
        `break`, `continue` or `else`; the target is left bound to the last
        class, as the loop leaves it. The tuple is written in place, or bound
        to a name nothing else binds."""
        tuples = {
            n.targets[0].id: n.value
            for n in ast.walk(module) if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and counts[n.targets[0].id] == 1
        }

        def classes(it):
            spelled = self._class_value(
                tuples.get(it.id) if isinstance(it, ast.Name) else it, known)
            if spelled is None or not spelled.startswith("("):
                return None
            return [ast.unparse(e) for e in ast.parse(spelled, mode="eval").body.elts]

        def copies(nodes, name, values):
            out = []
            for value in values:
                copied = copy.deepcopy(nodes)
                self._replace_reads(copied, name, value)
                out.append(copied)
            return out

        mixin = self

        class Unroll(ast.NodeTransformer):

            def visit_For(self, node):
                self.generic_visit(node)
                values = classes(node.iter)
                if (not values or not isinstance(node.target, ast.Name) or node.orelse or any(
                        isinstance(n, (ast.Break, ast.Continue)) for s in node.body
                        for n in ast.walk(s))):
                    return node
                body = [s for c in copies(node.body, node.target.id, values) for s in c]
                last = ast.parse(f"{node.target.id} = {values[-1]}").body[0]
                return [ast.copy_location(s, node) for s in body + [last]]

            def visit_Call(self, node):
                self.generic_visit(node)
                folded = mixin._fold_class_generator(node, classes, copies, counts)
                return ast.copy_location(folded, node) if folded else node

        Unroll().visit(module)
        ast.fix_missing_locations(module)

    @staticmethod
    def _fold_class_generator(node, classes, copies, counts):
        """`bool(e1 or e2 ...)` for `any(e for x in <classes>)`, with `and`
        for `all`, else None."""
        gen = ClassContextMixin._class_generator(node, counts)
        values = classes(gen.iter) if gen else None
        if not values:
            return None
        elts = [c[0] for c in copies([node.args[0].elt], gen.target.id, values)]
        joined = elts[0] if len(elts) == 1 else ast.BoolOp(
            op=ast.Or() if node.func.id == "any" else ast.And(), values=elts)
        return ast.Call(func=ast.Name(id="bool", ctx=ast.Load()), args=[joined], keywords=[])

    @staticmethod
    def _class_generator(node, counts):
        """The one plain `for x in ...` clause of a call `any(...)` or
        `all(...)` on a generator, with neither name rebound, else None."""
        if (not isinstance(node.func, ast.Name) or node.func.id not in ("any", "all")
                or counts[node.func.id] or counts["bool"] or node.keywords or len(node.args) != 1
                or not isinstance(node.args[0], ast.GeneratorExp)):
            return None
        gen = node.args[0].generators
        if len(gen) != 1 or gen[0].ifs or gen[0].is_async or not isinstance(
                gen[0].target, ast.Name):
            return None
        return gen[0]

    @staticmethod
    def _known_classes(module, counts):
        defined = {
            s.name
            for s in module.body
            if (isinstance(s, ast.ClassDef) or ClassContextMixin._is_plain_function(s))
            and counts[s.name] == 1
        }
        builtin = {n for n, v in vars(builtins).items() if isinstance(v, type) and not counts[n]}
        none_type = set() if counts["NoneType"] else {"NoneType"}
        return defined | builtin | none_type | ClassContextMixin._typing_any(module)

    @staticmethod
    def _typing_any(module):
        """The names `from typing import Any` binds, when nothing rebinds them.
        Counted afresh, as `self._names_bound_by` resolves to
        GeneratorMixin's, which skips imports."""
        bound = Counter(name for n in ast.walk(module)
                        for name in ClassContextMixin._names_bound_by(n))
        return {
            a.asname or a.name
            for s in module.body if isinstance(s, ast.ImportFrom) and s.module == "typing"
            for a in s.names if a.name == "Any" and bound[a.asname or a.name] == 1
        }

    @staticmethod
    def _is_plain_function(stmt):
        """Whether `stmt` defines an undecorated function without *args or
        **kwargs: calls to a variadic function are lowered before this pass,
        so a call a copy introduces would be left unlowered (#8498)."""
        return (isinstance(stmt, ast.FunctionDef) and not stmt.decorator_list
                and not (stmt.args.vararg or stmt.args.kwarg))

    @staticmethod
    def _lambda_as_def(stmt, counts):
        """`def f(...): return body` for a statement `f = lambda ...: body`
        that is the only binding of `f`, else None."""
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Lambda)
                and counts[stmt.targets[0].id] == 1):
            return None
        func = ast.FunctionDef(name=stmt.targets[0].id,
                               args=stmt.value.args,
                               body=[ast.Return(value=stmt.value.body)],
                               decorator_list=[],
                               returns=None,
                               type_params=[])
        return ast.fix_missing_locations(ast.copy_location(func, stmt))

    @staticmethod
    def _called_parameters(func):
        """The indexes of `func`'s positional parameters it calls by name."""
        called = {
            n.func.id
            for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        return {i for i, p in enumerate(func.args.posonlyargs + func.args.args) if p.arg in called}

    @staticmethod
    def _module_level_calls(node):
        """The calls in `node` that run at module level, outside any def,
        lambda or comprehension whose names a lambda could capture."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                             ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, ast.Call):
            yield node
        for child in ast.iter_child_nodes(node):
            yield from ClassContextMixin._module_level_calls(child)

    def _hoist_lambda_arguments(self, module, counts):
        """Turn a lambda passed at module level to a top-level function that
        calls that parameter into a def named for it, so that the function is
        specialised on it as on any def (#8979); whether any was. At module
        level a lambda's free names are globals either way."""
        defs = {s.name: s for s in module.body if self._is_plain_function(s)}
        hoisted = 0
        for i, stmt in reversed(list(enumerate(module.body))):
            new = []
            for call in self._module_level_calls(stmt):
                callee = defs.get(call.func.id) if isinstance(call.func, ast.Name) else None
                if callee is None:
                    continue
                for index in self._called_parameters(callee):
                    if index < len(call.args) and isinstance(call.args[index], ast.Lambda):
                        name = f"ESBMC_lambda_arg_{hoisted}"
                        while counts[name]:
                            hoisted += 1
                            name = f"ESBMC_lambda_arg_{hoisted}"
                        hoisted += 1
                        lam = call.args[index]
                        new.append(
                            ast.fix_missing_locations(
                                ast.copy_location(
                                    ast.FunctionDef(name=name,
                                                    args=lam.args,
                                                    body=[ast.Return(value=lam.body)],
                                                    decorator_list=[],
                                                    returns=None,
                                                    type_params=[]), lam)))
                        call.args[index] = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), lam)
            module.body[i:i] = new
        return hoisted > 0

    def _lower_called_lambdas(self, module, known, counts):
        """Rewrite a top-level `f = lambda ...` that calls one of its
        parameters to `def f(...)` when every reference to `f` is a call
        passing a known class or function for each called parameter, so that
        the def is specialised. The converter types lambda parameters as
        numbers and cannot call them (#8485)."""
        loads = Counter(n.id for n in ast.walk(module)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
        for i, stmt in enumerate(module.body):
            func = self._lambda_as_def(stmt, counts)
            wanted = self._called_parameters(func) if func else None
            if not wanted:
                continue
            calls = [
                n for n in ast.walk(module) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == func.name
            ]
            if (calls and loads[func.name] == len(calls)
                    and wanted <= set(self._type_parameters(func, known, calls))):
                module.body[i] = func

    def _specialise_one_function(self, module, known, counts):
        """Specialise the first top-level function that only ever is called
        directly by name; whether one was."""
        loads = Counter(n.id for n in ast.walk(module) if isinstance(n, ast.Name))
        calls = defaultdict(list)
        for n in ast.walk(module):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                calls[n.func.id].append(n)
        taken = counts.keys() | loads.keys()
        return any(
            self._specialise_type_parameters(module, f, known, calls[f.name], taken)
            for f in module.body if isinstance(f, ast.FunctionDef) and not f.decorator_list
            and counts[f.name] == 1 and loads[f.name] == len(calls[f.name]))

    @staticmethod
    def _replace_reads(nodes, name, value):
        """Reads of `name` in `nodes` become `value`: a class name, or the
        source of a tuple of classes."""

        class Reads(ast.NodeTransformer):

            def visit_Name(self, n):
                if n.id != name or not isinstance(n.ctx, ast.Load):
                    return n
                if value.isidentifier():
                    n.id = value
                    return n
                spelled = ast.parse(value, mode="eval").body
                return ast.fix_missing_locations(ast.copy_location(spelled, n))

        for stmt in nodes:
            Reads().visit(stmt)

    def _bind_local_type_values(self, module, known):
        for func in ast.walk(module):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for i, stmt in enumerate(func.body):
                if (self._is_name_copy(stmt) and stmt.value.id in known
                        and not self._binds_name(func, stmt.targets[0].id, stmt.targets[0])):
                    self._replace_reads(func.body[i + 1:], stmt.targets[0].id, stmt.value.id)

    @staticmethod
    def _is_type_of_none(arg, known):
        """Whether `arg` is `type(None)`, which has no builtin name; the frontend
        reads NoneType as it (#8764)."""
        return (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                and arg.func.id == "type" and {"type", "NoneType"} <= known and not arg.keywords
                and len(arg.args) == 1 and isinstance(arg.args[0], ast.Constant)
                and arg.args[0].value is None)

    @staticmethod
    def _class_argument(call, func, index, known):
        """The known class `call` passes for `func`'s positional parameter
        `index`, else None."""
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None
                                                                     for k in call.keywords):
            return None
        param = (func.args.posonlyargs + func.args.args)[index].arg
        by_keyword = index >= len(func.args.posonlyargs)
        arg = call.args[index] if index < len(call.args) else next(
            (k.value for k in call.keywords if by_keyword and k.arg == param), None)
        return ClassContextMixin._class_value(arg, known)

    @staticmethod
    def _class_value(node, known):
        """The source of the known class, `type(None)`, or tuple of them that
        `node` spells, else None."""
        if isinstance(node, ast.Tuple) and isinstance(node.ctx, ast.Load):
            return ClassContextMixin._class_tuple(node, known)
        return ClassContextMixin._class_name(node, known)

    @staticmethod
    def _class_tuple(node, known):
        """The source of a tuple whose elements each spell a known class or
        `type(None)`, else None."""
        parts = [ClassContextMixin._class_name(e, known) for e in node.elts]
        if not parts or not all(parts):
            return None
        return "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"

    @staticmethod
    def _class_name(node, known):
        """The known class `node` names, or `NoneType` for `type(None)`, else
        None."""
        if ClassContextMixin._is_type_of_none(node, known):
            return "NoneType"
        return node.id if isinstance(node, ast.Name) and node.id in known else None

    def _type_parameters(self, func, known, calls):
        """The indexes of the parameters `func` reads and never rebinds, for
        which every one of `calls` passes a known class."""
        reads = {n.id for s in func.body for n in ast.walk(s) if isinstance(n, ast.Name)}
        return [
            i for i, p in enumerate(func.args.posonlyargs + func.args.args)
            if p.arg in reads and not self._binds_name(func, p.arg, p) and all(
                self._class_argument(c, func, i, known) for c in calls)
        ]

    def _type_copy(self, func, name, bindings):
        """A copy of `func` called `name` that reads each `(param, cls)` in
        `bindings` as `cls`."""
        spec = copy.deepcopy(func)
        spec.name = name
        for param, cls in bindings:
            self._replace_reads(spec.body, param, cls)
        return spec

    def _specialise_type_parameters(self, module, func, known, calls, taken):
        """Copy `func` once per combination of known classes its `calls` pass
        for its type parameters, naming no copy after a name in `taken`;
        whether it did."""
        indexes = self._type_parameters(func, known, calls)
        keys = [tuple(self._class_argument(c, func, i, known) for i in indexes) for c in calls]
        names = {key: f"{func.name}__esbmc_type{n}" for n, key in enumerate(dict.fromkeys(keys))}
        if not calls or not indexes or not taken.isdisjoint(names.values()):
            return False
        for call, key in zip(calls, keys):
            call.func.id = names[key]
            self._drop_tuple_arguments(call, func, indexes, key)
        params = [p.arg for p in func.args.posonlyargs + func.args.args]
        at = module.body.index(func)
        module.body[at:at + 1] = [
            self._type_copy(func, name, [(params[i], cls) for i, cls in zip(indexes, key)])
            for key, name in names.items()
        ]
        return True

    @staticmethod
    def _drop_tuple_arguments(call, func, indexes, key):
        """Pass `None` for each tuple of classes `call` gives a specialised
        parameter: the copy reads the tuple in place, and spelling it has no
        effect to keep, while a tuple argument would not match the copy's
        unread, untyped parameter."""
        params = [p.arg for p in func.args.posonlyargs + func.args.args]
        for index, value in zip(indexes, key):
            if not value.startswith("("):
                continue
            if index < len(call.args):
                call.args[index] = ast.copy_location(ast.Constant(value=None), call.args[index])
                continue
            for keyword in call.keywords:
                if keyword.arg == params[index]:
                    keyword.value = ast.copy_location(ast.Constant(value=None), keyword.value)

    @staticmethod
    def _construction(stmt):
        """`(o, C)` for a statement `o = C(...)`, else None."""
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)):
            return stmt.targets[0].id, stmt.value.func.id
        return None

    @staticmethod
    def _assigned_attributes(module):
        """The attribute names `module` assigns or deletes anywhere."""
        return {
            n.attr
            for n in ast.walk(module)
            if isinstance(n, ast.Attribute) and not isinstance(n.ctx, ast.Load)
        }

    @staticmethod
    def _variable_names(module):
        """The names `module` binds anywhere other than by an import or a
        definition."""
        return {
            name
            for n in ast.walk(module)
            if not isinstance(n, (ast.alias, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for name in ClassContextMixin._names_bound_by(n)
        }

    @staticmethod
    def _scope_binding_counts(module):
        """For each def in `module`, keyed by node id, and for `module`, keyed
        by None, the node and how often each name is bound in it outside its
        nested defs."""
        return {
            None if scope is module else id(scope):
            (scope,
             Counter(name for n in _def_scope_nodes(scope)
                     for name in ClassContextMixin._names_bound_by(n)))
            for scope in (module, *(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)))
        }

    def _scan_fixed_receivers(self, module):
        """Map each `o` of a top-level `o = C(...)` to the methods `C` defines,
        where `o` and `C` are each bound only once and `C` is a top-level
        class, less any attribute name the module ever assigns: there `o.m`
        names the same method wherever it is evaluated (#8354)."""
        counts = Counter(name for n in ast.walk(module)
                         for name in ClassContextMixin._names_bound_by(n))
        stored = self._assigned_attributes(module)
        methods = {
            stmt.name: {m.name
                        for m in stmt.body if isinstance(m, ast.FunctionDef)} - stored
            for stmt in module.body if isinstance(stmt, ast.ClassDef) and counts[stmt.name] == 1
        }
        self.fixed_receivers = {
            name: methods[cls]
            for name, cls in filter(None, map(self._construction, module.body))
            if counts[name] == 1 and cls in methods
        }
        self.once_bound_names = {name for name, count in counts.items() if count == 1}
        # Holding the nodes keeps their ids from being reused.
        self.module_statements = {id(stmt): stmt for stmt in module.body}

    def _record_exit_suppresses_all(self, class_node):
        """Cache classes whose __exit__ unconditionally returns True."""
        if not hasattr(self, "_exit_suppresses_all"):
            self._exit_suppresses_all = set()
        for member in class_node.body:
            is_exit_true = (isinstance(member, ast.FunctionDef) and member.name == "__exit__"
                            and len(member.body) == 1 and isinstance(member.body[0], ast.Return)
                            and isinstance(member.body[0].value, ast.Constant)
                            and member.body[0].value.value is True)
            if is_exit_true:
                self._exit_suppresses_all.add(class_node.name)
                return

    def _record_class_with_exit(self, class_node):
        """Cache classes whose __exit__ may suppress exceptions."""
        if not hasattr(self, "_classes_with_exit"):
            self._classes_with_exit = set()

        exit_method = None
        for member in class_node.body:
            if isinstance(member, ast.FunctionDef) and member.name == "__exit__":
                exit_method = member
                break

        if self._class_inherits_exit_handler(class_node):
            self._classes_with_exit.add(class_node.name)

        if exit_method is None:
            return

        if self._function_may_return_truthy(exit_method):
            self._classes_with_exit.add(class_node.name)

    def _class_inherits_exit_handler(self, class_node):
        if not hasattr(self, "_classes_with_exit"):
            return False
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in self._classes_with_exit:
                return True
        return False

    @staticmethod
    def _function_may_return_truthy(func_node):
        stack = list(func_node.body)
        while stack:
            stmt = stack.pop()
            if isinstance(stmt, ast.Return):
                if ClassContextMixin._return_may_be_truthy(stmt.value):
                    return True
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            stack.extend(reversed(getattr(stmt, "body", [])))
            stack.extend(reversed(getattr(stmt, "orelse", [])))
            stack.extend(reversed(getattr(stmt, "finalbody", [])))
        return False

    @staticmethod
    def _return_may_be_truthy(value):
        if value is None:
            return False
        if isinstance(value, ast.Constant):
            return bool(value.value)
        return True

    def _collect_class_attr_annotations(self, class_node):
        """Scan __init__ for self.attr: T = ... and cache annotations."""
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for stmt in item.body:
                    if (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Attribute)
                            and isinstance(stmt.target.value, ast.Name)
                            and stmt.target.value.id == "self" and stmt.annotation is not None):
                        class_name = class_node.name
                        attr_name = stmt.target.attr
                        if class_name not in self.class_attr_annotations:
                            self.class_attr_annotations[class_name] = {}
                        self.class_attr_annotations[class_name][attr_name] = stmt.annotation

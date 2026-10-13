import ast
import copy
from collections import Counter
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
        """Whether `module` can reach a class from an instance or by name."""
        return any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "type"
            and len(n.args) == 1 or isinstance(n, ast.Attribute) and n.attr == "__class__"
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

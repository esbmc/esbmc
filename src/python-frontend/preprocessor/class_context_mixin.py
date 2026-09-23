import ast
import copy
# pylint: disable=too-many-boolean-expressions


class ClassContextMixin:

    def visit_ClassDef(self, node):
        """Track class context for method definitions."""
        old_class_name = getattr(self, "current_class_name", None)
        self.current_class_name = node.name
        inits_before = len(self._pending_method_default_inits)

        node = self.expand_dataclass(node)
        self._collect_class_attr_annotations(node)
        self._record_exit_suppresses_all(node)
        self._record_class_with_exit(node)
        self.generic_visit(node)

        self.current_class_name = old_class_name

        # Hoist method default-helper assignments past the outermost ClassDef
        # so they become module-level statements visible at call sites. For
        # nested classes, keep them pending until the outer ClassDef exits.
        if old_class_name is None and len(self._pending_method_default_inits) > inits_before:
            hoisted = self._pending_method_default_inits[inits_before:]
            del self._pending_method_default_inits[inits_before:]
            return [node, *hoisted]
        return node

    @staticmethod
    def _cls_constructor_calls(member):
        """The `cls(...)` calls of a @classmethod whose `cls` is never rebound, else []."""
        decorators = member.decorator_list if isinstance(member, ast.FunctionDef) else []
        if not (len(decorators) == 1 and isinstance(decorators[0], ast.Name)
                and decorators[0].id == "classmethod"):
            return []
        params = member.args.posonlyargs + member.args.args
        if not params or params[0].arg != "cls":
            return []
        named = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler,
                 ast.MatchAs, ast.MatchStar)
        calls = []
        for n in ast.walk(member):
            if ((isinstance(n, ast.arg) and n.arg == "cls" and n is not params[0]) or
                (isinstance(n, ast.Name) and n.id == "cls" and not isinstance(n.ctx, ast.Load))
                    or (isinstance(n, named) and n.name == "cls")
                    or (isinstance(n, ast.alias) and (n.asname or n.name) == "cls")
                    or (isinstance(n, (ast.Global, ast.Nonlocal)) and "cls" in n.names)):
                return []
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "cls":
                calls.append(n)
        return calls

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

    def _bind_classmethod_constructors(self, module):
        """Rewrite `cls(...)` in a @classmethod to the class the method runs on.

        Only in the entry module, and only when that class is known statically: every
        reference to the method is a direct `Class.method(...)` call and the classes
        involved use single inheritance. A subclass inheriting the method gets its own
        copy. Anything else keeps `cls(...)`, which the converter rejects.
        """
        if not self.is_entry_module:
            return
        classes = {}
        for stmt in module.body:
            if isinstance(stmt, ast.ClassDef):
                if stmt.name in classes:
                    return
                classes[stmt.name] = stmt

        unsafe = self._unbindable_classes(classes)
        indirect = self._indirect_attributes(module, classes)
        for owner in classes.values():
            if owner.name in unsafe:
                continue
            for member in list(owner.body):
                if not self._cls_constructor_calls(member) or member.name in indirect:
                    continue
                heirs = [
                    c for c in classes.values()
                    if c is not owner and self._method_owner(classes, c, member.name) is owner
                ]
                args = member.args
                if heirs and (args.defaults or any(d is not None for d in args.kw_defaults)):
                    continue
                template = copy.deepcopy(member)
                self._rebind_cls_calls(member, owner.name)
                for heir in heirs:
                    method = copy.deepcopy(template)
                    self._rebind_cls_calls(method, heir.name)
                    heir.body.append(method)

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

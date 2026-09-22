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
        self._bind_classmethod_constructors(node)
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
    def _classmethod_receiver(member):
        """Name of a @classmethod's first parameter, or None."""
        if not isinstance(member, ast.FunctionDef):
            return None
        if not any(
                isinstance(d, ast.Name) and d.id == "classmethod" for d in member.decorator_list):
            return None
        positional = member.args.posonlyargs + member.args.args
        return positional[0].arg if positional else None

    def _bind_classmethod_constructors(self, class_node):
        """Rewrite `cls(...)` inside a @classmethod to a call of the class it runs on.

        A subclass that inherits such a method without overriding it gets its
        own copy bound to the subclass, so `Sub.make()` constructs a `Sub`.
        """
        if not hasattr(self, "_cls_constructing_methods"):
            self._cls_constructing_methods = {}

        own = {m.name for m in class_node.body if isinstance(m, ast.FunctionDef)}
        templates = {}
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                for name, method in self._cls_constructing_methods.get(base.id, {}).items():
                    if name not in own and name not in templates:
                        templates[name] = method
                        class_node.body.append(copy.deepcopy(method))

        for member in class_node.body:
            receiver = self._classmethod_receiver(member)
            if receiver is None:
                continue
            calls = [
                n for n in ast.walk(member) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == receiver
            ]
            if not calls:
                continue
            templates.setdefault(member.name, copy.deepcopy(member))
            for call in calls:
                call.func = ast.copy_location(ast.Name(id=class_node.name, ctx=ast.Load()),
                                              call.func)

        if templates:
            self._cls_constructing_methods[class_node.name] = templates

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

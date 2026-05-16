import ast
# pylint: disable=too-many-boolean-expressions


class ClassContextMixin:

    def visit_ClassDef(self, node):
        """Track class context for method definitions."""
        old_class_name = getattr(self, "current_class_name", None)
        self.current_class_name = node.name

        node = self.expand_dataclass(node)
        self._collect_class_attr_annotations(node)
        self._record_exit_suppresses_all(node)
        self._record_class_with_exit(node)
        self.generic_visit(node)

        self.current_class_name = old_class_name
        return node

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

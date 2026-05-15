import ast


class ClassContextMixin:

    def visit_ClassDef(self, node):
        """Track class context for method definitions."""
        old_class_name = getattr(self, "current_class_name", None)
        self.current_class_name = node.name

        node = self.expand_dataclass(node)
        self._collect_class_attr_annotations(node)
        self._record_exit_suppresses_all(node)
        self.generic_visit(node)

        self.current_class_name = old_class_name
        return node

    def _record_exit_suppresses_all(self, class_node):
        """Cache classes whose __exit__ unconditionally returns True."""
        if not hasattr(self, "_exit_suppresses_all"):
            self._exit_suppresses_all = set()
        for member in class_node.body:
            if (isinstance(member, ast.FunctionDef) and member.name == "__exit__"
                    and len(member.body) == 1 and isinstance(member.body[0], ast.Return)
                    and isinstance(member.body[0].value, ast.Constant)
                    and member.body[0].value.value is True):
                self._exit_suppresses_all.add(class_node.name)
                return

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

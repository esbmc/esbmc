import ast


class AssignmentCoreMixin:

    def _is_newtype_call(self, call_node):
        """True if call_node is a typing.NewType(...) call, in any import form."""
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id in self.newtype_names
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.attr == "NewType"):
            return func.value.id in self.typing_module_names
        return False

    def _as_load_target(self, target, source_node):
        """Create a Load-context version of an AugAssign target."""
        if isinstance(target, ast.Name):
            load_target = ast.Name(id=target.id, ctx=ast.Load())
        elif isinstance(target, ast.Subscript):
            load_target = ast.Subscript(value=target.value, slice=target.slice, ctx=ast.Load())
        elif isinstance(target, ast.Attribute):
            load_target = ast.Attribute(value=target.value, attr=target.attr, ctx=ast.Load())
        else:
            load_target = target
        return self.ensure_all_locations(load_target, source_node)

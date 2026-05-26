import ast


class AstUtilsMixin:

    @staticmethod
    def _sanitize_identifier_fragment(fragment):
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in fragment)

    def ensure_all_locations(self, node, source_node=None, line=1, col=0):
        """Recursively ensure all nodes in an AST tree have location information."""
        if source_node:
            line = getattr(source_node, "lineno", 1)
            col = getattr(source_node, "col_offset", 0)

        if not hasattr(node, "lineno") or node.lineno is None:
            node.lineno = line
        if not hasattr(node, "col_offset") or node.col_offset is None:
            node.col_offset = col

        for child in ast.iter_child_nodes(node):
            self.ensure_all_locations(child, source_node, line, col)

        return node

    def create_name_node(self, name_id, ctx, source_node=None):
        """Create a Name node with proper location info."""
        node = ast.Name(id=name_id, ctx=ctx)
        return self.ensure_all_locations(node, source_node)

    def create_constant_node(self, value, source_node=None):
        """Create a Constant node with proper location info."""
        node = ast.Constant(value=value)
        return self.ensure_all_locations(node, source_node)

    def _rename_loads(self, node, old_name, new_name):

        class _RenameLoad(ast.NodeTransformer):

            def __init__(self, old_name, new_name):
                self.old_name = old_name
                self.new_name = new_name

            def visit_Name(self, name_node):
                if name_node.id == self.old_name and isinstance(name_node.ctx, ast.Load):
                    return ast.copy_location(ast.Name(id=self.new_name, ctx=ast.Load()), name_node)
                return name_node

        renamed = _RenameLoad(old_name, new_name).visit(node)
        ast.fix_missing_locations(renamed)
        return renamed

    def _bound_target_names(self, target):
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            names = set()
            for elt in target.elts:
                names.update(self._bound_target_names(elt))
            return names
        return set()

    def _create_bool_ann_assign(self, target_name, value, source_node):
        assign = ast.AnnAssign(
            target=self.create_name_node(target_name, ast.Store(), source_node),
            annotation=self.create_name_node("bool", ast.Load(), source_node),
            value=value,
            simple=1,
        )
        self.ensure_all_locations(assign, source_node)
        ast.fix_missing_locations(assign)
        return assign

    def generate_variable_copy(self, qualified_name, arg_node, default_name_node):
        """
        Materialize a Name default value into a stable temporary variable.
        """
        func_part = self._sanitize_identifier_fragment(qualified_name)
        arg_part = self._sanitize_identifier_fragment(arg_node.arg)
        target_var = f"ESBMC_default_{func_part}_{arg_part}"

        assignment_node = ast.Assign(
            targets=[self.create_name_node(target_var, ast.Store(), default_name_node)],
            value=ast.Name(id=default_name_node.id, ctx=ast.Load()),
        )
        self.ensure_all_locations(assignment_node, default_name_node)
        ast.fix_missing_locations(assignment_node)
        target_ref = self.create_name_node(target_var, ast.Load(), default_name_node)
        ast.fix_missing_locations(target_ref)
        return assignment_node, target_ref

    @staticmethod
    def _body_has_node_shallow(body_stmts, node_type):
        """Return True if node_type appears in body_stmts without nested defs/lambdas."""

        def _walk(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return
            if isinstance(node, node_type):
                yield node
            for child in ast.iter_child_nodes(node):
                yield from _walk(child)

        module = ast.Module(body=list(body_stmts), type_ignores=[])
        return any(True for _ in _walk(module))

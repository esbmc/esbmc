import ast


class ModuleLifecycleMixin:

    def finalize_module(self, node):
        """Run generic_visit and inject helper nodes requested during traversal."""
        # Per-module scope for the eq-only set and call-origin map.
        saved_eq_only = set(self._eq_only_items_view_targets)
        self._eq_only_items_view_targets = (
            self._scan_eq_only_items_view_targets(node.body))
        saved_call_origins = dict(self._assignment_call_origins)
        self._assignment_call_origins.clear()
        try:
            node = self.generic_visit(node)

            if self._needs_dataclass_initvar_import:
                self._ensure_dataclass_initvar_import(node)

            if self.helper_functions_added:
                helper_functions = self._create_helper_functions()
                for func in helper_functions:
                    self.ensure_all_locations(func)
                    ast.fix_missing_locations(func)
                node.body = helper_functions + node.body

            if self._needs_dataclass_field_helper:
                helper_class = self._build_dataclass_field_helper_class(node)
                node.body = [helper_class] + node.body

            if self._needs_dataclass_replace_error_helper:
                helper_fn = self._build_dataclass_replace_error_helper(node)
                node.body = [helper_fn] + node.body

            if self._needs_dataclass_getattr_helper:
                helper_fn = self._build_dataclass_getattr_helper(node)
                node.body = [helper_fn] + node.body

            return node
        finally:
            self._eq_only_items_view_targets = saved_eq_only
            self._assignment_call_origins = saved_call_origins

    def visit_Module(self, node):
        """Back-compat entry point for callers not using import-aware seeds."""
        self.prepare_module(node)
        return self.finalize_module(node)

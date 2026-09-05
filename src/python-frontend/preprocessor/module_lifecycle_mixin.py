import ast
import pytype_infer


class ModuleLifecycleMixin:

    def adopt_module_signatures(self, other, imported_names, include_methods=False):
        """Learn an imported module's call signatures and defaults.

        Each module gets its own Preprocessor, so a call to an imported
        function is otherwise converted without the arguments its signature
        defaults would supply.

        Only entries owned by a name in `imported_names` are taken; the tables
        are keyed by bare name, so adopting wholesale would let another
        module's `__init__` answer arity checks for a local class. Methods are
        taken only when `include_methods` is set, which excludes the operational
        models: those deliberately simplify a constructor, and checking calls
        against one would reject calls the converter handles (#4665).
        Own definitions win: a locally defined name keeps its own signature.
        """
        for table, source in (
            (self.functionParams, other.functionParams),
            (self.functionKwonlyParams, other.functionKwonlyParams),
            (self.functionDefaults, other.functionDefaults),
        ):
            for key, value in source.items():
                name = key[0] if isinstance(key, tuple) else key
                owner, dot, _ = name.partition(".")
                if owner not in imported_names or (dot and not include_methods):
                    continue
                table.setdefault(key, value)

    def finalize_module(self, node):
        """Run generic_visit and inject helper nodes requested during traversal."""
        # Per-module scope for the eq-only set and call-origin map.
        saved_eq_only = set(self._eq_only_items_view_targets)
        self._eq_only_items_view_targets = (self._scan_eq_only_items_view_targets(node.body))
        saved_call_origins = dict(self._assignment_call_origins)
        self._assignment_call_origins.clear()
        try:
            node = self.generic_visit(node)
                   # try:
            ast.fix_missing_locations(node)
           # print("Before pytype module annotations:")
           # print(ast.unparse(node))
            node =  pytype_infer.annotate_tree(node)
          #  print("After pytype annotation module:")
          #  print(ast.unparse(node))

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

            self._inject_vararg_specializations(node)

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

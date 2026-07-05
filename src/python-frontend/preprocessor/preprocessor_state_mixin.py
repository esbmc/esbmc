"""State initialization for composed preprocessor mixins."""


class PreprocessorStateMixin:

    def _init_preprocessor_state(self, module_name):  # pylint: disable=too-many-statements
        self.target_name = ""
        self.functionDefaults = {}
        self.functionParams = {}
        self.module_name = module_name
        self.is_range_loop = False
        self.known_variable_types = {}
        self.range_loop_counter = 0
        self.iterable_loop_counter = 0
        self.enumerate_loop_counter = 0
        self.nondet_expand_counter = 0
        self.helper_functions_added = False
        self.functionKwonlyParams = {}
        self.listcomp_counter = 0
        self.variable_annotations = {}
        self.function_return_annotations = {}
        self.class_attr_annotations = {}
        self.instance_class_map = {}
        self.attr_list_element_classes = {}
        self.decimal_imported = False
        self.decimal_module_imported = False
        self.decimal_class_alias = None
        self.decimal_module_alias = None
        self.defaultdict_imported = False
        self.defaultdict_alias = None
        self.collections_module_imported = False
        self.collections_module_alias = None
        self._subscript_inferred_vars = set()
        self.generator_funcs = set()
        self.early_return_generator_funcs = set()
        self.generator_vars = {}
        self.generator_func_defs = {}
        self.generator_next_index = {}
        self.generator_emitted_init = set()
        self.dict_items_vars = {}
        self._defaultdict_factory = {}
        self._defaultdict_initialized_keys = {}
        self.het_dict_literals = {}
        self.het_value_dict_literals = {}
        # Names currently bound to a dict literal (any key types). Used to
        # rewrite list(d)/sorted(d) into the correctly-typed d.keys() path.
        self.dict_literal_vars = set()
        # Pre-pass results for unannotated parameter-dict element recovery
        # (#5444): function name -> list of per-call positional shape lists,
        # where each entry is the inferred dict[K, V] annotation of that
        # argument (or None when unknown). Argument names are resolved within
        # the scope of their call site so a function-local dict never poisons a
        # same-named module global. Recovery succeeds only when every call site
        # agrees (see _recover_param_dict_annotation).
        self._dict_param_call_shapes = {}
        self.bound_method_vars = {}
        self.called_names = set()
        self.list_literal_values = {}
        # Map var -> RHS Call node; used by _apply_assert_eq_rewrites to
        # substitute the Name back to its defining call.
        self._assignment_call_origins = {}
        # Items-view target names safe to neutralise to `[]` at Assign time.
        # Populated by _scan_eq_only_items_view_targets per scope.
        self._eq_only_items_view_targets = set()
        self.newtype_vars = set()
        self.newtype_names = {"NewType"}
        self.typing_module_names = set()
        self._typing_imported_names = set()
        self._with_counter = 0
        self._unroll_counter = 0
        self.type_aliases = {}
        self._dataclass_decorator_names = {"dataclass"}
        self._dataclass_field_names = {"field"}
        self._dataclass_initvar_names = {"InitVar"}
        self._dataclass_is_dataclass_names = {"is_dataclass"}
        self._dataclass_fields_api_names = {"fields"}
        self._dataclass_asdict_names = {"asdict"}
        self._dataclass_astuple_names = {"astuple"}
        self._dataclass_replace_names = {"replace"}
        self.dataclasses_module_names = {"dataclasses"}
        self._typing_classvar_names = {"ClassVar"}
        self._classes_with_post_init = set()
        self._dataclass_class_specs = {}
        self._needs_dataclass_field_helper = False
        self._needs_dataclass_replace_error_helper = False
        self._needs_dataclass_getattr_helper = False
        self._needs_dataclass_initvar_import = False
        self._assert_eq_counter = 0
        self._known_literal_values = {}
        self._identity_functions = set()
        self.exported_range_aliases = set()
        self.exported_range_wrappers = {}
        self.module_dunder_all = None
        self._pending_method_default_inits = []

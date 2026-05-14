"""DataclassMixin — extracted from preprocessor.py.

Contains all dataclass-expansion logic: detection, field collection,
dunder generation (__init__, __repr__, __eq__, __hash__, ordering),
AST helpers, the ``dataclasses`` API rewrite (asdict/astuple/fields/
is_dataclass/replace) and ``__post_init__`` validation.

All shared state lives on Preprocessor (set in Preprocessor.__init__);
this mixin only adds methods.
"""
import ast
import copy

# Internal helper variable names used by the runtime dataclass walker
# emitted by ``_build_runtime_recursive_dataclass_expr``.
_WALK = "__dataclass_walk"
_VALUE = "__dataclass_value"
_FIELD = "__dataclass_field"
_ITEM = "__dataclass_item"
_KEY = "__dataclass_key"


class DataclassMixin:
    def _ensure_dataclass_initvar_import(self, module_node):
        for stmt in module_node.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "dataclasses":
                if any(alias.name in ("InitVar", "*") for alias in stmt.names):
                    return

        import_stmt = ast.ImportFrom(
            module="dataclasses",
            names=[ast.alias(name="InitVar", asname=None)],
            level=0,
        )
        self.ensure_all_locations(import_stmt, module_node)
        ast.fix_missing_locations(import_stmt)

        insert_index = 0
        if module_node.body:
            first_stmt = module_node.body[0]
            if isinstance(first_stmt, ast.Expr) and (
                (
                    isinstance(first_stmt.value, ast.Constant)
                    and isinstance(first_stmt.value.value, str)
                )
                or isinstance(first_stmt.value, ast.Str)
            ):
                insert_index = 1

        while insert_index < len(module_node.body):
            stmt = module_node.body[insert_index]
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                insert_index += 1
                continue
            break

        module_node.body.insert(insert_index, import_stmt)

    def _build_dataclass_field_helper_class(self, source_node):
        init_fn = ast.FunctionDef(
            name="__init__",
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg="self", annotation=None),
                    ast.arg(arg="name", annotation=ast.Name(id="str", ctx=ast.Load())),
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[
                ast.AnnAssign(
                    target=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="name",
                        ctx=ast.Store(),
                    ),
                    annotation=ast.Name(id="str", ctx=ast.Load()),
                    value=ast.Name(id="name", ctx=ast.Load()),
                    simple=0,
                )
            ],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        cls = ast.ClassDef(
            name="__ESBMC_DataclassField",
            bases=[],
            keywords=[],
            body=[init_fn],
            decorator_list=[],
        )
        self.ensure_all_locations(cls, source_node)
        ast.fix_missing_locations(cls)
        return cls

    def _build_dataclass_replace_error_helper(self, source_node):
        fn = ast.FunctionDef(
            name="__ESBMC_dataclass_replace_invalid_field",
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="TypeError", ctx=ast.Load()),
                        args=[
                            ast.Constant(value="replace() got an unexpected field name")
                        ],
                        keywords=[],
                    ),
                    cause=None,
                )
            ],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        self.ensure_all_locations(fn, source_node)
        ast.fix_missing_locations(fn)
        return fn

    def _build_dataclass_getattr_helper(self, source_node):
        fn = ast.FunctionDef(
            name="__ESBMC_dataclass_getattr",
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg="obj", annotation=None),
                    ast.arg(arg="name", annotation=ast.Name(id="str", ctx=ast.Load())),
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[
                ast.Return(
                    value=ast.Call(
                        func=ast.Name(id="getattr", ctx=ast.Load()),
                        args=[
                            ast.Name(id="obj", ctx=ast.Load()),
                            ast.Name(id="name", ctx=ast.Load()),
                        ],
                        keywords=[],
                    )
                )
            ],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        self.ensure_all_locations(fn, source_node)
        ast.fix_missing_locations(fn)
        return fn

    # Mapping from ``dataclasses`` attribute name to the API kind used by
    # ``_dataclass_api_kind`` / ``_rewrite_dataclass_api_call``. The bare-name
    # path additionally consults the per-instance alias sets registered in
    # ``__init__``.
    _DATACLASS_API_ATTR_KINDS = {
        "is_dataclass": "is_dataclass",
        "fields": "fields",
        "asdict": "asdict",
        "astuple": "astuple",
        "replace": "replace",
    }

    def _dataclass_api_kind_from_name(self, name):
        if name in self._dataclass_is_dataclass_names:
            return "is_dataclass"
        if name in self._dataclass_fields_api_names:
            return "fields"
        if name in self._dataclass_asdict_names:
            return "asdict"
        if name in self._dataclass_astuple_names:
            return "astuple"
        if name in self._dataclass_replace_names:
            return "replace"
        return None

    def _dataclass_api_kind(self, node):
        if isinstance(node.func, ast.Name):
            return self._dataclass_api_kind_from_name(node.func.id)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.dataclasses_module_names
        ):
            return self._DATACLASS_API_ATTR_KINDS.get(node.func.attr)
        return None

    def _dataclass_instance_type_of_expr(self, expr):
        if isinstance(expr, ast.Name):
            expr_name = expr.id
            if expr_name in self.known_variable_types:
                candidate = self.known_variable_types[expr_name]
                if candidate in self._dataclass_class_specs:
                    return candidate
            if expr_name in self.instance_class_map:
                candidate = self.instance_class_map[expr_name]
                if candidate in self._dataclass_class_specs:
                    return candidate
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            if expr.func.id in self._dataclass_class_specs:
                return expr.func.id
        return None

    def _dataclass_class_of_expr(self, expr):
        if isinstance(expr, ast.Name) and expr.id in self._dataclass_class_specs:
            return expr.id
        return None

    def _annotation_class_name(self, annotation):
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Subscript):
            base = annotation.value
            if isinstance(base, ast.Name):
                return base.id
            if isinstance(base, ast.Attribute):
                return base.attr
        if isinstance(annotation, ast.Attribute):
            return annotation.attr
        return None

    def _dataclass_field_specs(self, class_name):
        if class_name not in self._dataclass_class_specs:
            return []
        return [
            field
            for field in self._dataclass_class_specs[class_name]["fields"]
            if field["kind"] == "instance"
        ]

    def _build_recursive_dataclass_call(self, walk_name, value_expr, source_node):
        call_expr = ast.Call(
            func=ast.Name(id=walk_name, ctx=ast.Load()),
            args=[
                ast.Name(id=walk_name, ctx=ast.Load()),
                value_expr,
            ],
            keywords=[],
        )
        self.ensure_all_locations(call_expr, source_node)
        ast.fix_missing_locations(call_expr)
        return call_expr

    def _build_static_known_dataclass_expr(
        self, kind, class_name, instance_expr, source_node
    ):
        if class_name not in self._dataclass_class_specs:
            return None

        field_specs = self._dataclass_field_specs(class_name)
        container_names = {
            "list",
            "dict",
            "tuple",
            "set",
            "List",
            "Dict",
            "Tuple",
            "Set",
        }

        dict_keys = []
        dict_values = []
        tuple_elts = []

        for field in field_specs:
            field_value = ast.Attribute(
                value=copy.deepcopy(instance_expr),
                attr=field["name"],
                ctx=ast.Load(),
            )

            field_type = self._annotation_class_name(field["annotation"])
            if field_type in container_names:
                # Keep the runtime walker for container-typed fields so nested
                # list/dict/tuple recursion semantics stay unchanged.
                return None

            if field_type in self._dataclass_class_specs:
                value_expr = self._build_static_known_dataclass_expr(
                    kind, field_type, field_value, source_node
                )
                if value_expr is None:
                    return None
            else:
                value_expr = field_value

            if kind == "asdict":
                dict_keys.append(ast.Constant(value=field["name"]))
                dict_values.append(value_expr)
            else:
                tuple_elts.append(value_expr)

        if kind == "asdict":
            result = ast.Dict(keys=dict_keys, values=dict_values)
        else:
            result = ast.Tuple(elts=tuple_elts, ctx=ast.Load())

        self.ensure_all_locations(result, source_node)
        ast.fix_missing_locations(result)
        return result

    def _walk_recurse_on_item(self, source_node):
        return self._build_recursive_dataclass_call(
            _WALK,
            ast.Name(id=_ITEM, ctx=ast.Load()),
            source_node,
        )

    def _build_walk_dict_comp(self, source_node):
        node = ast.DictComp(
            key=ast.Name(id=_KEY, ctx=ast.Load()),
            value=self._walk_recurse_on_item(source_node),
            generators=[
                ast.comprehension(
                    target=ast.Tuple(
                        elts=[
                            ast.Name(id=_KEY, ctx=ast.Store()),
                            ast.Name(id=_ITEM, ctx=ast.Store()),
                        ],
                        ctx=ast.Store(),
                    ),
                    iter=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=_VALUE, ctx=ast.Load()),
                            attr="items",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                    ifs=[],
                    is_async=0,
                )
            ],
        )
        self.ensure_all_locations(node, source_node)
        ast.fix_missing_locations(node)
        return node

    def _build_walk_iter_call(self, builder, source_node):
        """Return ``builder(elt)`` over the items of ``_VALUE``.

        ``builder`` must be ``ast.ListComp`` or wrap ``ast.GeneratorExp``;
        the generator iterates ``_VALUE`` binding ``_ITEM`` on each step.
        """
        elt = self._walk_recurse_on_item(source_node)
        gen = ast.comprehension(
            target=ast.Name(id=_ITEM, ctx=ast.Store()),
            iter=ast.Name(id=_VALUE, ctx=ast.Load()),
            ifs=[],
            is_async=0,
        )
        node = builder(elt, gen)
        self.ensure_all_locations(node, source_node)
        ast.fix_missing_locations(node)
        return node

    def _build_walk_dataclass_expr(self, kind, source_node):
        self._needs_dataclass_getattr_helper = True
        dataclass_iter = ast.Call(
            func=ast.Name(id="fields", ctx=ast.Load()),
            args=[ast.Name(id=_VALUE, ctx=ast.Load())],
            keywords=[],
        )
        self.ensure_all_locations(dataclass_iter, source_node)
        field_name_attr = ast.Attribute(
            value=ast.Name(id=_FIELD, ctx=ast.Load()),
            attr="name",
            ctx=ast.Load(),
        )
        recursive_value = self._build_recursive_dataclass_call(
            _WALK,
            ast.Call(
                func=ast.Name(id="__ESBMC_dataclass_getattr", ctx=ast.Load()),
                args=[
                    ast.Name(id=_VALUE, ctx=ast.Load()),
                    ast.Attribute(
                        value=ast.Name(id=_FIELD, ctx=ast.Load()),
                        attr="name",
                        ctx=ast.Load(),
                    ),
                ],
                keywords=[],
            ),
            source_node,
        )
        gen = ast.comprehension(
            target=ast.Name(id=_FIELD, ctx=ast.Store()),
            iter=dataclass_iter,
            ifs=[],
            is_async=0,
        )
        if kind == "asdict":
            node = ast.DictComp(
                key=field_name_attr, value=recursive_value, generators=[gen]
            )
        else:
            node = ast.Call(
                func=ast.Name(id="tuple", ctx=ast.Load()),
                args=[ast.GeneratorExp(elt=recursive_value, generators=[gen])],
                keywords=[],
            )
        self.ensure_all_locations(node, source_node)
        ast.fix_missing_locations(node)
        return node

    @staticmethod
    def _walk_isinstance_ifexp(type_name, body, orelse):
        return ast.IfExp(
            test=ast.Call(
                func=ast.Name(id="isinstance", ctx=ast.Load()),
                args=[
                    ast.Name(id=_VALUE, ctx=ast.Load()),
                    ast.Name(id=type_name, ctx=ast.Load()),
                ],
                keywords=[],
            ),
            body=body,
            orelse=orelse,
        )

    @staticmethod
    def _walker_two_arg_args():
        return ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=_WALK), ast.arg(arg=_VALUE)],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )

    def _build_runtime_recursive_dataclass_expr(self, kind, instance_expr, source_node):
        # Build the per-shape branches that the walker dispatches on.
        dataclass_branch = self._build_walk_dataclass_expr(kind, source_node)
        dict_branch = self._build_walk_dict_comp(source_node)
        tuple_branch = self._build_walk_iter_call(
            lambda elt, gen: ast.Call(
                func=ast.Name(id="tuple", ctx=ast.Load()),
                args=[ast.GeneratorExp(elt=elt, generators=[gen])],
                keywords=[],
            ),
            source_node,
        )
        list_branch = self._build_walk_iter_call(
            lambda elt, gen: ast.ListComp(elt=elt, generators=[gen]),
            source_node,
        )

        # Chain the branches: list ? tuple ? dict ? dataclass ? value
        current = ast.Name(id=_VALUE, ctx=ast.Load())
        current = ast.IfExp(
            test=ast.Call(
                func=ast.Name(id="is_dataclass", ctx=ast.Load()),
                args=[ast.Name(id=_VALUE, ctx=ast.Load())],
                keywords=[],
            ),
            body=dataclass_branch,
            orelse=current,
        )
        current = self._walk_isinstance_ifexp("dict", dict_branch, current)
        current = self._walk_isinstance_ifexp("tuple", tuple_branch, current)
        current = self._walk_isinstance_ifexp("list", list_branch, current)

        # Wrap the body in lambdas to allow recursive dispatch via Y-combinator.
        walker_lambda = ast.Lambda(args=self._walker_two_arg_args(), body=current)
        wrapper_lambda = ast.Lambda(
            args=self._walker_two_arg_args(),
            body=ast.Call(
                func=ast.Name(id=_WALK, ctx=ast.Load()),
                args=[
                    ast.Name(id=_WALK, ctx=ast.Load()),
                    ast.Name(id=_VALUE, ctx=ast.Load()),
                ],
                keywords=[],
            ),
        )
        call_expr = ast.Call(
            func=wrapper_lambda,
            args=[walker_lambda, copy.deepcopy(instance_expr)],
            keywords=[],
        )
        self.ensure_all_locations(call_expr, source_node)
        ast.fix_missing_locations(call_expr)
        return call_expr

    def _build_asdict_expr(self, class_name, instance_expr, source_node):
        static_expr = self._build_static_known_dataclass_expr(
            "asdict", class_name, instance_expr, source_node
        )
        if static_expr is not None:
            return static_expr

        dict_expr = self._build_runtime_recursive_dataclass_expr(
            "asdict", instance_expr, source_node
        )
        self.ensure_all_locations(dict_expr, source_node)
        ast.fix_missing_locations(dict_expr)
        return dict_expr

    def _build_astuple_expr(self, class_name, instance_expr, source_node):
        static_expr = self._build_static_known_dataclass_expr(
            "astuple", class_name, instance_expr, source_node
        )
        if static_expr is not None:
            return static_expr

        tuple_expr = self._build_runtime_recursive_dataclass_expr(
            "astuple", instance_expr, source_node
        )
        self.ensure_all_locations(tuple_expr, source_node)
        ast.fix_missing_locations(tuple_expr)
        return tuple_expr

    def _rewrite_dataclass_replace_call(self, instance_class, target, node):
        """Rewrite ``replace(obj, k=v, ...)`` into ``Cls(...)`` when possible.

        Returns ``node`` unchanged if a ``**kwargs`` splat prevents resolution,
        or an error-helper call when an unknown field is referenced.
        """
        if any(kw.arg is None for kw in node.keywords):
            return node
        field_specs = self._dataclass_field_specs(instance_class)
        allowed_names = {field["name"] for field in field_specs}
        keyword_names = {kw.arg for kw in node.keywords}
        if keyword_names and not keyword_names.issubset(allowed_names):
            self._needs_dataclass_replace_error_helper = True
            result = ast.Call(
                func=ast.Name(
                    id="__ESBMC_dataclass_replace_invalid_field", ctx=ast.Load()
                ),
                args=[],
                keywords=[],
            )
            self.ensure_all_locations(result, node)
            ast.fix_missing_locations(result)
            return result

        replacements = {kw.arg: kw.value for kw in node.keywords}
        ctor_args = []
        for field in field_specs:
            field_name = field["name"]
            if field_name in replacements:
                ctor_args.append(replacements[field_name])
            else:
                ctor_args.append(
                    ast.Attribute(
                        value=copy.deepcopy(target),
                        attr=field_name,
                        ctx=ast.Load(),
                    )
                )
        result = ast.Call(
            func=ast.Name(id=instance_class, ctx=ast.Load()),
            args=ctor_args,
            keywords=[],
        )
        self.ensure_all_locations(result, node)
        ast.fix_missing_locations(result)
        return result

    def _rewrite_dataclass_fields_call(self, node, effective_class):
        """Rewrite ``fields(Cls or instance)`` into a literal list of fields."""
        self._needs_dataclass_field_helper = True
        field_objs = [
            ast.Call(
                func=ast.Name(id="__ESBMC_DataclassField", ctx=ast.Load()),
                args=[ast.Constant(value=field["name"])],
                keywords=[],
            )
            for field in self._dataclass_field_specs(effective_class)
        ]
        result = ast.List(elts=field_objs, ctx=ast.Load())
        self.ensure_all_locations(result, node)
        ast.fix_missing_locations(result)
        return result

    def _rewrite_is_dataclass_call(self, node, target, class_name, instance_class):
        if class_name is not None or instance_class is not None:
            result = ast.Constant(value=True)
            self.ensure_all_locations(result, node)
            return result
        if isinstance(target, ast.Constant):
            result = ast.Constant(value=False)
            self.ensure_all_locations(result, node)
            return result
        return node

    # Handlers for ``replace(obj, ...)``-style APIs that need a resolved
    # ``instance_class``. Looked up by ``_rewrite_dataclass_api_call``.
    _DATACLASS_INSTANCE_HANDLERS = {
        "asdict": "_build_asdict_expr",
        "astuple": "_build_astuple_expr",
        "replace": "_rewrite_dataclass_replace_call",
    }

    def _rewrite_dataclass_fields_kind(self, node, class_name, instance_class):
        effective_class = class_name or instance_class
        if effective_class is None:
            return node
        return self._rewrite_dataclass_fields_call(node, effective_class)

    def _rewrite_dataclass_instance_kind(self, kind, node, target, instance_class):
        if instance_class is None:
            return node
        handler_name = self._DATACLASS_INSTANCE_HANDLERS.get(kind)
        if handler_name is None:
            return node
        return getattr(self, handler_name)(instance_class, target, node)

    def _rewrite_dataclass_api_call(self, node):
        kind = self._dataclass_api_kind(node)
        if kind is None or len(node.args) != 1:
            return node

        target = node.args[0]
        class_name = self._dataclass_class_of_expr(target)
        instance_class = self._dataclass_instance_type_of_expr(target)

        if kind == "is_dataclass":
            return self._rewrite_is_dataclass_call(
                node, target, class_name, instance_class
            )
        if kind == "fields":
            return self._rewrite_dataclass_fields_kind(
                node, class_name, instance_class
            )
        return self._rewrite_dataclass_instance_kind(
            kind, node, target, instance_class
        )

    def is_dataclass(self, class_node):
        """Return True when a class is decorated with @dataclass.

        Recognizes the canonical ``@dataclass`` form, the qualified
        ``@dataclasses.dataclass`` form, and any local alias introduced via
        ``from dataclasses import dataclass as <alias>`` (tracked in
        ``self._dataclass_decorator_names``).
        """
        for decorator in class_node.decorator_list:
            target = decorator
            if isinstance(decorator, ast.Call):
                target = decorator.func

            if (
                isinstance(target, ast.Name)
                and target.id in self._dataclass_decorator_names
            ):
                return True

            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "dataclasses"
                and target.attr == "dataclass"
            ):
                return True

        return False

    def _unwrap_annotation_slice(self, annotation):
        if not isinstance(annotation, ast.Subscript):
            return None
        if isinstance(annotation.slice, ast.Index):
            return annotation.slice.value
        return annotation.slice

    def _is_dataclass_initvar_base(self, node):
        return (
            isinstance(node, ast.Name) and node.id in self._dataclass_initvar_names
        ) or (
            isinstance(node, ast.Attribute)
            and node.attr == "InitVar"
            and isinstance(node.value, ast.Name)
            and node.value.id in self.dataclasses_module_names
        )

    def _is_typing_classvar_base(self, node):
        return (
            isinstance(node, ast.Name) and node.id in self._typing_classvar_names
        ) or (
            isinstance(node, ast.Attribute)
            and node.attr == "ClassVar"
            and isinstance(node.value, ast.Name)
            and node.value.id in self.typing_module_names
        )

    def _analyze_dataclass_field_annotation(self, annotation):
        """Classify a dataclass annotation as instance field, InitVar or ClassVar."""
        if annotation is None:
            return "instance", None

        if isinstance(annotation, ast.Subscript):
            if self._is_dataclass_initvar_base(annotation.value):
                inner = self._unwrap_annotation_slice(annotation)
                return "initvar", self._copy_annotation_node(inner)
            if self._is_typing_classvar_base(annotation.value):
                inner = self._unwrap_annotation_slice(annotation)
                return "classvar", self._copy_annotation_node(inner)

        if self._is_dataclass_initvar_base(annotation):
            return "initvar", None
        if self._is_typing_classvar_base(annotation):
            return "classvar", None
        return "instance", annotation

    def _get_post_init_method(self, class_node):
        return next(
            (
                member
                for member in class_node.body
                if isinstance(member, ast.FunctionDef)
                and member.name == "__post_init__"
            ),
            None,
        )

    def _class_has_post_init_behavior(self, class_node):
        if self._get_post_init_method(class_node) is not None:
            return True

        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in self._classes_with_post_init:
                return True
        return False

    def _class_defines_name(self, class_node, name):
        for member in class_node.body:
            if isinstance(member, ast.FunctionDef) and member.name == name:
                return True
            if isinstance(member, ast.Assign):
                for target in member.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return True
            if isinstance(member, ast.AnnAssign):
                if isinstance(member.target, ast.Name) and member.target.id == name:
                    return True
        return False

    def _validate_post_init_signature(self, class_node, fields, post_init_method):
        """Validate that ``__post_init__`` can receive the declared InitVar values."""
        if post_init_method is None:
            return

        initvar_count = sum(1 for field in fields if field["kind"] == "initvar")
        total_positional = len(post_init_method.args.posonlyargs) + len(
            post_init_method.args.args
        )
        if total_positional == 0:
            raise SyntaxError(
                f"dataclass {class_node.name!r} has invalid __post_init__ signature: "
                "missing bound instance parameter"
            )

        positional_after_self = total_positional - 1
        min_positional_after_self = positional_after_self - len(
            post_init_method.args.defaults
        )
        max_positional_after_self = (
            float("inf")
            if post_init_method.args.vararg is not None
            else positional_after_self
        )

        if any(default is None for default in post_init_method.args.kw_defaults):
            raise SyntaxError(
                f"dataclass {class_node.name!r} has incompatible __post_init__ signature: "
                "required keyword-only parameters are not supported"
            )

        if not (
            min_positional_after_self <= initvar_count <= max_positional_after_self
        ):
            raise SyntaxError(
                f"dataclass {class_node.name!r} has incompatible __post_init__ signature: "
                f"expected {initvar_count} InitVar argument(s)"
            )

    def _dataclass_default_options(self):
        return {
            "init": True,
            "repr": True,
            "eq": True,
            "order": False,
            "frozen": False,
            "unsafe_hash": False,
            "kw_only": False,
            "slots": False,
            "match_args": True,
        }

    def _field_spec(self, name, annotation, defaults, options):
        """Return a normalized field spec.

        ``defaults`` is the ``(default_expr, factory_expr)`` pair returned by
        ``_parse_field_call``. ``options`` carries per-field flags plus
        ``"kind"`` (one of ``instance``, ``initvar``, ``classvar``).
        """
        default_expr, factory_expr = defaults
        return {
            "name": name,
            "annotation": annotation,
            "default_expr": default_expr,
            "factory_expr": factory_expr,
            "kind": options["kind"],
            "init": options.get("init", True),
            "repr": options.get("repr", True),
            "compare": options.get("compare", True),
            "hash": options.get("hash", None),
            "kw_only": options.get("kw_only", False),
        }

    def _parse_dataclass_bool_option(self, value, option_name):
        if not isinstance(value, ast.Constant) or not isinstance(value.value, bool):
            raise SyntaxError(
                f"dataclass option {option_name!r} must be a boolean literal"
            )
        return value.value

    def _parse_dataclass_options(self, class_node):
        options = self._dataclass_default_options()
        decorator = next(
            (
                dec
                for dec in class_node.decorator_list
                if (
                    isinstance(dec, ast.Name)
                    and dec.id in self._dataclass_decorator_names
                )
                or (
                    isinstance(dec, ast.Attribute)
                    and isinstance(dec.value, ast.Name)
                    and dec.value.id in self.dataclasses_module_names
                    and dec.attr == "dataclass"
                )
                or (
                    isinstance(dec, ast.Call)
                    and (
                        (
                            isinstance(dec.func, ast.Name)
                            and dec.func.id in self._dataclass_decorator_names
                        )
                        or (
                            isinstance(dec.func, ast.Attribute)
                            and isinstance(dec.func.value, ast.Name)
                            and dec.func.value.id in self.dataclasses_module_names
                            and dec.func.attr == "dataclass"
                        )
                    )
                )
            ),
            None,
        )
        if decorator is None:
            return options
        if not isinstance(decorator, ast.Call):
            return options
        if decorator.args:
            raise SyntaxError(
                "dataclass decorator does not accept positional arguments"
            )
        for kw in decorator.keywords:
            if kw.arg is None:
                raise SyntaxError("dataclass decorator does not support **kwargs")
            if kw.arg not in options:
                raise SyntaxError(f"unsupported dataclass option {kw.arg!r}")
            options[kw.arg] = self._parse_dataclass_bool_option(kw.value, kw.arg)
        if options["order"] and not options["eq"]:
            raise SyntaxError("dataclass option 'order=True' requires 'eq=True'")
        if options["slots"] and self._class_defines_name(class_node, "__slots__"):
            raise SyntaxError(
                "dataclass option 'slots=True' cannot be used when '__slots__' is already defined"
            )
        if options["unsafe_hash"] and self._class_defines_name(class_node, "__hash__"):
            raise SyntaxError(
                "dataclass option 'unsafe_hash=True' cannot be used when "
                "'__hash__' is explicitly defined"
            )
        return options

    _FIELD_ALLOWED_OPTS = frozenset(
        {"default", "default_factory", "init", "repr", "compare", "hash", "kw_only"}
    )
    _FIELD_BOOL_OPTS = frozenset({"init", "repr", "compare", "kw_only"})

    def _is_field_call(self, call_node):
        """Return True if ``call_node`` invokes ``dataclasses.field``."""
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id in self._dataclass_field_names
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "field"
            and isinstance(func.value, ast.Name)
            and func.value.id in self.dataclasses_module_names
        )

    def _parse_field_hash_value(self, value):
        if isinstance(value, ast.Constant) and value.value in (True, False, None):
            return value.value
        raise SyntaxError(
            "dataclass field option 'hash' must be True, False or None"
        )

    def _apply_field_keyword(self, kw, state):
        """Mutate ``state`` with the value carried by ``kw``.

        ``state`` is the dict ``{"default_expr": ..., "factory_expr": ...,
        "options": {...}}`` populated by ``_parse_field_call``.
        """
        if kw.arg == "default":
            state["default_expr"] = kw.value
        elif kw.arg == "default_factory":
            state["factory_expr"] = kw.value
        elif kw.arg in self._FIELD_BOOL_OPTS:
            state["options"][kw.arg] = self._parse_dataclass_bool_option(
                kw.value, kw.arg
            )
        else:  # kw.arg == "hash"
            state["options"]["hash"] = self._parse_field_hash_value(kw.value)

    def _parse_field_call(self, default_value):
        """Decompose ``field(...)`` calls into (default_expr, factory_expr, options).

        Returns a triple where ``default_expr`` is the AST node for the
        ``default=`` value (or the original raw value when ``default_value`` is
        not a ``field(...)`` call), and ``factory_expr`` is the AST node passed
        as ``default_factory=`` (or ``None`` when not present).

        The model in ``models/dataclasses.py`` already collapses
        ``field(default=X)`` to ``X`` and ``field(default_factory=F)`` to
        ``F()`` at runtime, but operating at AST level lets us:
          * keep ``default=`` literals as ordinary Python defaults
            (cheap and consistent with Marco B), and
          * desugar ``default_factory=`` into a per-instance call so each
            constructed object gets a fresh value.
        """
        if not isinstance(default_value, ast.Call) or not self._is_field_call(
            default_value
        ):
            return default_value, None, {}

        if default_value.args:
            raise SyntaxError("field(...) does not accept positional arguments")

        state = {"default_expr": None, "factory_expr": None, "options": {}}
        seen = set()
        for kw in default_value.keywords:
            if kw.arg is None:
                raise SyntaxError("field(...) does not support **kwargs")
            if kw.arg not in self._FIELD_ALLOWED_OPTS:
                raise SyntaxError(f"unsupported dataclass field option {kw.arg!r}")
            if kw.arg in seen:
                raise SyntaxError(f"duplicate dataclass field option {kw.arg!r}")
            seen.add(kw.arg)
            self._apply_field_keyword(kw, state)

        if state["default_expr"] is not None and state["factory_expr"] is not None:
            raise SyntaxError(
                "field(...) cannot specify both default and default_factory"
            )
        # ``field()`` with no default and no factory is a required field.
        return state["default_expr"], state["factory_expr"], state["options"]

    def collect_fields(self, class_node):
        """Collect dataclass field specs.

        Each entry is ``(name, annotation, default_expr, factory_expr, kind)`` where
        ``kind`` is one of ``instance``, ``initvar`` or ``classvar``.
        """
        dataclass_options = self._parse_dataclass_options(class_node)
        fields = []
        inherited_fields = []
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in self._dataclass_class_specs:
                inherited_fields.extend(
                    copy.deepcopy(self._dataclass_class_specs[base.id]["fields"])
                )
        by_name = {f["name"]: idx for idx, f in enumerate(inherited_fields)}
        for stmt in class_node.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if not isinstance(stmt.target, ast.Name):
                continue
            field_kind, annotation = self._analyze_dataclass_field_annotation(
                stmt.annotation
            )
            default_expr, factory_expr, per_field_options = self._parse_field_call(
                stmt.value
            )
            per_field_options = dict(per_field_options)
            per_field_options["kind"] = field_kind
            field = self._field_spec(
                stmt.target.id,
                annotation,
                (default_expr, factory_expr),
                per_field_options,
            )
            if dataclass_options["kw_only"] and field["kind"] in (
                "instance",
                "initvar",
            ):
                field["kw_only"] = True
            if field["kind"] in ("classvar", "initvar"):
                field["init"] = field["kind"] == "initvar"
                field["repr"] = False
                field["compare"] = False
                field["hash"] = False
            if field["name"] in by_name:
                inherited_fields[by_name[field["name"]]] = field
            else:
                by_name[field["name"]] = len(inherited_fields)
                inherited_fields.append(field)
        fields.extend(inherited_fields)
        return fields, dataclass_options

    @staticmethod
    def _partition_init_fields(fields):
        """Split fields into (param_fields, pos_fields, kwonly_fields, initvar_names)."""
        param_fields = [
            field for field in fields if field["kind"] != "classvar" and field["init"]
        ]
        initvar_names = [
            field["name"] for field in fields if field["kind"] == "initvar"
        ]
        pos_fields = [field for field in param_fields if not field["kw_only"]]
        kwonly_fields = [field for field in param_fields if field["kw_only"]]
        return param_fields, pos_fields, kwonly_fields, initvar_names

    @staticmethod
    def _build_init_positional_defaults(pos_fields):
        """Return positional ``defaults`` list mirroring trailing pos_fields."""
        first_default_idx = None
        for index, field in enumerate(pos_fields):
            if field["default_expr"] is not None or field["factory_expr"] is not None:
                first_default_idx = index
                break
        if first_default_idx is None:
            return []
        defaults = []
        for field in pos_fields[first_default_idx:]:
            default_expr = field["default_expr"]
            if default_expr is not None:
                if not isinstance(default_expr, (ast.Constant, ast.Name)):
                    raise SyntaxError(
                        "unsupported dataclass default expression: "
                        "synthesized __init__ defaults must be a Constant "
                        "or a simple Name"
                    )
                defaults.append(copy.deepcopy(default_expr))
            else:
                defaults.append(ast.Constant(value=None))
        return defaults

    def _build_init_args(self, class_node, fields):
        """Return (args, defaults, kwonlyargs, kw_defaults, initvar_names)."""
        param_fields, pos_fields, kwonly_fields, initvar_names = (
            self._partition_init_fields(fields)
        )
        args = [ast.arg(arg="self", annotation=None)]
        defaults = self._build_init_positional_defaults(pos_fields)
        kwonlyargs = []
        kw_defaults = []
        for field in param_fields:
            arg = ast.arg(
                arg=field["name"], annotation=copy.deepcopy(field["annotation"])
            )
            if field["kw_only"]:
                kwonlyargs.append(arg)
                if field["default_expr"] is not None:
                    kw_defaults.append(copy.deepcopy(field["default_expr"]))
                elif field["factory_expr"] is not None:
                    kw_defaults.append(ast.Constant(value=None))
                else:
                    kw_defaults.append(None)
            else:
                args.append(arg)
        # Register synthesized dataclass __init__ keyword-only parameters
        # before generic visiting so call-site validation can consume them.
        self.functionKwonlyParams[f"{class_node.name}.__init__"] = [
            field["name"] for field in kwonly_fields
        ]
        return args, defaults, kwonlyargs, kw_defaults, initvar_names

    def _build_init_field_rhs(self, class_node, field):
        """Return the RHS expression for an instance field assignment."""
        field_name = field["name"]
        if field["factory_expr"] is not None:
            # Factory fields are always assigned via the factory call;
            # they are not exposed as __init__ parameters.
            return ast.Call(
                func=copy.deepcopy(field["factory_expr"]), args=[], keywords=[]
            )
        if field["init"]:
            return self.create_name_node(field_name, ast.Load(), class_node)
        rhs = copy.deepcopy(field["default_expr"])
        if rhs is None:
            raise SyntaxError(
                f"field(init=False) for instance field {field_name!r} "
                "requires a default or default_factory"
            )
        return rhs

    def _build_init_field_assigns(self, class_node, fields):
        """Build ``self.<field> = ...`` assignments for instance fields.

        Emits plain ``Assign`` statements rather than ``AnnAssign`` for
        ``self.<field>``. Optional-typed instance attributes are already
        handled by the normal class/parameter typing paths, whereas an
        explicit ``self.x: Optional[T] = ...`` here can confuse later
        arithmetic over values proven non-None by guards (see optional7).
        """
        body = []
        for field in fields:
            if field["kind"] != "instance":
                continue
            assign_stmt = ast.Assign(
                targets=[
                    ast.Attribute(
                        value=self.create_name_node("self", ast.Load(), class_node),
                        attr=field["name"],
                        ctx=ast.Store(),
                    )
                ],
                value=self._build_init_field_rhs(class_node, field),
            )
            self.ensure_all_locations(assign_stmt, class_node)
            body.append(assign_stmt)
        return body

    def _build_post_init_call_stmt(self, class_node, initvar_names):
        """Return an ``Expr`` calling ``__post_init__`` or ``None``.

        When the class defines ``__post_init__`` directly, the call is
        addressed as ``ClassName.__post_init__(self, *initvars)`` so the
        unbound-method form is preserved. Otherwise, if the class merely
        inherits a ``__post_init__`` (detected by behavior), the call uses
        the bound form ``self.__post_init__(*initvars)``.
        """
        post_init_method = self._get_post_init_method(class_node)
        if post_init_method is not None:
            call = ast.Call(
                func=ast.Attribute(
                    value=self.create_name_node(
                        class_node.name, ast.Load(), class_node
                    ),
                    attr="__post_init__",
                    ctx=ast.Load(),
                ),
                args=[self.create_name_node("self", ast.Load(), class_node)]
                + [
                    self.create_name_node(name, ast.Load(), class_node)
                    for name in initvar_names
                ],
                keywords=[],
            )
        elif self._class_has_post_init_behavior(class_node):
            call = ast.Call(
                func=ast.Attribute(
                    value=self.create_name_node("self", ast.Load(), class_node),
                    attr="__post_init__",
                    ctx=ast.Load(),
                ),
                args=[
                    self.create_name_node(name, ast.Load(), class_node)
                    for name in initvar_names
                ],
                keywords=[],
            )
        else:
            return None
        stmt = ast.Expr(value=call)
        self.ensure_all_locations(stmt, class_node)
        return stmt

    def build_init(self, class_node, fields):
        """Build __init__(self, ...) that assigns self.<field> = <field>.

        Supports raw defaults (``x: int = 5``), ``field(default=...)`` and
        ``field(default_factory=...)``.

        Factory fields are exposed as ``__init__`` parameters with ``None`` as
        default so signature ordering and defaults match expectations from the
        dataclass preprocessor tests. The assignment remains a direct
        ``self.<field> = <factory>()`` in the body.
        """
        args, defaults, kwonlyargs, kw_defaults, initvar_names = (
            self._build_init_args(class_node, fields)
        )
        body = self._build_init_field_assigns(class_node, fields)
        post_init_stmt = self._build_post_init_call_stmt(class_node, initvar_names)
        if post_init_stmt is not None:
            body.append(post_init_stmt)
        if not body:
            body = [ast.Pass()]
        init_func = ast.FunctionDef(
            name="__init__",
            args=ast.arguments(
                posonlyargs=[],
                args=args,
                vararg=None,
                kwonlyargs=kwonlyargs,
                kw_defaults=kw_defaults,
                kwarg=None,
                defaults=defaults,
            ),
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        self.ensure_all_locations(init_func, class_node)
        ast.fix_missing_locations(init_func)
        return init_func

    def _build_tuple_from_fields(self, class_node, fields, predicate):
        tuple_node = ast.Tuple(
            elts=[
                ast.Attribute(
                    value=self.create_name_node("self", ast.Load(), class_node),
                    attr=field["name"],
                    ctx=ast.Load(),
                )
                for field in fields
                if field["kind"] == "instance" and predicate(field)
            ],
            ctx=ast.Load(),
        )
        self.ensure_all_locations(tuple_node, class_node)
        return tuple_node

    def build_dataclass_repr(self, class_node, fields):
        parts = [ast.Constant(value=f"{class_node.name}(")]
        repr_fields = [f for f in fields if f["kind"] == "instance" and f["repr"]]
        for idx, field in enumerate(repr_fields):
            prefix = "" if idx == 0 else ", "
            parts.append(ast.Constant(value=f"{prefix}{field['name']}="))
            parts.append(
                ast.Call(
                    func=self.create_name_node("repr", ast.Load(), class_node),
                    args=[
                        ast.Attribute(
                            value=self.create_name_node("self", ast.Load(), class_node),
                            attr=field["name"],
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[],
                )
            )
        parts.append(ast.Constant(value=")"))
        ret = ast.Return(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Constant(value=""), attr="join", ctx=ast.Load()
                ),
                args=[ast.List(elts=parts, ctx=ast.Load())],
                keywords=[],
            )
        )
        fn = ast.FunctionDef(
            name="__repr__",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[ret],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        self.ensure_all_locations(fn, class_node)
        ast.fix_missing_locations(fn)
        return fn

    def build_dataclass_eq(self, class_node, fields):
        compare_fields = [f for f in fields if f["kind"] == "instance" and f["compare"]]
        self_tuple = ast.Tuple(
            elts=[
                ast.Attribute(
                    value=self.create_name_node("self", ast.Load(), class_node),
                    attr=f["name"],
                    ctx=ast.Load(),
                )
                for f in compare_fields
            ],
            ctx=ast.Load(),
        )
        other_tuple = ast.Tuple(
            elts=[
                ast.Attribute(
                    value=self.create_name_node("other", ast.Load(), class_node),
                    attr=f["name"],
                    ctx=ast.Load(),
                )
                for f in compare_fields
            ],
            ctx=ast.Load(),
        )
        body = [
            ast.If(
                test=ast.Call(
                    func=self.create_name_node("isinstance", ast.Load(), class_node),
                    args=[
                        self.create_name_node("other", ast.Load(), class_node),
                        self.create_name_node(class_node.name, ast.Load(), class_node),
                    ],
                    keywords=[],
                ),
                body=[
                    ast.Return(
                        value=ast.Compare(
                            left=self_tuple, ops=[ast.Eq()], comparators=[other_tuple]
                        )
                    )
                ],
                orelse=[ast.Return(value=ast.Constant(value=False))],
            )
        ]
        fn = ast.FunctionDef(
            name="__eq__",
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg="self", annotation=None),
                    ast.arg(arg="other", annotation=None),
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        self.ensure_all_locations(fn, class_node)
        ast.fix_missing_locations(fn)
        return fn

    def build_dataclass_hash(self, class_node, fields):
        hash_fields = []
        for field in fields:
            if field["kind"] != "instance":
                continue
            if field["hash"] is False:
                continue
            if field["hash"] is None and not field["compare"]:
                continue
            hash_fields.append(field)
        tup = ast.Tuple(
            elts=[
                ast.Attribute(
                    value=self.create_name_node("self", ast.Load(), class_node),
                    attr=f["name"],
                    ctx=ast.Load(),
                )
                for f in hash_fields
            ],
            ctx=ast.Load(),
        )
        fn = ast.FunctionDef(
            name="__hash__",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[
                ast.Return(
                    value=ast.Call(
                        func=self.create_name_node("hash", ast.Load(), class_node),
                        args=[tup],
                        keywords=[],
                    )
                )
            ],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        self.ensure_all_locations(fn, class_node)
        ast.fix_missing_locations(fn)
        return fn

    def build_dataclass_order(self, class_node, fields, method_name, op_cls):
        compare_fields = [f for f in fields if f["kind"] == "instance" and f["compare"]]
        self_tuple = ast.Tuple(
            elts=[
                ast.Attribute(
                    value=self.create_name_node("self", ast.Load(), class_node),
                    attr=f["name"],
                    ctx=ast.Load(),
                )
                for f in compare_fields
            ],
            ctx=ast.Load(),
        )
        other_tuple = ast.Tuple(
            elts=[
                ast.Attribute(
                    value=self.create_name_node("other", ast.Load(), class_node),
                    attr=f["name"],
                    ctx=ast.Load(),
                )
                for f in compare_fields
            ],
            ctx=ast.Load(),
        )
        body = [
            ast.If(
                test=ast.Call(
                    func=self.create_name_node("isinstance", ast.Load(), class_node),
                    args=[
                        self.create_name_node("other", ast.Load(), class_node),
                        self.create_name_node(class_node.name, ast.Load(), class_node),
                    ],
                    keywords=[],
                ),
                body=[
                    ast.Return(
                        value=ast.Compare(
                            left=self_tuple, ops=[op_cls()], comparators=[other_tuple]
                        )
                    )
                ],
                orelse=[ast.Return(value=ast.Constant(value=False))],
            )
        ]
        fn = ast.FunctionDef(
            name=method_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg="self", annotation=None),
                    ast.arg(arg="other", annotation=None),
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        self.ensure_all_locations(fn, class_node)
        ast.fix_missing_locations(fn)
        return fn

    def build_dataclass_fields_metadata(self, class_node, fields):
        field_names = [field["name"] for field in fields if field["kind"] == "instance"]
        metadata_assign = ast.Assign(
            targets=[ast.Name(id="__dataclass_fields__", ctx=ast.Store())],
            value=ast.Tuple(
                elts=[ast.Constant(value=field_name) for field_name in field_names],
                ctx=ast.Load(),
            ),
        )
        self.ensure_all_locations(metadata_assign, class_node)
        ast.fix_missing_locations(metadata_assign)
        return metadata_assign

    @staticmethod
    def _validate_dataclass_field_ordering(class_node, fields):
        """Raise SyntaxError if a non-default field follows a defaulted one.

        Mirrors CPython's ``non-default argument follows default argument``
        diagnostic. Class variables and keyword-only fields are exempt.
        """
        seen_default = False
        for field in fields:
            if field["kind"] == "classvar" or field["kw_only"]:
                continue
            has_default = (
                field["default_expr"] is not None or field["factory_expr"] is not None
            )
            if seen_default and not has_default:
                raise SyntaxError(
                    f"non-default argument {field['name']!r} follows default "
                    f"argument in dataclass {class_node.name!r} "
                    f"(line {class_node.lineno})"
                )
            if has_default:
                seen_default = True

    def _record_dataclass_attr_annotations(self, class_node, fields):
        """Preserve instance-field annotations for later attribute lookups."""
        if class_node.name not in self.class_attr_annotations:
            self.class_attr_annotations[class_node.name] = {}
        for field in fields:
            if field["kind"] != "instance":
                continue
            annotation = field["annotation"]
            if annotation is not None:
                self.class_attr_annotations[class_node.name][field["name"]] = annotation

    @staticmethod
    def _strip_field_annassigns(class_node, fields):
        """Remove ``x: T [= ...]`` declarations now subsumed by ``__init__``."""
        field_names = {
            field["name"]
            for field in fields
            if field["kind"] in ("instance", "initvar")
        }
        class_node.body = [
            stmt
            for stmt in class_node.body
            if not (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id in field_names
            )
        ]

    @staticmethod
    def _compute_init_insert_index(class_node):
        """Return 1 if the class body opens with a docstring, else 0."""
        if not class_node.body:
            return 0
        first_stmt = class_node.body[0]
        if isinstance(first_stmt, ast.Expr) and (
            (
                isinstance(first_stmt.value, ast.Constant)
                and isinstance(first_stmt.value.value, str)
            )
            or isinstance(first_stmt.value, ast.Str)
        ):
            return 1
        return 0

    @staticmethod
    def _class_defines_method(class_node, method_name):
        return any(
            isinstance(member, ast.FunctionDef) and member.name == method_name
            for member in class_node.body
        )

    def _inject_dataclass_synth_methods(
        self, class_node, fields, dataclass_options, insert_index
    ):
        """Insert __init__/fields metadata/__repr__/__eq__/order/__hash__."""
        if dataclass_options["init"]:
            class_node.body.insert(
                insert_index, self.build_init(class_node, fields)
            )
        class_node.body.insert(
            insert_index + 1, self.build_dataclass_fields_metadata(class_node, fields)
        )
        if dataclass_options["repr"] and not self._class_defines_method(
            class_node, "__repr__"
        ):
            class_node.body.insert(
                insert_index + 2, self.build_dataclass_repr(class_node, fields)
            )
        if dataclass_options["eq"] and not self._class_defines_method(
            class_node, "__eq__"
        ):
            class_node.body.insert(
                insert_index + 2, self.build_dataclass_eq(class_node, fields)
            )
        if dataclass_options["order"]:
            existing = {
                m.name for m in class_node.body if isinstance(m, ast.FunctionDef)
            }
            for method_name, op in (
                ("__lt__", ast.Lt),
                ("__le__", ast.LtE),
                ("__gt__", ast.Gt),
                ("__ge__", ast.GtE),
            ):
                if method_name not in existing:
                    class_node.body.insert(
                        insert_index + 2,
                        self.build_dataclass_order(
                            class_node, fields, method_name, op
                        ),
                    )
        should_generate_hash = dataclass_options["unsafe_hash"] or (
            dataclass_options["eq"] and dataclass_options["frozen"]
        )
        if should_generate_hash and not self._class_defines_method(
            class_node, "__hash__"
        ):
            class_node.body.insert(
                insert_index + 2, self.build_dataclass_hash(class_node, fields)
            )

    def expand_dataclass(self, class_node):
        """Inject a minimal generated __init__ for dataclass-decorated classes."""
        if not self.is_dataclass(class_node):
            return class_node
        if self._class_defines_method(class_node, "__init__"):
            return class_node

        fields, dataclass_options = self.collect_fields(class_node)
        if any(field["kind"] == "initvar" for field in fields):
            self._needs_dataclass_initvar_import = True

        active_fields = [
            field for field in fields if field["kind"] in ("instance", "initvar")
        ]
        has_post_init_behavior = self._class_has_post_init_behavior(class_node)
        has_classvars = any(f["kind"] == "classvar" for f in fields)
        if not active_fields and not has_post_init_behavior and not has_classvars:
            return class_node

        post_init_method = self._get_post_init_method(class_node)
        self._validate_post_init_signature(class_node, fields, post_init_method)
        if has_post_init_behavior:
            self._classes_with_post_init.add(class_node.name)

        self._validate_dataclass_field_ordering(class_node, fields)
        self._record_dataclass_attr_annotations(class_node, fields)
        self._strip_field_annassigns(class_node, fields)

        insert_index = self._compute_init_insert_index(class_node)
        self._inject_dataclass_synth_methods(
            class_node, fields, dataclass_options, insert_index
        )

        self._dataclass_class_specs[class_node.name] = {
            "fields": copy.deepcopy(fields),
            "options": copy.deepcopy(dataclass_options),
        }
        return class_node

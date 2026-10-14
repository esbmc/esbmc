import ast
# pylint: disable=too-many-branches


class ImportTypingMixin:
    _TYPING_GENERIC_NAMES = (
        "Tuple",
        "List",
        "Dict",
        "Set",
        "Optional",
        "Union",
        "Callable",
        "ClassVar",
    )

    def _is_type_alias_expression(self, value):
        """Check whether value is a typing alias RHS like Tuple[int, int]."""
        if not isinstance(value, ast.Subscript):
            return False

        base = value.value
        if isinstance(base, ast.Name):
            if base.id not in self._typing_imported_names:
                return False
        elif isinstance(base, ast.Attribute):
            if base.attr not in self._TYPING_GENERIC_NAMES:
                return False
            mod = base.value
            if not (isinstance(mod, ast.Name) and mod.id in self.typing_module_names):
                return False
        else:
            return False

        idx = value.slice
        if isinstance(idx, ast.Index):
            idx = idx.value
        if isinstance(idx, ast.Constant) and isinstance(idx.value, int):
            return False
        if (isinstance(idx, ast.UnaryOp) and isinstance(idx.op, (ast.UAdd, ast.USub))
                and isinstance(idx.operand, ast.Constant) and isinstance(idx.operand.value, int)):
            return False
        return True

    def _copy_annotation_node(self, node):
        """Deep copy an annotation AST node."""
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return ast.Name(id=node.id, ctx=ast.Load())
        if isinstance(node, ast.Subscript):
            return ast.Subscript(
                value=self._copy_annotation_node(node.value),
                slice=self._copy_annotation_node(node.slice),
                ctx=ast.Load(),
            )
        if isinstance(node, ast.Index):
            return ast.Index(value=self._copy_annotation_node(node.value))
        if isinstance(node, ast.Tuple):
            return ast.Tuple(elts=[self._copy_annotation_node(e) for e in node.elts],
                             ctx=ast.Load())
        if isinstance(node, ast.Constant):
            return ast.Constant(value=node.value)
        if isinstance(node, ast.Str):
            return ast.Constant(value=node.s)
        if isinstance(node, ast.Attribute):
            return ast.Attribute(
                value=self._copy_annotation_node(node.value),
                attr=node.attr,
                ctx=ast.Load(),
            )
        return node

    def _resolve_annotation_aliases(self, annotation):
        """Recursively resolve type aliases in an annotation AST node."""
        if annotation is None:
            return None

        if isinstance(annotation, ast.Name):
            if annotation.id in self.type_aliases:
                copied = self._copy_annotation_node(self.type_aliases[annotation.id])
                return self._resolve_annotation_aliases(copied)
            return annotation

        if isinstance(annotation, ast.Subscript):
            resolved_value = self._resolve_annotation_aliases(annotation.value)
            resolved_slice = annotation.slice
            if isinstance(annotation.slice, ast.Index):
                resolved_slice = ast.Index(
                    value=self._resolve_annotation_aliases(annotation.slice.value))
            elif not isinstance(annotation.slice, (ast.Slice, ast.ExtSlice)):
                resolved_slice = self._resolve_annotation_aliases(annotation.slice)
            return ast.Subscript(value=resolved_value, slice=resolved_slice, ctx=ast.Load())

        if isinstance(annotation, ast.Tuple):
            resolved_elts = [self._resolve_annotation_aliases(e) for e in annotation.elts]
            return ast.Tuple(elts=resolved_elts, ctx=ast.Load())

        if isinstance(annotation, ast.Attribute):
            resolved_value = self._resolve_annotation_aliases(annotation.value)
            return ast.Attribute(value=resolved_value, attr=annotation.attr, ctx=ast.Load())

        return annotation

    def _handle_decimal_importfrom(self, alias):
        if alias.name in ("Decimal", "*"):
            self.decimal_imported = True
            if alias.asname:
                self.decimal_class_alias = alias.asname

    def _handle_collections_importfrom(self, alias):
        if alias.name in ("defaultdict", "*"):
            self.defaultdict_imported = True
            if alias.asname:
                self.defaultdict_alias = alias.asname

    def _handle_dataclasses_importfrom(self, alias):
        if alias.name == "dataclass":
            self._dataclass_decorator_names.add(alias.asname or "dataclass")
        elif alias.name == "field":
            self._dataclass_field_names.add(alias.asname or "field")
        elif alias.name == "InitVar":
            self._dataclass_initvar_names.add(alias.asname or "InitVar")
        elif alias.name == "is_dataclass":
            self._dataclass_is_dataclass_names.add(alias.asname or "is_dataclass")
        elif alias.name == "fields":
            self._dataclass_fields_api_names.add(alias.asname or "fields")
        elif alias.name == "asdict":
            self._dataclass_asdict_names.add(alias.asname or "asdict")
        elif alias.name == "astuple":
            self._dataclass_astuple_names.add(alias.asname or "astuple")
        elif alias.name == "replace":
            self._dataclass_replace_names.add(alias.asname or "replace")
        elif alias.name == "*":
            self._dataclass_decorator_names.add("dataclass")
            self._dataclass_field_names.add("field")
            self._dataclass_initvar_names.add("InitVar")
            self._dataclass_is_dataclass_names.add("is_dataclass")
            self._dataclass_fields_api_names.add("fields")
            self._dataclass_asdict_names.add("asdict")
            self._dataclass_astuple_names.add("astuple")
            self._dataclass_replace_names.add("replace")

    def _handle_typing_importfrom(self, alias):
        if alias.name == "NewType":
            self.newtype_names.add(alias.asname or "NewType")
        elif alias.name == "*":
            self.newtype_names.add("NewType")
            self._typing_imported_names.update(self._TYPING_GENERIC_NAMES)
            self._typing_classvar_names.add("ClassVar")
        if alias.name in self._TYPING_GENERIC_NAMES:
            self._typing_imported_names.add(alias.asname or alias.name)
        if alias.name == "ClassVar":
            self._typing_classvar_names.add(alias.asname or alias.name)

    def visit_ImportFrom(self, node):
        if node.module == "decimal":
            for alias in node.names:
                self._handle_decimal_importfrom(alias)
        elif node.module == "collections":
            for alias in node.names:
                self._handle_collections_importfrom(alias)
        elif node.module == "dataclasses":
            for alias in node.names:
                self._handle_dataclasses_importfrom(alias)
        elif node.module == "typing":
            for alias in node.names:
                self._handle_typing_importfrom(alias)
        self.generic_visit(node)
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "decimal":
                self.decimal_module_imported = True
                if alias.asname:
                    self.decimal_module_alias = alias.asname
            if alias.name == "collections":
                self.collections_module_imported = True
                if alias.asname:
                    self.collections_module_alias = alias.asname
            if alias.name == "typing":
                self.typing_module_names.add(alias.asname or "typing")
            if alias.name == "dataclasses":
                self.dataclasses_module_names.add(alias.asname or "dataclasses")
        self.generic_visit(node)
        return node

    def _infer_type_from_subscript(self, subscript_node):
        """Infer type from subscript operations like d[\"key\"] or lst[0]."""
        if not isinstance(subscript_node.value, ast.Name):
            return "Any"

        base_var = subscript_node.value.id

        if not hasattr(self, "variable_annotations") or base_var not in self.variable_annotations:
            return "Any"

        annotation = self.variable_annotations[base_var]

        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name) and annotation.value.id == "dict":
                if isinstance(annotation.slice, ast.Tuple) and len(annotation.slice.elts) == 2:
                    value_type_annotation = annotation.slice.elts[1]
                    return self._extract_full_type_string(value_type_annotation)
            elif isinstance(annotation.value,
                            ast.Name) and annotation.value.id in ["list", "tuple"]:
                return self._extract_full_type_string(annotation.slice)

        return "Any"

    def _extract_full_type_string(self, type_node):
        """Extract type string from an annotation node."""
        if isinstance(type_node, ast.Name):
            return type_node.id
        if isinstance(type_node, ast.Subscript):
            base_type = type_node.value.id if isinstance(type_node.value, ast.Name) else "Any"
            return base_type
        return "Any"

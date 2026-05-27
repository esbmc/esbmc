import ast
# pylint: disable=too-many-nested-blocks,too-many-branches


class TypeInferenceMixin:

    def _extract_dict_method_element_type(self, iterable_node):
        if not (isinstance(iterable_node, ast.Call)
                and isinstance(iterable_node.func, ast.Attribute)):
            return None
        method_name = iterable_node.func.attr
        if method_name not in ("keys", "values"):
            return None
        if not isinstance(iterable_node.func.value, ast.Name):
            return None
        dict_var_name = iterable_node.func.value.id
        if not (hasattr(self, "variable_annotations")
                and dict_var_name in self.variable_annotations):
            return None
        dict_annotation = self.variable_annotations[dict_var_name]
        if not (isinstance(dict_annotation, ast.Subscript) and isinstance(
                dict_annotation.slice, ast.Tuple) and len(dict_annotation.slice.elts) >= 2):
            return None
        key_type, value_type = dict_annotation.slice.elts[0], dict_annotation.slice.elts[1]
        candidate = key_type if method_name == "keys" else value_type
        if isinstance(candidate, ast.Name):
            return candidate.id
        if isinstance(candidate, ast.Subscript) and isinstance(candidate.value, ast.Name):
            return candidate.value.id
        return None

    def _extract_dict_name_element_type(self, iterable_node):
        if not isinstance(iterable_node, ast.Name):
            return None
        var_name = iterable_node.id
        if not (hasattr(self, "variable_annotations") and var_name in self.variable_annotations):
            return None
        annotation = self.variable_annotations[var_name]
        if not (isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name)
                and annotation.value.id == "dict" and isinstance(annotation.slice, ast.Tuple)
                and len(annotation.slice.elts) >= 1):
            return None
        key_type = annotation.slice.elts[0]
        if isinstance(key_type, ast.Name):
            return key_type.id
        return None

    def _extract_list_tuple_annotation_element_type(self, iterable_node):
        if not (isinstance(iterable_node, ast.Name) and hasattr(self, "variable_annotations")):
            return None
        annotation = self.variable_annotations.get(iterable_node.id)
        if not isinstance(annotation, ast.Subscript):
            return None
        element_annotation = annotation.slice
        if isinstance(element_annotation, ast.Name):
            return element_annotation.id
        if isinstance(element_annotation, ast.Subscript) and isinstance(
                element_annotation.value, ast.Name):
            return element_annotation.value.id
        return None

    def _extract_type_from_annotation(self, annotation):
        if annotation is None:
            return "Any"

        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            return annotation.value.split("[")[0]

        return "Any"

    def _get_iterable_type_annotation(self, iterable):
        if isinstance(iterable, ast.Constant) and isinstance(iterable.value, str):
            return "str"
        if isinstance(iterable, ast.List):
            return "list"
        if isinstance(iterable, ast.Tuple):
            return "tuple"
        if isinstance(iterable, ast.Name):
            known_type = self.known_variable_types.get(iterable.id)
            if known_type and known_type != "Any":
                return known_type
            return "list"
        # `for c in str(x)` / `for c in str(abs(x))` etc. The iterable is the
        # str(...) call whose return type is `str`; without this branch we
        # fall through to "list" and lower the loop as a list iteration,
        # which trips an IndexError because the str length is shorter than
        # the list-style get_object_size bound.
        if (isinstance(iterable, ast.Call)
                and isinstance(iterable.func, ast.Name)
                and iterable.func.id == "str"):
            return "str"
        return "list"

    def _get_element_type_from_container(self, container_type, iterable_node=None):  # pylint: disable=too-many-branches,too-many-nested-blocks
        dict_method_type = self._extract_dict_method_element_type(iterable_node)
        if dict_method_type is not None:
            return dict_method_type

        dict_name_type = self._extract_dict_name_element_type(iterable_node)
        if dict_name_type is not None:
            return dict_name_type

        if container_type == "str":
            return "str"
        if isinstance(iterable_node, ast.List) and iterable_node.elts:
            first_elem = iterable_node.elts[0]
            if isinstance(first_elem, ast.Constant):
                return type(first_elem.value).__name__
        if container_type.lower() in ["list", "tuple"]:
            annotation_element_type = self._extract_list_tuple_annotation_element_type(
                iterable_node)
            if annotation_element_type is not None:
                return annotation_element_type
            return "Any"
        return "Any"

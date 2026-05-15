import ast
# pylint: disable=too-many-nested-blocks,too-many-branches


class TypeInferenceMixin:

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
        return "list"

    def _get_element_type_from_container(
            self, container_type, iterable_node=None):  # pylint: disable=too-many-branches,too-many-nested-blocks
        if isinstance(iterable_node, ast.Call) and isinstance(iterable_node.func, ast.Attribute):
            method_name = iterable_node.func.attr

            if method_name in ["keys", "values"]:
                if isinstance(iterable_node.func.value, ast.Name):
                    dict_var_name = iterable_node.func.value.id
                    if (hasattr(self, "variable_annotations")
                            and dict_var_name in self.variable_annotations):
                        dict_annotation = self.variable_annotations[dict_var_name]
                        if isinstance(dict_annotation, ast.Subscript):
                            if isinstance(dict_annotation.slice, ast.Tuple):
                                key_type = dict_annotation.slice.elts[0]
                                value_type = dict_annotation.slice.elts[1]

                                if method_name == "keys":
                                    if isinstance(key_type, ast.Name):
                                        return key_type.id
                                    if isinstance(key_type, ast.Subscript) and isinstance(
                                            key_type.value, ast.Name):
                                        return key_type.value.id
                                elif method_name == "values":
                                    if isinstance(value_type, ast.Name):
                                        return value_type.id
                                    if isinstance(value_type, ast.Subscript) and isinstance(
                                            value_type.value, ast.Name):
                                        return value_type.value.id

        if isinstance(iterable_node, ast.Name):
            var_name = iterable_node.id

            if (hasattr(self, "variable_annotations") and var_name in self.variable_annotations):
                annotation = self.variable_annotations[var_name]
                if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
                    if annotation.value.id == "dict":
                        if (isinstance(annotation.slice, ast.Tuple)
                                and len(annotation.slice.elts) >= 1):
                            key_type = annotation.slice.elts[0]
                            if isinstance(key_type, ast.Name):
                                return key_type.id

        if container_type == "str":
            return "str"
        if isinstance(iterable_node, ast.List) and iterable_node.elts:
            first_elem = iterable_node.elts[0]
            if isinstance(first_elem, ast.Constant):
                return type(first_elem.value).__name__
        if container_type.lower() in ["list", "tuple"]:
            if isinstance(iterable_node, ast.Name) and hasattr(self, "variable_annotations"):
                var_name = iterable_node.id
                if var_name in self.variable_annotations:
                    annotation = self.variable_annotations[var_name]
                    if isinstance(annotation, ast.Subscript):
                        element_annotation = annotation.slice
                        if isinstance(element_annotation, ast.Name):
                            return element_annotation.id
                        if isinstance(element_annotation, ast.Subscript):
                            if isinstance(element_annotation.value, ast.Name):
                                return element_annotation.value.id
            return "Any"
        return "Any"

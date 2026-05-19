import ast
import copy


class AssignmentVisitorsMixin:

    def visit_AnnAssign(self, node):
        """Track type annotations and lower annotated assignments."""
        if node.annotation is not None:
            node.annotation = self._resolve_annotation_aliases(node.annotation)

        node = self.generic_visit(node)

        if getattr(node, "value", None) is not None:
            prefix, lowered_value, lowered_type = self._lower_listcomp_in_expr(node.value)
            node.value = lowered_value
            if prefix:
                if not isinstance(node.target, ast.Name):
                    raise NotImplementedError(
                        "Annotated list comprehension assignment requires a simple target name")
                lowered_assign = ast.AnnAssign(
                    target=node.target,
                    annotation=node.annotation,
                    value=node.value,
                    simple=node.simple,
                )
                self._copy_location_info(node, lowered_assign)
                self.ensure_all_locations(lowered_assign, node)
                ast.fix_missing_locations(lowered_assign)
                self.known_variable_types[node.target.id] = lowered_type
                self.variable_annotations[node.target.id] = node.annotation
                return prefix + [lowered_assign]

        if isinstance(node.target, ast.Name) and node.annotation is not None:
            var_name = node.target.id
            var_type = self._extract_type_from_annotation(node.annotation)
            self.known_variable_types[var_name] = var_type
            self.variable_annotations[var_name] = node.annotation
            if isinstance(node.value, ast.List):
                self.list_literal_values[var_name] = copy.deepcopy(node.value)
            else:
                self.list_literal_values.pop(var_name, None)

            if (node.value is not None and isinstance(node.value, ast.Call)
                    and self._is_defaultdict_call(node.value)):
                factory = self._get_defaultdict_factory(node.value)
                if factory is not None:
                    self._defaultdict_factory[var_name] = factory
                empty_dict = ast.Dict(keys=[], values=[])
                ast.copy_location(empty_dict, node.value)
                ast.fix_missing_locations(empty_dict)
                node.value = empty_dict

        if (node.value is not None and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in self._defaultdict_factory):
            dict_name = node.value.value.id
            key_node = node.value.slice
            factory = self._defaultdict_factory[dict_name]
            init_stmts, key_expr = self._make_defaultdict_missing_check(
                dict_name, key_node, factory, node)
            node.value.slice = key_expr
            return init_stmts + [node]

        return node

    def visit_AugAssign(self, node):
        """Lower augmented assignment into simple assignment when needed."""
        if (isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Name)
                and node.target.value.id in self.list_literal_values):
            self.list_literal_values.pop(node.target.value.id, None)

        node = self.generic_visit(node)

        if not isinstance(node.target, ast.Subscript):
            return node

        pre_stmts = []
        if (isinstance(node.target.value, ast.Name)
                and node.target.value.id in self._defaultdict_factory):
            dict_name = node.target.value.id
            key_node = node.target.slice
            factory = self._defaultdict_factory[dict_name]
            pre_stmts, key_expr = self._make_defaultdict_missing_check(
                dict_name, key_node, factory, node)
            node.target.slice = key_expr

        load_target = self._as_load_target(node.target, node)
        binop = ast.BinOp(left=load_target, op=node.op, right=node.value)
        assign = ast.Assign(targets=[node.target], value=binop)
        self._copy_location_info(node, assign)
        self.ensure_all_locations(assign, node)
        ast.fix_missing_locations(assign)

        if pre_stmts:
            return pre_stmts + [assign]
        return assign

import ast
import copy


class ExpressionRewriteMixin:

    class _ListCompExpressionLowerer(ast.NodeTransformer):
        """Utility transformer that lowers list comprehensions, any(genexpr), and all(genexpr) inside an expression."""

        def __init__(self, preprocessor):
            super().__init__()
            self.preprocessor = preprocessor
            self.statements = []

        def visit_ListComp(self, node):
            prefix, result_expr = self.preprocessor._lower_listcomp(node)
            self.statements.extend(prefix)
            return result_expr

        def visit_SetComp(self, node):
            listcomp = ast.ListComp(elt=node.elt, generators=node.generators)
            ast.copy_location(listcomp, node)
            ast.fix_missing_locations(listcomp)
            prefix, list_name = self.preprocessor._lower_listcomp(listcomp)
            self.statements.extend(prefix)
            set_call = ast.Call(
                func=ast.Name(id="set", ctx=ast.Load()),
                args=[list_name],
                keywords=[],
            )
            ast.copy_location(set_call, node)
            ast.fix_missing_locations(set_call)
            return set_call

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "join"
                    and len(node.args) == 1 and not node.keywords
                    and isinstance(node.args[0], ast.GeneratorExp)):
                gen = node.args[0]
                elt_expr = copy.deepcopy(gen.elt)

                if (isinstance(elt_expr, ast.Call) and isinstance(elt_expr.func, ast.Name)
                        and elt_expr.func.id == "str" and len(elt_expr.args) == 1
                        and not elt_expr.keywords):
                    obj_expr = copy.deepcopy(elt_expr.args[0])
                    dunder_attr = ast.Attribute(
                        value=obj_expr,
                        attr="__str__",
                        ctx=ast.Load(),
                    )
                    elt_expr = ast.Call(func=dunder_attr, args=[], keywords=[])
                    ast.copy_location(elt_expr, gen.elt)
                    ast.fix_missing_locations(elt_expr)

                listcomp = ast.ListComp(
                    elt=elt_expr,
                    generators=copy.deepcopy(gen.generators),
                )
                ast.copy_location(listcomp, gen)
                ast.fix_missing_locations(listcomp)

                new_call = copy.deepcopy(node)
                new_call.args = [listcomp]
                ast.copy_location(new_call, node)
                ast.fix_missing_locations(new_call)

                return self.visit(new_call)

            if (isinstance(node.func, ast.Name) and node.func.id == "any" and len(node.args) == 1
                    and not node.keywords and isinstance(node.args[0], ast.GeneratorExp)):
                prefix, result = self.preprocessor._lower_any_genexp(node.args[0])
                self.statements.extend(prefix)
                return result

            if (isinstance(node.func, ast.Name) and node.func.id == "all" and len(node.args) == 1
                    and not node.keywords and isinstance(node.args[0], ast.GeneratorExp)):
                prefix, result = self.preprocessor._lower_all_genexp(node.args[0])
                self.statements.extend(prefix)
                return result

            if (isinstance(node.func, ast.Name) and node.func.id == "list" and len(node.args) == 1
                    and not node.keywords and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Name) and node.args[0].func.id == "map"
                    and len(node.args[0].args) == 2):
                map_call = node.args[0]
                func_expr = map_call.args[0]
                iterable_expr = map_call.args[1]
                if isinstance(func_expr, ast.Lambda) and len(func_expr.args.args) == 1:
                    param = func_expr.args.args[0]
                    target = ast.Name(id=param.arg, ctx=ast.Store())
                    elt = func_expr.body
                else:
                    tmp_id = f"ESBMC_map_elt_{self.preprocessor.listcomp_counter}"
                    target = ast.Name(id=tmp_id, ctx=ast.Store())
                    elt = ast.Call(
                        func=func_expr,
                        args=[ast.Name(id=tmp_id, ctx=ast.Load())],
                        keywords=[],
                    )
                listcomp = ast.ListComp(
                    elt=elt,
                    generators=[
                        ast.comprehension(target=target, iter=iterable_expr, ifs=[], is_async=0)
                    ],
                )
                ast.copy_location(listcomp, node)
                ast.fix_missing_locations(listcomp)
                return self.visit(listcomp)

            if (isinstance(node.func, ast.Name) and node.func.id == "list" and len(node.args) == 1
                    and not node.keywords and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Name)
                    and node.args[0].func.id in self.preprocessor.generator_funcs):
                prefix, result = self.preprocessor._lower_list_gen_call(node.args[0], node)
                if prefix is not None:
                    self.statements.extend(prefix)
                    return result

            lowered_sorted = self.preprocessor._lower_sorted_with_key_call(node)
            if lowered_sorted is not None:
                prefix, result = lowered_sorted
                self.statements.extend(prefix)
                return result

            lowered_min_max = self.preprocessor._lower_min_max_with_key_call(node)
            if lowered_min_max is not None:
                prefix, result = lowered_min_max
                self.statements.extend(prefix)
                return result

            lowered_tuple_sorted_pair = self.preprocessor._lower_tuple_sorted_pair_call(node)
            if lowered_tuple_sorted_pair is not None:
                prefix, result = lowered_tuple_sorted_pair
                self.statements.extend(prefix)
                return result

            return self.generic_visit(node)

    def visit_Return(self, node):
        node = self.generic_visit(node)
        prefix, new_value, _ = self._lower_listcomp_in_expr(node.value)
        node.value = new_value
        if node.value is not None:
            dd_inits, node.value = self._lower_defaultdict_reads_in_expr(node.value, node)
            prefix = dd_inits + prefix
        if prefix:
            return prefix + [node]
        return node

    def visit_Subscript(self, node):
        node = self.generic_visit(node)

        if (isinstance(node.value, ast.Name) and node.value.id in self.list_literal_values):
            list_node = self.list_literal_values[node.value.id]

            idx_node = node.slice
            if isinstance(idx_node, ast.Index):
                idx_node = idx_node.value

            idx = None
            if isinstance(idx_node, ast.Constant) and isinstance(idx_node.value, int):
                idx = idx_node.value
            elif (isinstance(idx_node, ast.UnaryOp) and isinstance(idx_node.op,
                                                                   (ast.UAdd, ast.USub))
                  and isinstance(idx_node.operand, ast.Constant)
                  and isinstance(idx_node.operand.value, int)):
                sign = -1 if isinstance(idx_node.op, ast.USub) else 1
                idx = sign * idx_node.operand.value

            if idx is not None:
                elts = list_node.elts
                if idx < 0:
                    idx = len(elts) + idx
                if 0 <= idx < len(elts):
                    elt = elts[idx]
                    is_pure_literal = isinstance(
                        elt, ast.Constant) or (isinstance(elt, ast.UnaryOp) and isinstance(
                            elt.op, (ast.UAdd, ast.USub)) and isinstance(elt.operand, ast.Constant))
                    if is_pure_literal:
                        folded = copy.deepcopy(elt)
                        self.ensure_all_locations(folded, node)
                        ast.fix_missing_locations(folded)
                        return folded

        return node

    def visit_Expr(self, node):
        rewritten = self._maybe_rewrite_list_sort_with_key(node)
        if rewritten is not None:
            return rewritten

        node = self.generic_visit(node)

        next_gen_info = self._find_generator_next_call(node.value)
        if next_gen_info is not None:
            gen_var, func_name = next_gen_info
            if func_name in self.early_return_generator_funcs:
                return self._make_stop_iteration_raise(node)
            stmts = self._inline_next_call(None, func_name, gen_var, node)
            if stmts is not None:
                return stmts

        prefix, new_value, _ = self._lower_listcomp_in_expr(node.value)
        node.value = new_value
        dd_inits, node.value = self._lower_defaultdict_reads_in_expr(node.value, node)
        prefix = dd_inits + prefix
        if prefix:
            return prefix + [node]
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        prefix, new_test, _ = self._lower_listcomp_in_expr(node.test)
        node.test = new_test
        node.test = self._transform_list_truthiness(node.test, node)
        if prefix:
            return prefix + [node]
        return node

    def _transform_list_truthiness(self, test_expr, source_node):
        if not isinstance(test_expr, ast.Name):
            return test_expr

        var_name = test_expr.id
        var_type = self.known_variable_types.get(var_name)
        if var_type != "list":
            return test_expr

        len_call = ast.Call(
            func=self.create_name_node("len", ast.Load(), source_node),
            args=[self.create_name_node(var_name, ast.Load(), source_node)],
            keywords=[],
        )
        self.ensure_all_locations(len_call, source_node)

        comparison = ast.Compare(
            left=len_call,
            ops=[ast.Gt()],
            comparators=[self.create_constant_node(0, source_node)],
        )
        self.ensure_all_locations(comparison, source_node)
        return comparison

    def visit_While(self, node):
        node = self.generic_visit(node)
        prefix, new_test, _ = self._lower_listcomp_in_expr(node.test)
        node.test = new_test
        node.test = self._transform_list_truthiness(node.test, node)
        if prefix:
            return prefix + [node]
        return node

    def _simplify_isinstance(self, node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "isinstance" and len(node.args) == 2):
            return node
        obj_node, type_node = node.args[0], node.args[1]
        if not (isinstance(obj_node, ast.Name) and isinstance(type_node, ast.Name)):
            return node
        ann = self.variable_annotations.get(obj_node.id)
        if not isinstance(ann, ast.Name) or ann.id == "Any":
            return node
        if ann.id == type_node.id:
            if obj_node.id in self._subscript_inferred_vars:
                return node
            return ast.Constant(value=True)
        return ast.Constant(value=False)

    def visit_Assert(self, node):
        node = self.generic_visit(node)
        tuple_eq_prefix = []
        if (isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq) and len(node.test.comparators) == 1):
            left = node.test.left
            right = node.test.comparators[0]
            rewritten = self._try_transform_items_set_eq(left, right, node)
            if rewritten is None:
                rewritten = self._try_transform_items_set_eq(right, left, node)
            if rewritten is None:
                rewritten = self._try_transform_list_tuple_eq(left, right, node)
            if rewritten is None:
                rewritten = self._try_transform_list_tuple_eq(right, left, node)
            if rewritten is None:
                tuple_eq_prefix, rewritten = self._try_lower_expr_tuple_literal_eq(
                    left, right, node)
            if rewritten is None:
                tuple_eq_prefix, rewritten = self._try_lower_expr_tuple_literal_eq(
                    right, left, node)
            if tuple_eq_prefix is None:
                tuple_eq_prefix = []
            if rewritten is not None:
                node.test = rewritten
        eq_prefix, maybe_eq_test = self._lower_assert_eq_literal(node.test, node)
        node.test = maybe_eq_test
        node.test = self._simplify_isinstance(node.test)
        prefix, new_test, _ = self._lower_listcomp_in_expr(node.test)
        node.test = new_test
        dd_inits, node.test = self._lower_defaultdict_reads_in_expr(node.test, node)
        prefix = tuple_eq_prefix + eq_prefix + dd_inits + prefix
        if node.msg:
            msg_prefix, new_msg, _ = self._lower_listcomp_in_expr(node.msg)
            node.msg = new_msg
            prefix.extend(msg_prefix)
        if prefix:
            return prefix + [node]
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        return node

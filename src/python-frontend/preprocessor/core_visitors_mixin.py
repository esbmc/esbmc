import ast
import copy
# pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-return-statements
# pylint: disable=too-many-nested-blocks,too-many-boolean-expressions,no-else-return,no-else-raise
# pylint: disable=import-outside-toplevel


class CoreVisitorsMixin:

    def visit_Assign(self, node):
        """
        Handle assignment nodes, including multiple assignments and tuple unpacking.
        """
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id in self.list_literal_values):
                self.list_literal_values.pop(target.value.id, None)

        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and self._is_type_alias_expression(node.value)):
            alias_name = node.targets[0].id
            self.type_aliases[alias_name] = node.value
            return None

        node = self.generic_visit(node)

        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            if self._is_assert_literal_shape(node.value):
                self._known_literal_values[target_name] = copy.deepcopy(node.value)
            elif (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                  and node.value.func.id in self._identity_functions and len(node.value.args) == 1
                  and not node.value.keywords
                  and self._is_assert_literal_shape(node.value.args[0])):
                self._known_literal_values[target_name] = copy.deepcopy(node.value.args[0])
            else:
                self._known_literal_values.pop(target_name, None)

        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("nondet_list", "nondet_dict")):
            expanded = self._expand_nondet_call(node.targets[0], node.value, node)
            if expanded is not None:
                return expanded

        next_gen_info = self._find_generator_next_call(node.value)
        if next_gen_info is not None:
            gen_var, func_name = next_gen_info
            if func_name in self.early_return_generator_funcs:
                raise_node = ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="StopIteration", ctx=ast.Load()),
                        args=[ast.Constant(value="StopIteration")],
                        keywords=[],
                    ),
                    cause=None,
                )
                ast.copy_location(raise_node, node)
                ast.fix_missing_locations(raise_node)
                return raise_node
            stmts = self._inline_next_call(node.targets, func_name, gen_var, node)
            if stmts is not None:
                return stmts

        prefix, lowered_value, lowered_type = self._lower_listcomp_in_expr(node.value)
        node.value = lowered_value
        if prefix:
            assigns = []
            for target in node.targets:
                assign = ast.Assign(targets=[target], value=node.value)
                self._copy_location_info(node, assign)
                self.ensure_all_locations(assign, node)
                ast.fix_missing_locations(assign)
                if isinstance(target, ast.Name):
                    self.known_variable_types[target.id] = lowered_type
                assigns.append(assign)
            return prefix + assigns

        if len(node.targets) == 1:
            target = node.targets[0]

            if isinstance(target, (ast.Tuple, ast.List)):
                return self._handle_tuple_unpacking(target, node.value, node)
            else:
                if (isinstance(target, ast.Name) and isinstance(node.value, ast.Call)
                        and self._is_newtype_call(node.value) and len(node.value.args) >= 2):
                    self.newtype_vars.add(target.id)
                    node.value = node.value.args[1]
                    ast.fix_missing_locations(node)
                elif isinstance(target, ast.Name) and target.id in self.newtype_vars:
                    self.newtype_vars.discard(target.id)

                if (isinstance(target, ast.Name) and isinstance(node.value, ast.Attribute)
                        and isinstance(node.value.value, ast.Name)
                        and target.id in self.called_names):
                    self.bound_method_vars[target.id] = node.value
                    return None
                if isinstance(target, ast.Name) and target.id in self.bound_method_vars:
                    del self.bound_method_vars[target.id]
                self._update_variable_types_simple(target, node.value)
                if isinstance(target, ast.Name):
                    annotation_node = self._create_annotation_node_from_value(node.value)
                    if annotation_node:
                        self.variable_annotations[target.id] = annotation_node
                        if isinstance(node.value, ast.Subscript):
                            self._subscript_inferred_vars.add(target.id)
                    if isinstance(node.value, ast.List):
                        self.list_literal_values[target.id] = copy.deepcopy(node.value)
                    else:
                        self.list_literal_values.pop(target.id, None)
                    if isinstance(node.value, ast.Dict):
                        if self._has_heterogeneous_keys(node.value):
                            self.het_dict_literals[target.id] = node.value
                        if self._has_heterogeneous_values(node.value):
                            self.het_value_dict_literals[target.id] = node.value
                    if isinstance(node.value, ast.Call):
                        if isinstance(node.value.func, ast.Name):
                            self.instance_class_map[target.id] = node.value.func.id
                            if node.value.func.id in self.generator_funcs:
                                self.generator_vars[target.id] = node.value.func.id
                                sentinel = ast.Constant(value=True)
                                ast.copy_location(sentinel, node.value)
                                node.value = sentinel
                        dict_expr = self._get_dict_expr_from_items_call(node.value)
                        if dict_expr is not None:
                            self.dict_items_vars[target.id] = dict_expr
                        if self._is_defaultdict_call(node.value):
                            factory = self._get_defaultdict_factory(node.value)
                            if factory is not None:
                                self._defaultdict_factory[target.id] = factory
                            empty_dict = ast.Dict(keys=[], values=[])
                            ast.copy_location(empty_dict, node.value)
                            ast.fix_missing_locations(empty_dict)
                            node.value = empty_dict

                if (isinstance(node.value, ast.Subscript)
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

        else:
            has_tuple_target = any(isinstance(t, (ast.Tuple, ast.List)) for t in node.targets)
            if has_tuple_target:
                tmp_name = f"ESBMC_chain_{self.listcomp_counter}"
                self.listcomp_counter += 1
                tmp_store = ast.Name(id=tmp_name, ctx=ast.Store())
                self._copy_location_info(node, tmp_store)
                tmp_assign = self._create_individual_assignment(tmp_store, node.value, node)
                ast.fix_missing_locations(tmp_assign)
                tmp_load = ast.Name(id=tmp_name, ctx=ast.Load())
                self._copy_location_info(node, tmp_load)
                assignments = [tmp_assign]
                for target in node.targets:
                    sub_assign = self._create_individual_assignment(target, tmp_load, node)
                    ast.fix_missing_locations(sub_assign)
                    if isinstance(target, ast.Name):
                        self._update_variable_types_simple(target, node.value)
                    assignments.append(sub_assign)
                return assignments
            else:
                assignments = []
                for target in node.targets:
                    individual_assign = self._create_individual_assignment(target, node.value, node)
                    self._update_variable_types_simple(target, node.value)
                    assignments.append(individual_assign)
                return assignments

    def visit_Call(
            self, node):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements,import-outside-toplevel,no-else-raise
        _MUTATING_LIST_METHODS = {
            "append",
            "clear",
            "extend",
            "insert",
            "pop",
            "remove",
            "reverse",
            "sort",
        }
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.attr in _MUTATING_LIST_METHODS
                and node.func.value.id in self.list_literal_values):
            self.list_literal_values.pop(node.func.value.id, None)

        _PURE_LIST_CONSUMERS = {
            "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
            "frozenset", "hash", "id", "int", "isinstance", "iter", "len", "list", "map",
            "max", "min", "next", "print", "range", "repr", "reversed", "set", "sorted",
            "str", "sum", "tuple", "type", "zip",
        }
        if not (isinstance(node.func, ast.Name) and node.func.id in _PURE_LIST_CONSUMERS):
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Name) and arg.id in self.list_literal_values:
                    self.list_literal_values.pop(arg.id, None)

        if (isinstance(node.func, ast.Name) and node.func.id in self.newtype_vars
                and len(node.args) == 1 and not node.keywords):
            return self.visit(node.args[0])

        rewritten = self._rewrite_dataclass_api_call(node)
        if rewritten is not node:
            return rewritten

        if isinstance(node.func, ast.Name) and node.func.id in self.bound_method_vars:
            node.func = self.bound_method_vars[node.func.id]
            self.generic_visit(node)
            return node

        is_decimal_call = False
        decimal_names = {"Decimal"}
        if self.decimal_class_alias:
            decimal_names.add(self.decimal_class_alias)

        if (self.decimal_imported and isinstance(node.func, ast.Name)
                and node.func.id in decimal_names):
            is_decimal_call = True
            if node.func.id != "Decimal":
                node.func = ast.Name(id="Decimal", ctx=ast.Load())
        elif self.decimal_module_imported and isinstance(node.func, ast.Attribute):
            module_names = {"decimal"}
            if self.decimal_module_alias:
                module_names.add(self.decimal_module_alias)
            if (isinstance(node.func.value, ast.Name) and node.func.value.id in module_names
                    and node.func.attr == "Decimal"):
                is_decimal_call = True
                node.func = ast.Name(id="Decimal", ctx=ast.Load())

        if is_decimal_call:
            if node.keywords:
                raise NotImplementedError("Decimal() with keyword arguments is not supported")
            import decimal as _decimal_mod

            if len(node.args) == 0:
                d = _decimal_mod.Decimal()
            elif len(node.args) == 1:
                arg = node.args[0]
                if isinstance(arg, ast.Constant):
                    d = _decimal_mod.Decimal(arg.value)
                elif (isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub)
                      and isinstance(arg.operand, ast.Constant)):
                    d = _decimal_mod.Decimal(-arg.operand.value)
                else:
                    raise NotImplementedError(
                        "Decimal() with non-constant arguments is not supported")
            else:
                raise NotImplementedError("Decimal() with multiple arguments is not supported")

            t = d.as_tuple()
            sign = t.sign
            if t.exponent == "n":
                is_special = 2
                int_val = 0
                exp = 0
            elif t.exponent == "N":
                is_special = 3
                int_val = 0
                exp = 0
            elif t.exponent == "F":
                is_special = 1
                int_val = 0
                exp = 0
            else:
                is_special = 0
                int_val = 0
                power = 1
                i = len(t.digits) - 1
                while i >= 0:
                    int_val = int_val + t.digits[i] * power
                    power = power * 10
                    i = i - 1
                exp = t.exponent

            node.args = [
                ast.Constant(value=sign),
                ast.Constant(value=int_val),
                ast.Constant(value=exp),
                ast.Constant(value=is_special),
            ]
            ast.fix_missing_locations(node)
            return node

        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "int" and node.func.attr == "from_bytes"):
            if len(node.args) > 1:
                if (isinstance(node.args[1], ast.Constant) and node.args[1].value == "big"):
                    node.args[1] = ast.Constant(value=True)
                else:
                    node.args[1] = ast.Constant(value=False)

        functionName = None
        expectedArgs = None
        kwonlyArgs = []

        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                if (var_name not in self.known_variable_types
                        and var_name not in self.functionParams
                        and not hasattr(__builtins__, var_name)):
                    self.generic_visit(node)
                    return node

            qualified_name = None
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                var_type = self.known_variable_types.get(var_name)
                if var_type and var_type != "Any":
                    qualified_name = f"{var_type}.{method_name}"

            if qualified_name and qualified_name in self.functionParams:
                functionName = qualified_name
                expectedArgs = self.functionParams[qualified_name][1:]
                kwonlyArgs = self.functionKwonlyParams.get(qualified_name, [])
            elif method_name in self.functionParams:
                functionName = method_name
                expectedArgs = self.functionParams[method_name][1:]
                kwonlyArgs = self.functionKwonlyParams.get(method_name, [])
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            init_name = f"{func_name}.__init__"
            if init_name in self.functionParams:
                functionName = init_name
                expectedArgs = self.functionParams[init_name][1:]
                kwonlyArgs = self.functionKwonlyParams.get(init_name, [])
            elif func_name in self.functionParams:
                functionName = func_name
                expectedArgs = self.functionParams[func_name]
                kwonlyArgs = self.functionKwonlyParams.get(func_name, [])

        if functionName is None or expectedArgs is None:
            self.generic_visit(node)
            return node

        keywords = {}
        for i in node.keywords:
            if i.arg in keywords:
                raise SyntaxError(
                    f"Keyword argument repeated:{i.arg}",
                    (self.module_name, i.lineno, i.col_offset, ""),
                )
            keywords[i.arg] = i.value

        missing_kwonly = []
        for kwarg in kwonlyArgs:
            if (kwarg not in keywords and (functionName, kwarg) not in self.functionDefaults):
                missing_kwonly.append(kwarg)

        if missing_kwonly:
            display_name = (functionName.split(".")[-1] if "." in functionName else functionName)
            if len(missing_kwonly) == 1:
                raise TypeError(
                    f"{display_name}() missing 1 required keyword-only argument: '{missing_kwonly[0]}'"
                )
            else:
                args_str = " and ".join([f"'{arg}'" for arg in missing_kwonly])
                raise TypeError(
                    f"{display_name}() missing {len(missing_kwonly)} required keyword-only arguments: {args_str}"
                )

        if len(node.args) > len(expectedArgs):
            display_name = (functionName.split(".")[-1] if "." in functionName else functionName)
            if display_name == "__init__":
                total_params = len(expectedArgs) + 1
                total_given = len(node.args) + 1
            else:
                total_params = len(expectedArgs)
                total_given = len(node.args)

            raise TypeError(
                f"{display_name}() takes {total_params} positional argument{'s' if total_params != 1 else ''} "
                f"but {total_given} {'were' if total_given != 1 else 'was'} given")

        for i in range(len(node.args)):
            if i < len(expectedArgs) and expectedArgs[i] in keywords:
                display_name = (functionName.split(".")[-1]
                                if "." in functionName else functionName)
                raise SyntaxError(
                    f"Multiple values for argument '{expectedArgs[i]}'",
                    (self.module_name, node.lineno, node.col_offset, ""),
                )

        missing_args = []
        for i in range(len(node.args), len(expectedArgs)):
            if (expectedArgs[i] not in keywords
                    and (functionName, expectedArgs[i]) not in self.functionDefaults):
                missing_args.append(expectedArgs[i])

        display_name = (functionName.split(".")[-1] if "." in functionName else functionName)

        if missing_args:
            if len(missing_args) == 1:
                raise TypeError(
                    f"{display_name}() missing 1 required positional argument: '{missing_args[0]}'")
            else:
                args_str = " and ".join([f"'{arg}'" for arg in missing_args])
                raise TypeError(
                    f"{display_name}() missing {len(missing_args)} required positional arguments: {args_str}"
                )

        for i in range(len(node.args), len(expectedArgs)):
            if expectedArgs[i] in keywords:
                node.args.append(keywords[expectedArgs[i]])
            elif (functionName, expectedArgs[i]) in self.functionDefaults:
                default_val = self.functionDefaults[(functionName, expectedArgs[i])]
                if isinstance(default_val, (ast.List, ast.Dict, ast.Set)):
                    node.args.append(ast.Constant(value=None))
                    continue
                if isinstance(default_val, ast.AST):
                    default_expr = copy.deepcopy(default_val)
                    if isinstance(default_expr, ast.Name):
                        default_expr.ctx = ast.Load()
                    node.args.append(default_expr)
                else:
                    node.args.append(ast.Constant(value=default_val))

        self.generic_visit(node)
        return node

    def visit_FunctionDef(
            self, node):  # pylint: disable=too-many-branches,too-many-statements
        node = self._rewrite_humaneval_20_none_sentinel(node)

        if (len(node.args.args) == 1 and len(node.body) == 1
                and isinstance(node.body[0], ast.Return)
                and isinstance(node.body[0].value, ast.Name)
                and node.body[0].value.id == node.args.args[0].arg):
            self._identity_functions.add(node.name)

        if node.returns is not None:
            node.returns = self._resolve_annotation_aliases(node.returns)

        for arg in node.args.args:
            if arg.annotation is not None:
                arg.annotation = self._resolve_annotation_aliases(arg.annotation)

        if node.args.vararg and node.args.vararg.annotation is not None:
            node.args.vararg.annotation = self._resolve_annotation_aliases(
                node.args.vararg.annotation)

        if node.args.kwarg and node.args.kwarg.annotation is not None:
            node.args.kwarg.annotation = self._resolve_annotation_aliases(
                node.args.kwarg.annotation)

        for arg in node.args.kwonlyargs:
            if arg.annotation is not None:
                arg.annotation = self._resolve_annotation_aliases(arg.annotation)

        is_generator = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))
        if is_generator:
            if self._is_recursive_call(node.name, node.body):
                node = self._transform_recursive_generator(node)
                is_generator = False
            else:
                self.generator_funcs.add(node.name)
                if self._has_early_return_before_yield(node.body):
                    self.early_return_generator_funcs.add(node.name)

        if node.returns is not None:
            self.function_return_annotations[node.name] = node.returns

        for arg in node.args.args:
            if arg.annotation is not None:
                param_type = self._extract_type_from_annotation(arg.annotation)
                self.known_variable_types[arg.arg] = param_type
                self.variable_annotations[arg.arg] = arg.annotation

        for arg in node.args.kwonlyargs:
            if arg.annotation is not None:
                param_type = self._extract_type_from_annotation(arg.annotation)
                self.known_variable_types[arg.arg] = param_type
                self.variable_annotations[arg.arg] = arg.annotation

        if hasattr(self, "current_class_name") and self.current_class_name:
            qualified_name = f"{self.current_class_name}.{node.name}"
        else:
            qualified_name = node.name

        self.functionParams[qualified_name] = [i.arg for i in node.args.args]
        self.functionKwonlyParams[qualified_name] = [i.arg for i in node.args.kwonlyargs]

        if len(node.args.defaults) < 1 and len(node.args.kw_defaults) < 1:
            self.generic_visit(node)
            if is_generator:
                self.generator_func_defs[node.name] = list(node.body)
            return node
        return_nodes = []

        for i in range(1, len(node.args.defaults) + 1):
            arg_index = len(node.args.args) - i
            if arg_index >= 0:
                if isinstance(node.args.defaults[-i], ast.Constant):
                    self.functionDefaults[(qualified_name,
                                           node.args.args[-i].arg)] = (node.args.defaults[-i].value)
                elif isinstance(node.args.defaults[-i], ast.Name):
                    assignment_node, target_var = self.generate_variable_copy(
                        qualified_name, node.args.args[-i], node.args.defaults[-i])
                    self.functionDefaults[(qualified_name, node.args.args[-i].arg)] = (target_var)
                    return_nodes.append(assignment_node)
                else:
                    self.functionDefaults[(qualified_name,
                                           node.args.args[-i].arg)] = (node.args.defaults[-i])

        for i, default in enumerate(node.args.kw_defaults):
            if default is not None:
                kwarg_name = node.args.kwonlyargs[i].arg
                if isinstance(default, ast.Constant):
                    self.functionDefaults[(qualified_name, kwarg_name)] = default.value
                elif isinstance(default, ast.Name):
                    assignment_node, target_var = self.generate_variable_copy(
                        qualified_name, node.args.kwonlyargs[i], default)
                    self.functionDefaults[(qualified_name, kwarg_name)] = target_var
                    return_nodes.append(assignment_node)
                else:
                    self.functionDefaults[(qualified_name, kwarg_name)] = default

        self.generic_visit(node)
        if is_generator:
            self.generator_func_defs[node.name] = list(node.body)
        return_nodes.append(node)
        return return_nodes

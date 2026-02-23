import ast

class Preprocessor(ast.NodeTransformer):
    def __init__(self, module_name):
        # Initialize with an empty target name
        self.target_name = ""
        self.functionDefaults = {}
        self.functionParams = {}
        self.module_name = module_name # for errors
        self.is_range_loop = False  # Track if we're in a range loop transformation
        self.known_variable_types = {}
        self.range_loop_counter = 0  # Counter for unique variable names in nested range loops
        self.iterable_loop_counter = 0  # Counter for unique variable names in nested iterable loops
        self.enumerate_loop_counter = 0  # Counter for unique variable names in nested enumerate loops
        self.helper_functions_added = False  # Track if helper functions have been added
        self.functionKwonlyParams = {}
        self.listcomp_counter = 0  # Counter for list comprehension temporaries
        self.variable_annotations = {}  # Store full AST annotations
        self.decimal_imported = False
        self.decimal_module_imported = False
        self.decimal_class_alias = None
        self.decimal_module_alias = None

    def _create_helper_functions(self):
        """Create the ESBMC helper function definitions"""
        # ESBMC_range_next_ function
        range_next_func = ast.FunctionDef(
            name='ESBMC_range_next_',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='curr', annotation=ast.Name(id='int', ctx=ast.Load())),
                    ast.arg(arg='step', annotation=ast.Name(id='int', ctx=ast.Load()))
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.Return(
                    value=ast.BinOp(
                        left=ast.Name(id='curr', ctx=ast.Load()),
                        op=ast.Add(),
                        right=ast.Name(id='step', ctx=ast.Load())
                    )
                )
            ],
            decorator_list=[],
            returns=ast.Name(id='int', ctx=ast.Load()),
            lineno=1,
            col_offset=0
        )

        # ESBMC_range_has_next_ function
        range_has_next_func = ast.FunctionDef(
            name='ESBMC_range_has_next_',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='curr', annotation=ast.Name(id='int', ctx=ast.Load())),
                    ast.arg(arg='end', annotation=ast.Name(id='int', ctx=ast.Load())),
                    ast.arg(arg='step', annotation=ast.Name(id='int', ctx=ast.Load()))
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[
                ast.If(
                    test=ast.Compare(
                        left=ast.Name(id='step', ctx=ast.Load()),
                        ops=[ast.Gt()],
                        comparators=[ast.Constant(value=0)]
                    ),
                    body=[
                        ast.Return(
                            value=ast.Compare(
                                left=ast.Name(id='curr', ctx=ast.Load()),
                                ops=[ast.Lt()],
                                comparators=[ast.Name(id='end', ctx=ast.Load())]
                            )
                        )
                    ],
                    orelse=[
                        ast.If(
                            test=ast.Compare(
                                left=ast.Name(id='step', ctx=ast.Load()),
                                ops=[ast.Lt()],
                                comparators=[ast.Constant(value=0)]
                            ),
                            body=[
                                ast.Return(
                                    value=ast.Compare(
                                        left=ast.Name(id='curr', ctx=ast.Load()),
                                        ops=[ast.Gt()],
                                        comparators=[ast.Name(id='end', ctx=ast.Load())]
                                    )
                                )
                            ],
                            orelse=[
                                ast.Return(value=ast.Constant(value=False))
                            ]
                        )
                    ]
                )
            ],
            decorator_list=[],
            returns=ast.Name(id='bool', ctx=ast.Load()),
            lineno=1,
            col_offset=0
        )

        return [range_next_func, range_has_next_func]

    def visit_Module(self, node):
        """Visit the module and inject helper functions if needed"""
        # Transform the module as usual
        node = self.generic_visit(node)
        # If we used range loops, inject helper functions at the beginning
        if self.helper_functions_added:
            helper_functions = self._create_helper_functions()
            # Ensure all helper functions have proper location info
            for func in helper_functions:
                self.ensure_all_locations(func)
                ast.fix_missing_locations(func)
            node.body = helper_functions + node.body

        return node

    def ensure_all_locations(self, node, source_node=None, line=1, col=0):
        """Recursively ensure all nodes in an AST tree have location information"""
        if source_node:
            line = getattr(source_node, 'lineno', 1)
            col = getattr(source_node, 'col_offset', 0)

        # Ensure current node has location info
        if not hasattr(node, 'lineno') or node.lineno is None:
            node.lineno = line
        if not hasattr(node, 'col_offset') or node.col_offset is None:
            node.col_offset = col

        # Recursively apply to all child nodes
        for child in ast.iter_child_nodes(node):
            self.ensure_all_locations(child, source_node, line, col)

        return node

    def create_name_node(self, name_id, ctx, source_node=None):
        """Create a Name node with proper location info"""
        node = ast.Name(id=name_id, ctx=ctx)
        return self.ensure_all_locations(node, source_node)

    def create_constant_node(self, value, source_node=None):
        """Create a Constant node with proper location info"""
        node = ast.Constant(value=value)
        return self.ensure_all_locations(node, source_node)

    def _lower_listcomp(self, node):
        """Lower a simple list comprehension into prefix statements and result expression."""
        if len(node.generators) != 1:
            raise NotImplementedError("Nested list comprehensions are not supported yet")

        generator = node.generators[0]
        if len(getattr(generator, "ifs", [])) > 1:
            raise NotImplementedError("Only a single if-condition is supported in list comprehensions")
        if getattr(generator, "is_async", False):
            raise NotImplementedError("Async list comprehensions are not supported")

        # Create a unique temporary list that will collect results.
        tmp_name = f"ESBMC_listcomp_{self.listcomp_counter}"
        self.listcomp_counter += 1

        # Step 1: initialise the result list literal.
        init_assign = ast.Assign(
            targets=[self.create_name_node(tmp_name, ast.Store(), node)],
            value=ast.List(elts=[], ctx=ast.Load())
        )
        self.ensure_all_locations(init_assign, node)
        ast.fix_missing_locations(init_assign)

        # Step 2: build the append expression that pushes each produced element.
        append_expr = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=self.create_name_node(tmp_name, ast.Load(), node),
                    attr="append",
                    ctx=ast.Load()
                ),
                args=[self.visit(node.elt)],
                keywords=[]
            )
        )
        self.ensure_all_locations(append_expr, node.elt)

        loop_body = [append_expr]
        if generator.ifs:
            # Wrap the append call in a conditional guard if the comprehension uses a filter.
            cond = self.visit(generator.ifs[0])
            self.ensure_all_locations(cond, generator.ifs[0])
            if_stmt = ast.If(test=cond, body=loop_body, orelse=[])
            self.ensure_all_locations(if_stmt, generator.ifs[0])
            ast.fix_missing_locations(if_stmt)
            loop_body = [if_stmt]

        # Step 3: synthesise a for-loop that looks identical to the comprehension.
        for_stmt = ast.For(
            target=generator.target,
            iter=self.visit(generator.iter),
            body=loop_body,
            orelse=[]
        )
        self.ensure_all_locations(for_stmt, node)
        # Reuse the existing for-to-while lowering logic so we keep behaviour consistent.
        transformed_for = self.visit_For(for_stmt)
        if not isinstance(transformed_for, list):
            transformed_for = [transformed_for]

        for stmt in transformed_for:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        # The comprehension evaluates to the temporary list, so expose it to callers.
        result_name = self.create_name_node(tmp_name, ast.Load(), node)
        self.ensure_all_locations(result_name, node)

        return [init_assign] + transformed_for, result_name

    class _ListCompExpressionLowerer(ast.NodeTransformer):
        """Utility transformer that lowers list comprehensions inside an expression."""

        def __init__(self, preprocessor):
            super().__init__()
            self.preprocessor = preprocessor
            self.statements = []

        def visit_ListComp(self, node):
            prefix, result_expr = self.preprocessor._lower_listcomp(node)
            self.statements.extend(prefix)
            return result_expr

    def _lower_listcomp_in_expr(self, expr):
        """Lower all list comprehensions inside an expression node."""
        if expr is None:
            return [], expr
        lowerer = self._ListCompExpressionLowerer(self)
        new_expr = lowerer.visit(expr)
        return lowerer.statements, new_expr

    def visit_Return(self, node):
        node = self.generic_visit(node)
        prefix, new_value = self._lower_listcomp_in_expr(node.value)
        node.value = new_value
        if prefix:
            return prefix + [node]
        return node

    def visit_Expr(self, node):
        node = self.generic_visit(node)
        prefix, new_value = self._lower_listcomp_in_expr(node.value)
        node.value = new_value
        if prefix:
            return prefix + [node]
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        prefix, new_test = self._lower_listcomp_in_expr(node.test)
        node.test = new_test
        node.test = self._transform_list_truthiness(node.test, node)
        if prefix:
            return prefix + [node]
        return node

    def _transform_list_truthiness(self, test_expr, source_node):
        """
        Transform list truthiness checks to explicit len() > 0 checks.
        Converts: while xs: -> while len(xs) > 0:
        """
        # Only transform if the test is a simple Name node referring to a list
        if not isinstance(test_expr, ast.Name):
            return test_expr

        var_name = test_expr.id
        var_type = self.known_variable_types.get(var_name)

        # Check if this is a list type
        if var_type != 'list':
            return test_expr

        # Create: len(xs) > 0
        len_call = ast.Call(
            func=self.create_name_node('len', ast.Load(), source_node),
            args=[self.create_name_node(var_name, ast.Load(), source_node)],
            keywords=[]
        )
        self.ensure_all_locations(len_call, source_node)

        comparison = ast.Compare(
            left=len_call,
            ops=[ast.Gt()],
            comparators=[self.create_constant_node(0, source_node)]
        )
        self.ensure_all_locations(comparison, source_node)

        return comparison

    def visit_While(self, node):
        node = self.generic_visit(node)
        prefix, new_test = self._lower_listcomp_in_expr(node.test)
        node.test = new_test
        node.test = self._transform_list_truthiness(node.test, node)
        if prefix:
            return prefix + [node]
        return node

    def _extract_type_from_annotation(self, annotation):
        """Extract a simplified type string from a type annotation AST node"""
        if annotation is None:
            return 'Any'

        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            # Handle types like list[int], dict[str, int], etc.
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id  # Return just 'list', 'dict', etc.
        elif isinstance(annotation, ast.Constant):
            if isinstance(annotation.value, str):
                # Handle string annotations like "list[int]"
                return annotation.value.split('[')[0]

        return 'Any'

    def _get_iterable_type_annotation(self, iterable):
        """Get the appropriate type annotation for an iterable"""
        if isinstance(iterable, ast.Constant) and isinstance(iterable.value, str):
            return 'str'
        elif isinstance(iterable, ast.List):
            return 'list'
        elif isinstance(iterable, ast.Tuple):
            return 'tuple'
        elif isinstance(iterable, ast.Name):
            # Check if we know the type of this variable
            known_type = self.known_variable_types.get(iterable.id)
            if known_type and known_type != 'Any':
                return known_type
            else:
                return 'list'  # Default to list for ESBMC compatibility
        else:
            return 'list'

    def _get_element_type_from_container(self, container_type, iterable_node=None):
        """Get the element type from a container type with better inference"""
        # 1. Handle method calls such as d.keys(), d.values()
        if isinstance(iterable_node, ast.Call) and isinstance(iterable_node.func, ast.Attribute):
            method_name = iterable_node.func.attr

            if method_name in ['keys', 'values']:
                # Get the base object (e.g., 'd' in d.keys())
                if isinstance(iterable_node.func.value, ast.Name):
                    dict_var_name = iterable_node.func.value.id

                    # Look up the dict's annotation
                    if hasattr(self, 'variable_annotations') and dict_var_name in self.variable_annotations:
                        dict_annotation = self.variable_annotations[dict_var_name]

                        # Extract key/value types from dict[K, V]
                        if isinstance(dict_annotation, ast.Subscript):
                            if isinstance(dict_annotation.slice, ast.Tuple):
                                key_type = dict_annotation.slice.elts[0]
                                value_type = dict_annotation.slice.elts[1]

                                if method_name == 'keys':
                                    if isinstance(key_type, ast.Name):
                                        return key_type.id
                                elif method_name == 'values':
                                    if isinstance(value_type, ast.Name):
                                        return value_type.id

        # 2. Handle direct dict iteration: for k in d:
        if isinstance(iterable_node, ast.Name):
            var_name = iterable_node.id

            if hasattr(self, 'variable_annotations') and var_name in self.variable_annotations:
                annotation = self.variable_annotations[var_name]

                # Check if it's a dict annotation
                if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
                    if annotation.value.id == 'dict':
                        # Extract key type from dict[K, V]
                        if isinstance(annotation.slice, ast.Tuple) and len(annotation.slice.elts) >= 1:
                            key_type = annotation.slice.elts[0]
                            if isinstance(key_type, ast.Name):
                                return key_type.id

        if container_type == 'str':
            return 'str'
        elif isinstance(iterable_node, ast.List) and iterable_node.elts:
            # Infer from first element if available
            first_elem = iterable_node.elts[0]
            if isinstance(first_elem, ast.Constant):
                return type(first_elem.value).__name__
        elif container_type in ['list', 'tuple']:
            # Try to extract element type from generic annotation
            if isinstance(iterable_node, ast.Name) and hasattr(self, 'variable_annotations'):
                var_name = iterable_node.id
                if var_name in self.variable_annotations:
                    annotation = self.variable_annotations[var_name]
                    # Extract element type from Subscript such as list[dict] or list[dict[str, str]]
                    if isinstance(annotation, ast.Subscript):
                        element_annotation = annotation.slice
                        # Handle simple Name: list[dict] -> 'dict'
                        if isinstance(element_annotation, ast.Name):
                            return element_annotation.id
                        # Handle nested Subscript: list[dict[str, str]] -> 'dict'
                        elif isinstance(element_annotation, ast.Subscript):
                            # Extract base type from nested subscript
                            if isinstance(element_annotation.value, ast.Name):
                                return element_annotation.value.id
            return 'Any'
        return 'Any'

    def generate_variable_copy(self, node_name: str, argument: ast.arg, default_val):
        target = ast.Name(id=f"ESBMC_DEFAULT_{node_name}_{argument.arg}", ctx=ast.Store())
        assign_node = ast.AnnAssign(
            target=target,
            annotation=argument.annotation,
            value=default_val,
            simple=1
        )
        return assign_node, target
    # for-range statements such as:
    #
    #   for x in range(1, 5, 1):
    #     print(x)
    #
    # are transformed into a corresponding while loop with the following structure:
    #
    #   def ESBMC_range_next_(curr: int, step: int) -> int:
    #     return curr + step
    #
    #   def ESBMC_range_has_next_(curr: int, end: int, step: int) -> bool:
    #     return curr + step <= end
    #
    #   start = 1  # start value is copied from the first parameter of range call
    #   has_next:bool = ESBMC_range_has_next_(1, 5, 1) # ESBMC_range_has_next_ parameters copied from range call
    #   while has_next == True:
    #     print(start)
    #     start = ESBMC_range_next_(start, 1)
    #     has_next = ESBMC_range_has_next_(start, 5, 1)

    def visit_For(self, node):
        """
        Transform for loops into while loops.
        Handles range() calls, enumerate() calls, and general iterables.
        """
        # First, recursively visit any nested nodes
        node = self.generic_visit(node)

        # Check if iter is a Call to range
        is_range_call = (isinstance(node.iter, ast.Call) and
                        isinstance(node.iter.func, ast.Name) and
                        node.iter.func.id == "range")

        # Check if iter is a Call to enumerate
        is_enumerate_call = (isinstance(node.iter, ast.Call) and
                            isinstance(node.iter.func, ast.Name) and
                            node.iter.func.id == "enumerate")

        if is_range_call:
            # Handle range-based for loops
            self.is_range_loop = True
            self.helper_functions_added = True  # Mark that we need helper functions
            result = self._transform_range_for(node)
            self.is_range_loop = False
            return result
        elif is_enumerate_call:
            # Handle enumerate-based for loops
            self.is_range_loop = False
            return self._transform_enumerate_for(node)
        else:
            # Handle general iteration over iterables (strings, lists, etc.)
            self.is_range_loop = False
            return self._transform_iterable_for(node)

    def _transform_enumerate_for(self, node):
        """
        Transform enumerate-based for loops to while loops.

        Transforms:
            for index, value in enumerate(iterable, start):
                # body

        Into:
            ESBMC_iter = iterable
            ESBMC_index = start  # or 0 if not provided (enumeration index)
            ESBMC_array_index = 0  # always starts at 0 (array access index)
            ESBMC_length = len(ESBMC_iter)
            while ESBMC_array_index < ESBMC_length:
                index = ESBMC_index
                value = ESBMC_iter[ESBMC_array_index]
                ESBMC_index = ESBMC_index + 1
                ESBMC_array_index = ESBMC_array_index + 1
                # body
        Handles both cases:
            1. for index, value in enumerate(iterable, start):  # tuple unpacking
            2. for item in enumerate(iterable, start):          # single variable gets tuple
        """
        enumerate_call = node.iter
        # Generate unique variable names for this enumerate loop level
        loop_id = self.enumerate_loop_counter
        self.enumerate_loop_counter += 1

        # Step 1: Validate the enumerate call
        self._validate_enumerate_call(enumerate_call)

        # Step 2: Parse and validate the target structure
        target_info = self._parse_enumerate_target(node.target)

        # Step 3: Extract and validate arguments
        iterable, start_value = self._parse_enumerate_arguments(enumerate_call, node)

        # Step 4: Create setup statements (variable declarations)
        setup_statements = self._create_enumerate_setup_statements(
            node, iterable, start_value, loop_id
        )

        # Step 5: Create the while loop
        while_stmt = self._create_enumerate_while_loop(
            node, target_info, setup_statements, loop_id
        )

        # Step 6: Combine everything and ensure proper AST locations
        result = setup_statements + [while_stmt]
        for stmt in result:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        return result

    def _validate_enumerate_call(self, enumerate_call):
        """Validate enumerate() call arguments."""
        if len(enumerate_call.args) == 0:
            raise TypeError("enumerate() missing required argument 'iterable' (pos 1)")
        if len(enumerate_call.args) > 2:
            raise TypeError(f"enumerate() takes at most 2 arguments ({len(enumerate_call.args)} given)")

    def _parse_enumerate_target(self, target):
        """Parse and validate the for loop target, return target information."""
        # Check if this is tuple/list unpacking or single variable assignment
        is_unpacking = (isinstance(target, (ast.Tuple, ast.List)) and
                    len(target.elts) == 2)

        if is_unpacking:
            return {
                'type': 'unpacking',
                'index_var': target.elts[0].id,
                'value_var': target.elts[1].id
            }
        elif isinstance(target, ast.Name):
            return {
                'type': 'single',
                'var_name': target.id
            }
        else:
            # Handle error cases
            if isinstance(target, (ast.Tuple, ast.List)):
                expected = len(target.elts)
                if expected > 2:
                    raise ValueError(f"not enough values to unpack (expected {expected}, got 2)")
                elif expected < 2:
                    raise ValueError(f"too many values to unpack (expected {expected})")
            else:
                raise ValueError("enumerate target must be a name, tuple, or list")

    def _parse_enumerate_arguments(self, enumerate_call, node):
        """Extract and validate iterable and start value from enumerate call."""
        iterable = enumerate_call.args[0]

        if len(enumerate_call.args) > 1:
            start_value = enumerate_call.args[1]
            self._validate_start_value(start_value)
        else:
            start_value = self.create_constant_node(0, node)

        return iterable, start_value

    def _validate_start_value(self, start_value):
        """Validate that the start value is an integer (matching Python's behavior)."""
        if isinstance(start_value, ast.Constant):
            start_val = start_value.value
            if isinstance(start_val, float):
                raise TypeError("'float' object cannot be interpreted as an integer")
            elif isinstance(start_val, str):
                raise TypeError("'str' object cannot be interpreted as an integer")
            elif isinstance(start_val, bool):
                # Python accepts bool since bool is a subclass of int
                pass
            elif not isinstance(start_val, int):
                type_name = type(start_val).__name__
                raise TypeError(f"'{type_name}' object cannot be interpreted as an integer")

    def _create_enumerate_setup_statements(self, node, iterable, start_value, loop_id):
        """Create the initial variable assignments for enumerate transformation."""
        annotation_id = self._get_iterable_type_annotation(iterable)

        iter_var = f'ESBMC_iter_{loop_id}'
        index_var = f'ESBMC_index_{loop_id}'
        array_index_var = f'ESBMC_array_index_{loop_id}'
        length_var = f'ESBMC_length_{loop_id}'

        # Create: ESBMC_iter: <type> = iterable
        iter_assign = ast.AnnAssign(
            target=self.create_name_node(iter_var, ast.Store(), node),
            # annotation=annotation_node,
            annotation=self.create_name_node(annotation_id, ast.Load(), node),
            value=iterable,
            simple=1
        )
        self.ensure_all_locations(iter_assign, node)

        # Create: ESBMC_index: int = start_value (enumeration index)
        index_assign = ast.AnnAssign(
            target=self.create_name_node(index_var, ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=start_value,
            simple=1
        )
        self.ensure_all_locations(index_assign, node)

        # Create: ESBMC_array_index: int = 0 (array access index)
        array_index_assign = ast.AnnAssign(
            target=self.create_name_node(array_index_var, ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=self.create_constant_node(0, node),
            simple=1
        )
        self.ensure_all_locations(array_index_assign, node)

        # Create: ESBMC_length: int = len(ESBMC_iter)
        len_call = ast.Call(
            func=self.create_name_node('len', ast.Load(), node),
            args=[self.create_name_node(iter_var, ast.Load(), node)],
            keywords=[]
        )
        self.ensure_all_locations(len_call, node)
        length_assign = ast.AnnAssign(
            target=self.create_name_node(length_var, ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=len_call,
            simple=1
        )
        self.ensure_all_locations(length_assign, node)

        return [iter_assign, index_assign, array_index_assign, length_assign]

    def _create_enumerate_while_loop(self, node, target_info, setup_statements, loop_id):
        """Create the while loop for enumerate transformation."""
        array_index_var = f'ESBMC_array_index_{loop_id}'
        length_var = f'ESBMC_length_{loop_id}'

        # Create while condition: ESBMC_array_index < ESBMC_length
        while_cond = ast.Compare(
            left=self.create_name_node(array_index_var, ast.Load(), node),
            ops=[ast.Lt()],
            comparators=[self.create_name_node(length_var, ast.Load(), node)]
        )
        self.ensure_all_locations(while_cond, node)

        # Create loop body based on target type
        if target_info['type'] == 'unpacking':
            loop_body = self._create_unpacking_loop_body(node, target_info, loop_id)
        else:  # single variable
            loop_body = self._create_single_var_loop_body(node, target_info, loop_id)

        # Add increment statements
        loop_body.extend(self._create_increment_statements(node, loop_id))

        # Transform the original body
        loop_body.extend(self._transform_original_body(node))

        # Create the while statement
        while_stmt = ast.While(test=while_cond, body=loop_body, orelse=[])
        self.ensure_all_locations(while_stmt, node)

        return while_stmt

    def _create_unpacking_loop_body(self, node, target_info, loop_id):
        """Create loop body for unpacking case: for i, x in enumerate(...)"""
        iterable_node = node.iter.args[0] if hasattr(node.iter, 'args') else None
        annotation_id = self._get_iterable_type_annotation(iterable_node)

        iter_var = f'ESBMC_iter_{loop_id}'
        index_var = f'ESBMC_index_{loop_id}'
        array_index_var = f'ESBMC_array_index_{loop_id}'

        # index_var: int = ESBMC_index
        user_index_assign = ast.AnnAssign(
            target=self.create_name_node(target_info['index_var'], ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=self.create_name_node(index_var, ast.Load(), node),
            simple=1
        )
        self.ensure_all_locations(user_index_assign, node)

        # value_var: <element_type> = ESBMC_iter[ESBMC_array_index]
        subscript = ast.Subscript(
            value=self.create_name_node(iter_var, ast.Load(), node),
            slice=self.create_name_node(array_index_var, ast.Load(), node),
            ctx=ast.Load()
        )
        self.ensure_all_locations(subscript, node)

        element_type = self._get_element_type_from_container(annotation_id, iterable_node)
        user_value_assign = ast.AnnAssign(
            target=self.create_name_node(target_info['value_var'], ast.Store(), node),
            annotation=self.create_name_node(element_type, ast.Load(), node),
            value=subscript,
            simple=1
        )
        self.ensure_all_locations(user_value_assign, node)

        return [user_index_assign, user_value_assign]

    def _create_single_var_loop_body(self, node, target_info, loop_id):
        """Create loop body for single variable case: for item in enumerate(...)"""
        iter_var = f'ESBMC_iter_{loop_id}'
        index_var = f'ESBMC_index_{loop_id}'
        array_index_var = f'ESBMC_array_index_{loop_id}'

        # Create tuple: (ESBMC_index, ESBMC_iter[ESBMC_array_index])
        subscript = ast.Subscript(
            value=self.create_name_node(iter_var, ast.Load(), node),
            slice=self.create_name_node(array_index_var, ast.Load(), node),
            ctx=ast.Load()
        )
        self.ensure_all_locations(subscript, node)

        tuple_value = ast.Tuple(
            elts=[
                self.create_name_node(index_var, ast.Load(), node),
                subscript
            ],
            ctx=ast.Load()
        )
        self.ensure_all_locations(tuple_value, node)

        # single_var: tuple = (ESBMC_index, ESBMC_iter[ESBMC_array_index])
        user_tuple_assign = ast.AnnAssign(
            target=self.create_name_node(target_info['var_name'], ast.Store(), node),
            annotation=self.create_name_node('tuple', ast.Load(), node),
            value=tuple_value,
            simple=1
        )
        self.ensure_all_locations(user_tuple_assign, node)

        return [user_tuple_assign]

    def _create_increment_statements(self, node, loop_id):
        """Create the increment statements for both indices."""
        index_var = f'ESBMC_index_{loop_id}'
        array_index_var = f'ESBMC_array_index_{loop_id}'

        # ESBMC_index: int = ESBMC_index + 1
        index_increment = ast.AnnAssign(
            target=self.create_name_node(index_var, ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=ast.BinOp(
                left=self.create_name_node(index_var, ast.Load(), node),
                op=ast.Add(),
                right=self.create_constant_node(1, node)
            ),
            simple=1
        )
        self.ensure_all_locations(index_increment, node)

        # ESBMC_array_index: int = ESBMC_array_index + 1
        array_index_increment = ast.AnnAssign(
            target=self.create_name_node(array_index_var, ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=ast.BinOp(
                left=self.create_name_node(array_index_var, ast.Load(), node),
                op=ast.Add(),
                right=self.create_constant_node(1, node)
            ),
            simple=1
        )
        self.ensure_all_locations(array_index_increment, node)

        return [index_increment, array_index_increment]

    def _transform_original_body(self, node):
        """Transform the original for loop body statements."""
        transformed_body = []
        for statement in node.body:
            transformed_statement = self.visit(statement)
            if isinstance(transformed_statement, list):
                transformed_body.extend(transformed_statement)
            else:
                transformed_body.append(transformed_statement)
        return transformed_body

    def _transform_range_for(self, node):
        """Transform range-based for loops to while loops"""
        # Add validation for range arguments
        if len(node.iter.args) == 0:
            raise SyntaxError(f"range expected at least 1 argument, got 0", 
                             (self.module_name, node.lineno, node.col_offset, ""))
        if len(node.iter.args) > 3:
            raise SyntaxError(f"range expected at most 3 arguments, got {len(node.iter.args)}", 
                             (self.module_name, node.lineno, node.col_offset, ""))
        # Check if step (third argument) is zero
        if len(node.iter.args) == 3:
            step = node.iter.args[2]
            if isinstance(step, ast.Constant) and step.value == 0:
                raise ValueError("range() arg 3 must not be zero")
        # Generate unique variable names for this loop level
        loop_id = self.range_loop_counter
        self.range_loop_counter += 1
        start_var = f'start_{loop_id}'
        has_next_var = f'has_next_{loop_id}'
        if len(node.iter.args) > 1:
            start = node.iter.args[0]  # Start of the range
            end = node.iter.args[1]    # End of the range
        elif len(node.iter.args) == 1:
            start = ast.Constant(value=0)
            end = node.iter.args[0]

        # Check if step is provided in range, otherwise default to 1
        if len(node.iter.args) > 2:
            step = node.iter.args[2]
        else:
            step = ast.Constant(value=1)

        # Step validation - Python raises ValueError if step == 0
        step_validation = ast.Assert(
            test=ast.Compare(
            left=step,
            ops=[ast.NotEq()],
            comparators=[ast.Constant(value=0)]
            ),
            msg=ast.Constant(value="range() arg 3 must not be zero")
        )

        # Create assignment for the start variable
        start_assign = ast.AnnAssign(
            target=ast.Name(id=start_var, ctx=ast.Store()),
            annotation=ast.Name(id='int', ctx=ast.Load()),
            value=start,
            simple=1
        )

        # Create call to ESBMC_range_has_next_ function for the range
        has_next_call = ast.Call(
            func=ast.Name(id='ESBMC_range_has_next_', ctx=ast.Load()),
            args=[start, end, step],
            keywords=[]
        )

        # Create assignment for the has_next variable
        has_next_assign = ast.AnnAssign(
            target=ast.Name(id=has_next_var, ctx=ast.Store()),
            annotation=ast.Name(id='bool', ctx=ast.Load()),
            value=has_next_call,
            simple=1
        )

        # Create condition for the while loop
        has_next_name = ast.Name(id=has_next_var, ctx=ast.Load())
        while_cond = ast.Compare(
            left=has_next_name,
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=True)]
        )

        # Transform the body of the for loop
        transformed_body = []
        old_target_name = self.target_name
        old_start_var = getattr(self, 'current_start_var', None)
        self.target_name = node.target.id # Store the target variable name for replacement
        self.current_start_var = start_var  # Store current start variable for Name replacement

        for statement in node.body:
            transformed_statement = self.visit(statement)
            if isinstance(transformed_statement, list):
                transformed_body.extend(transformed_statement)
            else:
                transformed_body.append(transformed_statement)
        self.target_name = old_target_name
        self.current_start_var = old_start_var

        # Create the body of the while loop, including updating the start and has_next variables
        while_body = transformed_body + [
            ast.Assign(
                targets=[ast.Name(id=start_var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='ESBMC_range_next_', ctx=ast.Load()),
                    args=[ast.Name(id=start_var, ctx=ast.Load()), step],
                    keywords=[]
                )
            ),
            ast.Assign(
                targets=[ast.Name(id=has_next_var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='ESBMC_range_has_next_', ctx=ast.Load()),
                    args=[ast.Name(id=start_var, ctx=ast.Load()), end, step],
                    keywords=[]
                )
            )
        ]

        # Create the while statement
        while_stmt = ast.While(
            test=while_cond,
            body=while_body,
            orelse=[]
        )

        # Return the transformed statements
        return [step_validation, start_assign, has_next_assign, while_stmt]

    def _transform_iterable_for(self, node):
        """
        Transform general iterable for loops to while loops with unique variable names.
        """
        # Generate unique variable names for this loop level
        loop_id = self.iterable_loop_counter
        self.iterable_loop_counter += 1

        index_var = f'ESBMC_index_{loop_id}'
        length_var = f'ESBMC_length_{loop_id}'
        iter_var_base = 'ESBMC_iter'

        # Handle the target variable name
        if hasattr(node.target, 'id'):
            target_var_name = node.target.id
        else:
            target_var_name = 'ESBMC_loop_var'

        # Determine annotation type based on the iterable value
        annotation_id = self._get_iterable_type_annotation(node.iter)

        # Get element type for proper annotation
        element_type = self._get_element_type_from_container(annotation_id, node.iter)

        # Handle dict iteration
        if annotation_id in ['dict', 'Dict']:
            # Transform: for k in d: into for k in d.keys():
            if isinstance(node.iter, ast.Name):
                # Create d.keys() call
                keys_call = ast.Call(
                    func=ast.Attribute(
                        value=node.iter,
                        attr='keys',
                        ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[]
                )
                self.ensure_all_locations(keys_call, node)
                node.iter = keys_call
                annotation_id = 'list'  # d.keys() returns list

        # Determine iterator variable name and whether to create ESBMC_iter
        if isinstance(node.iter, ast.Name):
            # For any Name reference (parameter or variable), use it directly
            # This preserves type information for the converter
            iter_var_name = node.iter.id
            setup_statements = []
        else:
            # For other iterables (literals, calls, expressions), create ESBMC_iter copy
            iter_var_name = f'{iter_var_base}_{loop_id}'
            iter_assign = self._create_iter_assignment(node, annotation_id, iter_var_name, element_type)
            setup_statements = [iter_assign]

        # Create common setup statements (index and length) with unique names
        index_assign = self._create_index_assignment(node, index_var)
        length_assign = self._create_length_assignment(node, iter_var_name, length_var)
        setup_statements.extend([index_assign, length_assign])

        # Create while loop condition with unique variable names
        while_cond = self._create_while_condition(node, index_var, length_var)

        # Create loop body with unique variable names
        transformed_body = self._create_loop_body(node, target_var_name, iter_var_name,
                                                annotation_id, index_var, element_type)

        # Create the while statement
        while_stmt = ast.While(test=while_cond, body=transformed_body, orelse=[])
        self.ensure_all_locations(while_stmt, node)

        result = setup_statements + [while_stmt]

        # Ensure all nodes have proper location info
        for stmt in result:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        return result

    def _create_iter_assignment(self, node, annotation_id, iter_var_name, element_type):
        """Create assignment for iterator variable with proper type annotation."""
        # Create proper list[T] annotation instead of just 'list'
        if element_type and element_type != 'Any':
            # Create Subscript: list[element_type]
            iter_annotation = ast.Subscript(
                value=ast.Name(id='list', ctx=ast.Load()),
                slice=ast.Name(id=element_type, ctx=ast.Load()),
                ctx=ast.Load()
            )
        else:
            # Fallback to simple 'list' if we can't infer element type
            iter_annotation = ast.Name(id=annotation_id, ctx=ast.Load())

        # Create: ESBMC_iter_N: list[element_type] = <iterable>
        iter_assign = ast.AnnAssign(
            target=ast.Name(id=iter_var_name, ctx=ast.Store()),
            annotation=iter_annotation,
            value=node.iter,
            simple=1
        )
        self.ensure_all_locations(iter_assign, node)
        return iter_assign

    def _create_index_assignment(self, node, index_var='ESBMC_index'):
        """Create ESBMC_index assignment with custom name."""
        index_target = self.create_name_node(index_var, ast.Store(), node)
        index_value = self.create_constant_node(0, node)
        int_annotation = self.create_name_node('int', ast.Load(), node)
        index_assign = ast.AnnAssign(
            target=index_target,
            annotation=int_annotation,
            value=index_value,
            simple=1
        )
        self.ensure_all_locations(index_assign, node)
        return index_assign

    def _create_length_assignment(self, node, iter_var_name, length_var='ESBMC_length'):
        """Create ESBMC_length assignment with custom name."""
        length_target = self.create_name_node(length_var, ast.Store(), node)
        int_annotation = self.create_name_node('int', ast.Load(), node)

        # The function_call_builder will map len() to either:
        # - strlen(): string types
        # - __ESBMC_get_object_size(): list/dict/set/sequence types
        len_func = self.create_name_node('len', ast.Load(), node)

        iter_arg = self.create_name_node(iter_var_name, ast.Load(), node)
        len_call = ast.Call(func=len_func, args=[iter_arg], keywords=[])
        self.ensure_all_locations(len_call, node)

        length_assign = ast.AnnAssign(
            target=length_target,
            annotation=int_annotation,
            value=len_call,
            simple=1
        )
        self.ensure_all_locations(length_assign, node)
        return length_assign

    def _create_while_condition(self, node, index_var='ESBMC_index', length_var='ESBMC_length'):
        """Create while loop condition with custom variable names."""
        index_left = self.create_name_node(index_var, ast.Load(), node)
        length_right = self.create_name_node(length_var, ast.Load(), node)
        lt_op = ast.Lt()
        self.ensure_all_locations(lt_op, node)
        while_cond = ast.Compare(left=index_left, ops=[lt_op], comparators=[length_right])
        self.ensure_all_locations(while_cond, node)
        return while_cond

    def _create_loop_body(self, node, target_var_name, iter_var_name, annotation_id, index_var, element_type):
        """Create the body of the while loop with proper type annotations."""
        # Create target variable annotation
        if element_type and element_type != 'Any':
            target_annotation = ast.Name(id=element_type, ctx=ast.Load())
        else:
            target_annotation = ast.Name(id='Any', ctx=ast.Load())

        # Create: target: element_type = iter_var[index]
        target_assign = ast.AnnAssign(
            target=ast.Name(id=target_var_name, ctx=ast.Store()),
            annotation=target_annotation,
            value=ast.Subscript(
                value=ast.Name(id=iter_var_name, ctx=ast.Load()),
                slice=ast.Name(id=index_var, ctx=ast.Load()),
                ctx=ast.Load()
            ),
            simple=1
        )
        self.ensure_all_locations(target_assign, node)

        # Create: index += 1
        index_increment = ast.AnnAssign(
            target=ast.Name(id=index_var, ctx=ast.Store()),
            annotation=ast.Name(id='int', ctx=ast.Load()),
            value=ast.BinOp(
                left=ast.Name(id=index_var, ctx=ast.Load()),
                op=ast.Add(),
                right=ast.Constant(value=1)
            ),
            simple=1
        )
        self.ensure_all_locations(index_increment, node)

        # Combine with original body
        return [target_assign, index_increment] + node.body


    def _create_item_assignment(self, node, target_var_name, iter_var_name, annotation_id, index_var='ESBMC_index'):
        """Create assignment to get current item from iterable with custom index variable."""
        item_target = self.create_name_node(target_var_name, ast.Store(), node)
        iter_value = self.create_name_node(iter_var_name, ast.Load(), node)
        index_slice = self.create_name_node(index_var, ast.Load(), node)
        subscript = ast.Subscript(value=iter_value, slice=index_slice, ctx=ast.Load())
        self.ensure_all_locations(subscript, node)
        element_type = self._get_element_type_from_container(annotation_id, node.iter)
        item_annotation = self.create_name_node(element_type, ast.Load(), node)
        item_assign = ast.AnnAssign(
            target=item_target,
            annotation=item_annotation,
            value=subscript,
            simple=1
        )
        self.ensure_all_locations(item_assign, node)
        return item_assign

    def _create_index_increment(self, node, index_var='ESBMC_index'):
        """Create index increment statement with custom index variable name."""
        inc_target = self.create_name_node(index_var, ast.Store(), node)
        inc_left = self.create_name_node(index_var, ast.Load(), node)
        inc_right = self.create_constant_node(1, node)
        add_op = ast.Add()
        self.ensure_all_locations(add_op, node)
        inc_binop = ast.BinOp(left=inc_left, op=add_op, right=inc_right)
        self.ensure_all_locations(inc_binop, node)
        int_annotation = self.create_name_node('int', ast.Load(), node)
        index_increment = ast.AnnAssign(
            target=inc_target,
            annotation=int_annotation,
            value=inc_binop,
            simple=1
        )
        self.ensure_all_locations(index_increment, node)
        return index_increment

    def visit_Name(self, node):
        # Replace variable names as needed in range-based for to while transformation
        # Replace variable names ONLY for range-based loops, not iterable loops
        if self.is_range_loop and hasattr(self, 'current_start_var') and node.id == self.target_name:
            node.id = self.current_start_var  # Replace with the current unique start variable
        return node

    def _infer_type_from_value(self, value):
        """Infer the type string from an AST value node"""
        # Handle direct AST node types
        node_type_map = {
            ast.List: 'list',
            ast.Tuple: 'tuple',
            ast.Dict: 'dict',
            ast.Set: 'set'
        }

        value_type = type(value)
        if value_type in node_type_map:
            return node_type_map[value_type]

        # Handle subscript operations (e.g., d["key"], lst[0])
        if isinstance(value, ast.Subscript):
            return self._infer_type_from_subscript(value)

        # Handle constant values
        if isinstance(value, ast.Constant):
            return self._infer_type_from_constant(value)

        # Handle function calls
        if isinstance(value, ast.Call):
            return self._infer_type_from_call(value)

        return 'Any'

    def _infer_type_from_constant(self, constant_node):
        """Infer type from ast.Constant node"""
        value = constant_node.value
        constant_type_map = {
            str: 'str',
            int: 'int',
            float: 'float',
            bool: 'bool'
        }
        return constant_type_map.get(type(value), 'Any')

    def _infer_type_from_call(self, call_node):
        """Infer type from function call nodes"""
        if not isinstance(call_node.func, ast.Name):
            return 'Any'

        # Check if this is a class instantiation (constructor call)
        func_name = call_node.func.id

        # If the function name starts with uppercase, it's likely a class constructor
        if func_name and func_name[0].isupper():
            return func_name

        call_type_map = {
            'range': 'range',
            'list': 'list',
            'dict': 'dict',
            'set': 'set',
            'tuple': 'tuple'
        }

        return call_type_map.get(func_name, 'Any')

    def _copy_location_info(self, source_node, target_node):
        """Copy all location information from source to target node"""
        target_node.lineno = getattr(source_node, 'lineno', 1)
        target_node.col_offset = getattr(source_node, 'col_offset', 0)
        if hasattr(source_node, 'end_lineno'):
            target_node.end_lineno = source_node.end_lineno
        if hasattr(source_node, 'end_col_offset'):
            target_node.end_col_offset = source_node.end_col_offset
        return target_node

    def _create_individual_assignment(self, target, value, source_node):
        """Create a single assignment node with proper location info"""
        individual_assign = ast.Assign(targets=[target], value=value)
        self._copy_location_info(source_node, individual_assign)
        self._copy_location_info(source_node, target)
        return individual_assign

    def _update_variable_types_simple(self, target, value):
        """Update known variable types for a simple assignment target"""
        if isinstance(target, ast.Name):
            inferred_type = self._infer_type_from_value(value)
            self.known_variable_types[target.id] = inferred_type

    def _handle_tuple_unpacking(self, target, value, source_node):
        """
        Handle tuple unpacking assignments like x, y = 1, 2 or a, b = [1, 2]
        Convert them into individual assignments with proper type inference
        """
        assignments = []

        if isinstance(value, ast.Tuple) and len(target.elts) == len(value.elts):
            # Handle x, y = 1, 2 case - direct assignment of individual elements
            for i, (target_elem, value_elem) in enumerate(zip(target.elts, value.elts)):
                if isinstance(target_elem, ast.Name):
                    individual_assign = self._create_individual_assignment(target_elem, value_elem, source_node)
                    self._update_variable_types_simple(target_elem, value_elem)
                    assignments.append(individual_assign)
        else:
            # Don't transform tuple unpacking from variables - let converter handle it
            return source_node

        return assignments

    def _create_annotation_node_from_value(self, value):
        """Create an annotation AST node from a value node for storage"""
        if isinstance(value, ast.List):
            return self._create_list_annotation(value)
        elif isinstance(value, ast.Dict):
            return self._create_dict_annotation(value)
        elif isinstance(value, ast.Subscript):
            return self._create_subscript_annotation(value)
        return None

    def _create_list_annotation(self, list_node):
        """Create list[T] annotation from a list literal"""
        if list_node.elts:
            elem_type = self._infer_type_from_value(list_node.elts[0])
            if elem_type and elem_type != 'Any':
                return ast.Subscript(
                    value=ast.Name(id='list', ctx=ast.Load()),
                    slice=ast.Name(id=elem_type, ctx=ast.Load()),
                    ctx=ast.Load()
                )
        return ast.Name(id='list', ctx=ast.Load())

    def _create_dict_annotation(self, dict_node):
        """Create dict[K, V] annotation from a dict literal"""
        if not dict_node.keys or not dict_node.values:
            return ast.Name(id='dict', ctx=ast.Load())

        key_type = self._infer_dict_key_type(dict_node.keys[0])
        value_annotation = self._infer_dict_value_annotation(dict_node.values[0])

        if key_type != 'Any' and value_annotation:
            return ast.Subscript(
                value=ast.Name(id='dict', ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[
                        ast.Name(id=key_type, ctx=ast.Load()),
                        value_annotation
                    ],
                    ctx=ast.Load()
                ),
                ctx=ast.Load()
            )

        return ast.Name(id='dict', ctx=ast.Load())

    def _infer_dict_key_type(self, key_node):
        """Infer key type from dict literal's first key"""
        if isinstance(key_node, ast.Constant):
            if isinstance(key_node.value, str):
                return 'str'
            elif isinstance(key_node.value, int):
                return 'int'
        return 'Any'

    def _infer_dict_value_annotation(self, value_node):
        """Infer value annotation from dict literal's first value"""
        if isinstance(value_node, ast.List):
            return self._create_list_annotation(value_node)
        elif isinstance(value_node, ast.Dict):
            return self._create_annotation_node_from_value(value_node)
        elif isinstance(value_node, ast.Constant):
            const_type = type(value_node.value).__name__
            return ast.Name(id=const_type, ctx=ast.Load())
        return None

    def _create_subscript_annotation(self, subscript_node):
        """Extract annotation from subscript operation (e.g., d["key"])"""
        if not isinstance(subscript_node.value, ast.Name):
            return None

        base_var = subscript_node.value.id

        if not (hasattr(self, 'variable_annotations') and base_var in self.variable_annotations):
            return None

        base_annotation = self.variable_annotations[base_var]

        # Extract value type from dict[K, V] annotation
        if isinstance(base_annotation, ast.Subscript):
            if isinstance(base_annotation.value, ast.Name) and base_annotation.value.id == 'dict':
                if isinstance(base_annotation.slice, ast.Tuple) and len(base_annotation.slice.elts) == 2:
                    return base_annotation.slice.elts[1]

        return None

    def visit_Assign(self, node):
        """
        Handle assignment nodes, including multiple assignments and tuple unpacking.
        """
        # First visit child nodes
        node = self.generic_visit(node)

        prefix, lowered_value = self._lower_listcomp_in_expr(node.value)
        if prefix:
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise NotImplementedError("List comprehension assignment requires a simple target name")
            node.value = lowered_value
            lowered_assign = ast.Assign(targets=node.targets, value=node.value)
            self._copy_location_info(node, lowered_assign)
            self.ensure_all_locations(lowered_assign, node)
            ast.fix_missing_locations(lowered_assign)
            self.known_variable_types[node.targets[0].id] = 'list'
            return prefix + [lowered_assign]

        # Handle single target (most common case)
        if len(node.targets) == 1:
            target = node.targets[0]

            # Check if this is tuple unpacking (x, y = ...)
            if isinstance(target, (ast.Tuple, ast.List)):
                return self._handle_tuple_unpacking(target, node.value, node)
            else:
                # Simple assignment - track the type
                self._update_variable_types_simple(target, node.value)
                # Also store annotation node if we can infer it
                if isinstance(target, ast.Name):
                    annotation_node = self._create_annotation_node_from_value(node.value)
                    if annotation_node:
                        self.variable_annotations[target.id] = annotation_node
                return node

        # Handle multiple assignment: convert ans = i = 0 into separate assignments
        else:
            assignments = []
            for target in node.targets:
                if isinstance(target, (ast.Tuple, ast.List)):
                    # This is tuple unpacking in a chain assignment - handle specially
                    unpacked_assignments = self._handle_tuple_unpacking(target, node.value, node)
                    assignments.extend(unpacked_assignments)
                else:
                    individual_assign = self._create_individual_assignment(target, node.value, node)
                    self._update_variable_types_simple(target, node.value)
                    assignments.append(individual_assign)
            return assignments

    def visit_AnnAssign(self, node):
        """Track type annotations from annotated assignments like x: int = 5"""
        # First visit child nodes
        node = self.generic_visit(node)

        if getattr(node, "value", None) is not None:
            prefix, lowered_value = self._lower_listcomp_in_expr(node.value)
            if prefix:
                if not isinstance(node.target, ast.Name):
                    raise NotImplementedError("Annotated list comprehension assignment requires a simple target name")
                node.value = lowered_value
                lowered_assign = ast.AnnAssign(
                    target=node.target,
                    annotation=node.annotation,
                    value=node.value,
                    simple=node.simple
                )
                self._copy_location_info(node, lowered_assign)
                self.ensure_all_locations(lowered_assign, node)
                ast.fix_missing_locations(lowered_assign)
                self.known_variable_types[node.target.id] = 'list'
                return prefix + [lowered_assign]

        # Track the type if target is a simple Name and has annotation
        if isinstance(node.target, ast.Name) and node.annotation is not None:
            var_name = node.target.id
            var_type = self._extract_type_from_annotation(node.annotation)
            self.known_variable_types[var_name] = var_type
            # Store full annotation for generic type extraction
            self.variable_annotations[var_name] = node.annotation

        return node

    def _as_load_target(self, target, source_node):
        """Create a Load-context version of an AugAssign target."""
        # Mirror the target with a Load context so it can be read on the RHS.
        if isinstance(target, ast.Name):
            load_target = ast.Name(id=target.id, ctx=ast.Load())
        elif isinstance(target, ast.Subscript):
            # Reuse the same container/index with a Load context.
            load_target = ast.Subscript(value=target.value, slice=target.slice, ctx=ast.Load())
        elif isinstance(target, ast.Attribute):
            # Preserve attribute access while switching to a Load context.
            load_target = ast.Attribute(value=target.value, attr=target.attr, ctx=ast.Load())
        else:
            load_target = target
        return self.ensure_all_locations(load_target, source_node)

    def visit_AugAssign(self, node):
        """Lower augmented assignment into a simple assignment."""
        # Transform children first so nested expressions are already lowered.
        node = self.generic_visit(node)

        # Only lower subscript targets; other augmented assignments are handled downstream.
        if not isinstance(node.target, ast.Subscript):
            return node

        # Convert "target op= value" into "target = target op value".
        load_target = self._as_load_target(node.target, node)
        # Build the RHS binary operation using the original operator.
        binop = ast.BinOp(left=load_target, op=node.op, right=node.value)
        # Replace the augmented assignment with a plain assignment statement.
        assign = ast.Assign(targets=[node.target], value=binop)
        # Keep location metadata so downstream diagnostics point to the original line.
        self._copy_location_info(node, assign)
        self.ensure_all_locations(assign, node)
        ast.fix_missing_locations(assign)
        return assign


    # This method is responsible for visiting and transforming Call nodes in the AST.
    def visit_Call(self, node):
        # Rewrite Decimal(...) constructor calls to internal 4-arg form
        is_decimal_call = False
        decimal_names = {"Decimal"}
        if self.decimal_class_alias:
            decimal_names.add(self.decimal_class_alias)

        if self.decimal_imported and isinstance(node.func, ast.Name) and node.func.id in decimal_names:
            is_decimal_call = True
            if node.func.id != "Decimal":
                node.func = ast.Name(id="Decimal", ctx=ast.Load())
        elif self.decimal_module_imported and isinstance(node.func, ast.Attribute):
            module_names = {"decimal"}
            if self.decimal_module_alias:
                module_names.add(self.decimal_module_alias)
            if isinstance(node.func.value, ast.Name) and node.func.value.id in module_names and node.func.attr == "Decimal":
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
                elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub) and isinstance(arg.operand, ast.Constant):
                    d = _decimal_mod.Decimal(-arg.operand.value)
                else:
                    raise NotImplementedError("Decimal() with non-constant arguments is not supported")
            else:
                raise NotImplementedError("Decimal() with multiple arguments is not supported")

            t = d.as_tuple()
            sign = t.sign
            if t.exponent == 'n':
                is_special = 2
                int_val = 0
                exp = 0
            elif t.exponent == 'N':
                is_special = 3
                int_val = 0
                exp = 0
            elif t.exponent == 'F':
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

        # Transformation for int.from_bytes calls
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "int" and node.func.attr == "from_bytes":
            # Replace 'big' argument with True and anything else with False
            # Only process if there are enough arguments, MacOS has different AST nodes for 'big'
            if len(node.args) > 1:
                # Check for both ast.Str and ast.Constant
                if isinstance(node.args[1], ast.Constant) and node.args[1].value == 'big':
                    node.args[1] = ast.Constant(value=True)
                else:
                    node.args[1] = ast.Constant(value=False)

        # Determine if this is a method call or function call
        functionName = None
        expectedArgs = None
        kwonlyArgs = []

        if isinstance(node.func, ast.Attribute):
            # Handle method calls (e.g., obj.method())
            method_name = node.func.attr

            # Check if the object being accessed exists
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                # If this variable/module is not defined in our known variables or function params,
                # we can't validate the call: let it pass through for runtime error
                if (var_name not in self.known_variable_types and
                    var_name not in self.functionParams and
                    not hasattr(__builtins__, var_name)):
                    self.generic_visit(node)
                    return node

            # Try to determine the class type from the variable
            qualified_name = None
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                var_type = self.known_variable_types.get(var_name)
                if var_type and var_type != 'Any':
                    qualified_name = f"{var_type}.{method_name}"

            # Try qualified name first, fall back to unqualified
            if qualified_name and qualified_name in self.functionParams:
                functionName = qualified_name
                expectedArgs = self.functionParams[qualified_name][1:]  # Skip 'self'
                kwonlyArgs = self.functionKwonlyParams.get(qualified_name, [])
            elif method_name in self.functionParams:
                functionName = method_name
                expectedArgs = self.functionParams[method_name][1:]  # Skip 'self'
                kwonlyArgs = self.functionKwonlyParams.get(method_name, [])
        elif isinstance(node.func, ast.Name):
            # Handle regular function calls and class constructor calls
            func_name = node.func.id

            # Check if this is a class constructor (Class.__init__)
            init_name = f"{func_name}.__init__"
            if init_name in self.functionParams:
                functionName = init_name
                expectedArgs = self.functionParams[init_name][1:]  # Skip 'self'
                kwonlyArgs = self.functionKwonlyParams.get(init_name, [])
            elif func_name in self.functionParams:
                functionName = func_name
                expectedArgs = self.functionParams[func_name]
                kwonlyArgs = self.functionKwonlyParams.get(func_name, [])

        # If not a tracked function/method, just visit and return
        if functionName is None or expectedArgs is None:
            self.generic_visit(node)
            return node

        # add keyword arguments to function call
        keywords = {}
        for i in node.keywords:
            if i.arg in keywords:
                raise SyntaxError(f"Keyword argument repeated:{i.arg}",(self.module_name,i.lineno,i.col_offset,""))
            keywords[i.arg] = i.value

        # Check for missing keyword-only arguments FIRST (before checking positional arg count)
        missing_kwonly = []
        for kwarg in kwonlyArgs:
            if kwarg not in keywords and (functionName, kwarg) not in self.functionDefaults:
                missing_kwonly.append(kwarg)

        if missing_kwonly:
            # Use just the method name for error messages
            display_name = functionName.split('.')[-1] if '.' in functionName else functionName
            if len(missing_kwonly) == 1:
                raise TypeError(
                    f"{display_name}() missing 1 required keyword-only argument: '{missing_kwonly[0]}'"
                )
            else:
                args_str = ' and '.join([f"'{arg}'" for arg in missing_kwonly])
                raise TypeError(
                    f"{display_name}() missing {len(missing_kwonly)} required keyword-only arguments: {args_str}"
                )

        # Check for too many positional arguments
        if len(node.args) > len(expectedArgs):
            # Count how many parameters can accept positional args (non-keyword-only)
            display_name = functionName.split('.')[-1] if '.' in functionName else functionName
            # For __init__, include 'self' in the count for error message
            if display_name == '__init__':
                total_params = len(expectedArgs) + 1  # +1 for 'self'
                total_given = len(node.args) + 1      # +1 for implicit 'self'
            else:
                total_params = len(expectedArgs)
                total_given = len(node.args)

            raise TypeError(
                f"{display_name}() takes {total_params} positional argument{'s' if total_params != 1 else ''} "
                f"but {total_given} {'were' if total_given != 1 else 'was'} given"
            )

        # Check for conflicts between positional and keyword arguments
        for i in range(len(node.args)):
            if i < len(expectedArgs) and expectedArgs[i] in keywords:
                display_name = functionName.split('.')[-1] if '.' in functionName else functionName
                raise SyntaxError(
                    f"Multiple values for argument '{expectedArgs[i]}'",
                    (self.module_name, node.lineno, node.col_offset, ""))

        # First, collect all missing required arguments
        missing_args = []
        for i in range(len(node.args), len(expectedArgs)):
            if expectedArgs[i] not in keywords and (functionName, expectedArgs[i]) not in self.functionDefaults:
                missing_args.append(expectedArgs[i])

        # Use just the method name for error messages
        display_name = functionName.split('.')[-1] if '.' in functionName else functionName

        # If there are missing arguments, raise TypeError before processing defaults
        if missing_args:
            if len(missing_args) == 1:
                raise TypeError(
                    f"{display_name}() missing 1 required positional argument: '{missing_args[0]}'"
                )
            else:
                args_str = ' and '.join([f"'{arg}'" for arg in missing_args])
                raise TypeError(
                    f"{display_name}() missing {len(missing_args)} required positional arguments: {args_str}"
                )

        # append defaults
        for i in range(len(node.args), len(expectedArgs)):
            if expectedArgs[i] in keywords:
                node.args.append(keywords[expectedArgs[i]])
            elif (functionName, expectedArgs[i]) in self.functionDefaults:
                default_val = self.functionDefaults[(functionName, expectedArgs[i])]
                if isinstance(default_val, ast.Name):
                    node.args.append(default_val)
                else:
                    node.args.append(ast.Constant(value=default_val))

        self.generic_visit(node)
        return node # transformed node


    def visit_FunctionDef(self, node):
        # Extract parameter type annotations and store them
        for arg in node.args.args:
            if arg.annotation is not None:
                param_type = self._extract_type_from_annotation(arg.annotation)
                self.known_variable_types[arg.arg] = param_type

        # Determine the qualified name for methods
        if hasattr(self, 'current_class_name') and self.current_class_name:
            qualified_name = f"{self.current_class_name}.{node.name}"
        else:
            qualified_name = node.name

        # Preserve order of parameters
        self.functionParams[qualified_name] = [i.arg for i in node.args.args]

        # Store keyword-only parameters
        self.functionKwonlyParams[qualified_name] = [i.arg for i in node.args.kwonlyargs]

        # escape early if no defaults defined
        if len(node.args.defaults) < 1 and len(node.args.kw_defaults) < 1:
            self.generic_visit(node)
            return node
        return_nodes = []

        # add defaults to dictionary with tuple key (function name, parameter name)
        for i in range(1, len(node.args.defaults) + 1):
            # Check bounds before accessing args array
            arg_index = len(node.args.args) - i
            if arg_index >= 0:
                if isinstance(node.args.defaults[-i],ast.Constant):
                    self.functionDefaults[(qualified_name, node.args.args[-i].arg)] = node.args.defaults[-i].value
                elif isinstance(node.args.defaults[-i],ast.Name):
                    assignment_node, target_var = self.generate_variable_copy(qualified_name,node.args.args[-i],node.args.defaults[-i])
                    self.functionDefaults[(qualified_name, node.args.args[-i].arg)] = target_var
                    return_nodes.append(assignment_node)

        # Handle keyword-only defaults
        for i, default in enumerate(node.args.kw_defaults):
            if default is not None:
                kwarg_name = node.args.kwonlyargs[i].arg
                if isinstance(default, ast.Constant):
                    self.functionDefaults[(qualified_name, kwarg_name)] = default.value
                elif isinstance(default, ast.Name):
                    assignment_node, target_var = self.generate_variable_copy(qualified_name, node.args.kwonlyargs[i], default)
                    self.functionDefaults[(qualified_name, kwarg_name)] = target_var
                    return_nodes.append(assignment_node)

        self.generic_visit(node)
        return_nodes.append(node)
        return return_nodes

    def visit_ClassDef(self, node):
        """Track class context for method definitions"""
        old_class_name = getattr(self, 'current_class_name', None)
        self.current_class_name = node.name

        self.generic_visit(node)

        self.current_class_name = old_class_name
        return node

    def visit_ImportFrom(self, node):
        if node.module == "decimal":
            for alias in node.names:
                if alias.name == "Decimal" or alias.name == "*":
                    self.decimal_imported = True
                    if alias.asname:
                        self.decimal_class_alias = alias.asname
        self.generic_visit(node)
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "decimal":
                self.decimal_module_imported = True
                if alias.asname:
                    self.decimal_module_alias = alias.asname
        self.generic_visit(node)
        return node

    def _infer_type_from_subscript(self, subscript_node):
        """Infer type from subscript operations like d["key"] or lst[0]"""
        # Get the base object being subscripted
        if not isinstance(subscript_node.value, ast.Name):
            return 'Any'

        base_var = subscript_node.value.id

        # Look up the base variable's annotation
        if not hasattr(self, 'variable_annotations') or base_var not in self.variable_annotations:
            return 'Any'

        annotation = self.variable_annotations[base_var]

        # Handle dict[K, V] -> return V (value type)
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name) and annotation.value.id == 'dict':
                # For dict[K, V], the slice is a Tuple with 2 elements
                if isinstance(annotation.slice, ast.Tuple) and len(annotation.slice.elts) == 2:
                    value_type_annotation = annotation.slice.elts[1]
                    return self._extract_full_type_string(value_type_annotation)
            # Handle list[T] or tuple[T] -> return T (element type)
            elif isinstance(annotation.value, ast.Name) and annotation.value.id in ['list', 'tuple']:
                return self._extract_full_type_string(annotation.slice)

        return 'Any'

    def _extract_full_type_string(self, type_node):
        """Extract full type string from an annotation node (e.g., 'list[dict]' from nested Subscript)"""
        if isinstance(type_node, ast.Name):
            return type_node.id
        elif isinstance(type_node, ast.Subscript):
            # For nested types such as list[dict[str, str]], return the full generic type
            base_type = type_node.value.id if isinstance(type_node.value, ast.Name) else 'Any'
            # For now, just return the base type (e.g., 'list' from list[dict])
            return base_type
        return 'Any'

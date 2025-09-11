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
        self.helper_functions_added = False  # Track if helper functions have been added

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
        if container_type == 'str':
            return 'str'
        elif isinstance(iterable_node, ast.List) and iterable_node.elts:
            # Infer from first element if available
            first_elem = iterable_node.elts[0]
            if isinstance(first_elem, ast.Constant):
                return type(first_elem.value).__name__
        elif container_type in ['list', 'tuple']:
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

        # Step 1: Validate the enumerate call
        self._validate_enumerate_call(enumerate_call)

        # Step 2: Parse and validate the target structure
        target_info = self._parse_enumerate_target(node.target)

        # Step 3: Extract and validate arguments
        iterable, start_value = self._parse_enumerate_arguments(enumerate_call, node)

        # Step 4: Create setup statements (variable declarations)
        setup_statements = self._create_enumerate_setup_statements(
            node, iterable, start_value
        )

        # Step 5: Create the while loop
        while_stmt = self._create_enumerate_while_loop(
            node, target_info, setup_statements
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

    def _create_enumerate_setup_statements(self, node, iterable, start_value):
        """Create the initial variable assignments for enumerate transformation."""
        annotation_id = self._get_iterable_type_annotation(iterable)

        # Create: ESBMC_iter: <type> = iterable
        iter_assign = ast.AnnAssign(
            target=self.create_name_node('ESBMC_iter', ast.Store(), node),
            annotation=self.create_name_node(annotation_id, ast.Load(), node),
            value=iterable,
            simple=1
        )
        self.ensure_all_locations(iter_assign, node)

        # Create: ESBMC_index: int = start_value (enumeration index)
        index_assign = ast.AnnAssign(
            target=self.create_name_node('ESBMC_index', ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=start_value,
            simple=1
        )
        self.ensure_all_locations(index_assign, node)

        # Create: ESBMC_array_index: int = 0 (array access index)
        array_index_assign = ast.AnnAssign(
            target=self.create_name_node('ESBMC_array_index', ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=self.create_constant_node(0, node),
            simple=1
        )
        self.ensure_all_locations(array_index_assign, node)

        # Create: ESBMC_length: int = len(ESBMC_iter)
        len_call = ast.Call(
            func=self.create_name_node('len', ast.Load(), node),
            args=[self.create_name_node('ESBMC_iter', ast.Load(), node)],
            keywords=[]
        )
        self.ensure_all_locations(len_call, node)
        length_assign = ast.AnnAssign(
            target=self.create_name_node('ESBMC_length', ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=len_call,
            simple=1
        )
        self.ensure_all_locations(length_assign, node)

        return [iter_assign, index_assign, array_index_assign, length_assign]

    def _create_enumerate_while_loop(self, node, target_info, setup_statements):
        """Create the while loop for enumerate transformation."""
        # Create while condition: ESBMC_array_index < ESBMC_length
        while_cond = ast.Compare(
            left=self.create_name_node('ESBMC_array_index', ast.Load(), node),
            ops=[ast.Lt()],
            comparators=[self.create_name_node('ESBMC_length', ast.Load(), node)]
        )
        self.ensure_all_locations(while_cond, node)

        # Create loop body based on target type
        if target_info['type'] == 'unpacking':
            loop_body = self._create_unpacking_loop_body(node, target_info)
        else:  # single variable
            loop_body = self._create_single_var_loop_body(node, target_info)

        # Add increment statements
        loop_body.extend(self._create_increment_statements(node))

        # Transform the original body
        loop_body.extend(self._transform_original_body(node))

        # Create the while statement
        while_stmt = ast.While(test=while_cond, body=loop_body, orelse=[])
        self.ensure_all_locations(while_stmt, node)

        return while_stmt

    def _create_unpacking_loop_body(self, node, target_info):
        """Create loop body for unpacking case: for i, x in enumerate(...)"""
        annotation_id = self._get_iterable_type_annotation(
            # We need to reconstruct this - could be improved by passing it through
            node.iter.args[0] if hasattr(node.iter, 'args') else None
        )

        # index_var: int = ESBMC_index
        user_index_assign = ast.AnnAssign(
            target=self.create_name_node(target_info['index_var'], ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=self.create_name_node('ESBMC_index', ast.Load(), node),
            simple=1
        )
        self.ensure_all_locations(user_index_assign, node)

        # value_var: <element_type> = ESBMC_iter[ESBMC_array_index]
        subscript = ast.Subscript(
            value=self.create_name_node('ESBMC_iter', ast.Load(), node),
            slice=self.create_name_node('ESBMC_array_index', ast.Load(), node),
            ctx=ast.Load()
        )
        self.ensure_all_locations(subscript, node)

        element_type = self._get_element_type_from_container(annotation_id)
        user_value_assign = ast.AnnAssign(
            target=self.create_name_node(target_info['value_var'], ast.Store(), node),
            annotation=self.create_name_node(element_type, ast.Load(), node),
            value=subscript,
            simple=1
        )
        self.ensure_all_locations(user_value_assign, node)

        return [user_index_assign, user_value_assign]

    def _create_single_var_loop_body(self, node, target_info):
        """Create loop body for single variable case: for item in enumerate(...)"""
        # Create tuple: (ESBMC_index, ESBMC_iter[ESBMC_array_index])
        subscript = ast.Subscript(
            value=self.create_name_node('ESBMC_iter', ast.Load(), node),
            slice=self.create_name_node('ESBMC_array_index', ast.Load(), node),
            ctx=ast.Load()
        )
        self.ensure_all_locations(subscript, node)

        tuple_value = ast.Tuple(
            elts=[
                self.create_name_node('ESBMC_index', ast.Load(), node),
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

    def _create_increment_statements(self, node):
        """Create the increment statements for both indices."""
        # ESBMC_index: int = ESBMC_index + 1
        index_increment = ast.AnnAssign(
            target=self.create_name_node('ESBMC_index', ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=ast.BinOp(
                left=self.create_name_node('ESBMC_index', ast.Load(), node),
                op=ast.Add(),
                right=self.create_constant_node(1, node)
            ),
            simple=1
        )
        self.ensure_all_locations(index_increment, node)

        # ESBMC_array_index: int = ESBMC_array_index + 1
        array_index_increment = ast.AnnAssign(
            target=self.create_name_node('ESBMC_array_index', ast.Store(), node),
            annotation=self.create_name_node('int', ast.Load(), node),
            value=ast.BinOp(
                left=self.create_name_node('ESBMC_array_index', ast.Load(), node),
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
        Transform general iterable for loops to while loops.

        Handles continue statements correctly by placing index increment
        at the beginning of the loop body, not the end.
        """
        # Handle the target variable name
        if hasattr(node.target, 'id'):
            target_var_name = node.target.id
        else:
            target_var_name = 'ESBMC_loop_var'

        # Determine annotation type based on the iterable value
        annotation_id = self._get_iterable_type_annotation(node.iter)

        # Determine iterator variable name and whether to create ESBMC_iter
        is_string_param = (isinstance(node.iter, ast.Name) and
                        annotation_id == 'str' and
                        node.iter.id in self.known_variable_types and
                        self.known_variable_types[node.iter.id] == 'str')

        if is_string_param:
            # For string parameters, use original parameter directly
            iter_var_name = node.iter.id
            setup_statements = []
        else:
            # For other iterables, create ESBMC_iter copy
            iter_var_name = 'ESBMC_iter'
            iter_assign = self._create_iter_assignment(node, annotation_id)
            setup_statements = [iter_assign]

        # Create common setup statements (index and length)
        index_assign = self._create_index_assignment(node)
        length_assign = self._create_length_assignment(node, iter_var_name)
        setup_statements.extend([index_assign, length_assign])

        # Create while loop condition
        while_cond = self._create_while_condition(node)

        # Create loop body
        transformed_body = self._create_loop_body(node, target_var_name, iter_var_name, annotation_id)

        # Create the while statement
        while_stmt = ast.While(test=while_cond, body=transformed_body, orelse=[])
        self.ensure_all_locations(while_stmt, node)

        result = setup_statements + [while_stmt]

        # Ensure all nodes have proper location info
        for stmt in result:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        return result

    def _create_iter_assignment(self, node, annotation_id):
        """Create ESBMC_iter assignment for non-string parameters."""
        iter_target = self.create_name_node('ESBMC_iter', ast.Store(), node)
        str_annotation = self.create_name_node(annotation_id, ast.Load(), node)
        iter_assign = ast.AnnAssign(
            target=iter_target,
            annotation=str_annotation,
            value=node.iter,
            simple=1
        )
        self.ensure_all_locations(iter_assign, node)
        return iter_assign

    def _create_index_assignment(self, node):
        """Create ESBMC_index assignment."""
        index_target = self.create_name_node('ESBMC_index', ast.Store(), node)
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

    def _create_length_assignment(self, node, iter_var_name):
        """Create ESBMC_length assignment."""
        length_target = self.create_name_node('ESBMC_length', ast.Store(), node)
        len_func = self.create_name_node('len', ast.Load(), node)
        iter_arg = self.create_name_node(iter_var_name, ast.Load(), node)
        len_call = ast.Call(func=len_func, args=[iter_arg], keywords=[])
        self.ensure_all_locations(len_call, node)
        int_annotation = self.create_name_node('int', ast.Load(), node)
        length_assign = ast.AnnAssign(
            target=length_target,
            annotation=int_annotation,
            value=len_call,
            simple=1
        )
        self.ensure_all_locations(length_assign, node)
        return length_assign

    def _create_while_condition(self, node):
        """Create while loop condition."""
        index_left = self.create_name_node('ESBMC_index', ast.Load(), node)
        length_right = self.create_name_node('ESBMC_length', ast.Load(), node)
        lt_op = ast.Lt()
        self.ensure_all_locations(lt_op, node)
        while_cond = ast.Compare(left=index_left, ops=[lt_op], comparators=[length_right])
        self.ensure_all_locations(while_cond, node)
        return while_cond

    def _create_loop_body(self, node, target_var_name, iter_var_name, annotation_id):
        """Create the complete loop body."""
        # Item assignment
        item_assign = self._create_item_assignment(node, target_var_name, iter_var_name, annotation_id)

        # Index increment
        index_increment = self._create_index_increment(node)

        # Transform original body
        transformed_original_body = []
        for statement in node.body:
            transformed_statement = self.visit(statement)
            if isinstance(transformed_statement, list):
                transformed_original_body.extend(transformed_statement)
            else:
                transformed_original_body.append(transformed_statement)

        return [item_assign, index_increment] + transformed_original_body

    def _create_item_assignment(self, node, target_var_name, iter_var_name, annotation_id):
        """Create assignment to get current item from iterable."""
        item_target = self.create_name_node(target_var_name, ast.Store(), node)
        iter_value = self.create_name_node(iter_var_name, ast.Load(), node)
        index_slice = self.create_name_node('ESBMC_index', ast.Load(), node)
        subscript = ast.Subscript(value=iter_value, slice=index_slice, ctx=ast.Load())
        self.ensure_all_locations(subscript, node)
        element_type = self._get_element_type_from_container(annotation_id)
        item_annotation = self.create_name_node(element_type, ast.Load(), node)
        item_assign = ast.AnnAssign(
            target=item_target,
            annotation=item_annotation,
            value=subscript,
            simple=1
        )
        self.ensure_all_locations(item_assign, node)
        return item_assign

    def _create_index_increment(self, node):
        """Create index increment statement."""
        inc_target = self.create_name_node('ESBMC_index', ast.Store(), node)
        inc_left = self.create_name_node('ESBMC_index', ast.Load(), node)
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

        # Handle constant values
        if isinstance(value, ast.Constant):
            return self._infer_type_from_constant(value)

        # Handle legacy AST nodes (older Python versions)
        if isinstance(value, (ast.Str, ast.Num)):
            return self._infer_type_from_legacy_node(value)

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

    def _infer_type_from_legacy_node(self, node):
        """Infer type from legacy AST nodes (ast.Str, ast.Num)"""
        if isinstance(node, ast.Str):
            return 'str'
        elif isinstance(node, ast.Num):
            return 'int' if isinstance(node.n, int) else 'float'
        return 'Any'

    def _infer_type_from_call(self, call_node):
        """Infer type from function call nodes"""
        if not isinstance(call_node.func, ast.Name):
            return 'Any'

        call_type_map = {
            'range': 'range',
            'list': 'list',
            'dict': 'dict',
            'set': 'set',
            'tuple': 'tuple'
        }

        func_name = call_node.func.id
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
            # For all other cases (including lists and complex expressions),
            # use temporary variable to avoid AST node sharing issues
            temp_var_name = f"ESBMC_unpack_temp_{id(source_node)}"
            temp_var = ast.Name(id=temp_var_name, ctx=ast.Store())
            self._copy_location_info(source_node, temp_var)
            temp_assign = self._create_individual_assignment(temp_var, value, source_node)
            assignments.append(temp_assign)

            # Now create individual assignments from temp variable
            for i, target_elem in enumerate(target.elts):
                if isinstance(target_elem, ast.Name):
                    subscript = ast.Subscript(
                        value=ast.Name(id=temp_var_name, ctx=ast.Load()),
                        slice=ast.Constant(value=i),
                        ctx=ast.Load()
                    )
                    self._copy_location_info(source_node, subscript)
                    self._copy_location_info(source_node, subscript.value)
                    self._copy_location_info(source_node, subscript.slice)

                    individual_assign = self._create_individual_assignment(target_elem, subscript, source_node)
                    self.known_variable_types[target_elem.id] = 'Any'
                    assignments.append(individual_assign)

        return assignments

    def visit_Assign(self, node):
        """
        Handle assignment nodes, including multiple assignments and tuple unpacking.
        """
        # First visit child nodes
        self.generic_visit(node)

        # Handle single target (most common case)
        if len(node.targets) == 1:
            target = node.targets[0]

            # Check if this is tuple unpacking (x, y = ...)
            if isinstance(target, (ast.Tuple, ast.List)):
                return self._handle_tuple_unpacking(target, node.value, node)
            else:
                # Simple assignment - just track the type
                self._update_variable_types_simple(target, node.value)
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


    # This method is responsible for visiting and transforming Call nodes in the AST.
    def visit_Call(self, node):
        # Transformation for int.from_bytes calls
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "int" and node.func.attr == "from_bytes":
            # Replace 'big' argument with True and anything else with False
            if len(node.args) > 1 and isinstance(node.args[1], ast.Str) and node.args[1].s == 'big':
                node.args[1] = ast.NameConstant(value=True)
            else:
                node.args[1] = ast.NameConstant(value=False)

        # if not a function or preprocessor doesn't have function definition return
        if not isinstance(node.func,ast.Name) or node.func.id not in self.functionParams:
            self.generic_visit(node)
            return node

        functionName = node.func.id
        expectedArgs = self.functionParams[functionName]
        keywords = {}
        # add keyword arguments to function call
        for i in node.keywords:
            if i.arg in keywords:
                raise SyntaxError(f"Keyword argument repeated:{i.arg}",(self.module_name,i.lineno,i.col_offset,""))
            keywords[i.arg] = i.value

        # return early if correct no. or too many parameters
        if len(node.args) >= len(expectedArgs):
            self.generic_visit(node)
            return node

        # append defaults
        for i in range(len(node.args),len(expectedArgs)):
            if expectedArgs[i] in keywords:
                node.args.append(keywords[expectedArgs[i]])
            elif (functionName, expectedArgs[i]) in self.functionDefaults:
                default_val = self.functionDefaults[(functionName, expectedArgs[i])]
                if isinstance(default_val,ast.Name):
                    node.args.append(default_val)
                else:
                    node.args.append(ast.Constant(value = default_val))
            else:
                print(f"WARNING: {functionName}() missing required positional argument: '{expectedArgs[i]}'\n")
                print(f"* file: {self.module_name}\n* line {node.lineno}\n* function: {functionName}\n* column: {node.col_offset} ")
                break # breaking means not enough arguments, solver should reject


        self.generic_visit(node)
        return node # transformed node

    def visit_FunctionDef(self, node):
        # Extract parameter type annotations and store them
        for arg in node.args.args:
            if arg.annotation is not None:
                param_type = self._extract_type_from_annotation(arg.annotation)
                self.known_variable_types[arg.arg] = param_type

        # Preserve order of parameters
        self.functionParams[node.name] = [i.arg for i in node.args.args]

        # escape early if no defaults defined
        if len(node.args.defaults) < 1:
            self.generic_visit(node)
            return node
        return_nodes = []

        # add defaults to dictionary with tuple key (function name, parameter name)
        for i in range(1, len(node.args.defaults) + 1):
            # Check bounds before accessing args array
            arg_index = len(node.args.args) - i
            if arg_index >= 0:
                if isinstance(node.args.defaults[-i],ast.Constant):
                    self.functionDefaults[(node.name, node.args.args[-i].arg)] = node.args.defaults[-i].value
                elif isinstance(node.args.defaults[-i],ast.Name):
                    assignment_node, target_var = self.generate_variable_copy(node.name,node.args.args[-i],node.args.defaults[-i])
                    self.functionDefaults[(node.name, node.args.args[-i].arg)] = target_var
                    return_nodes.append(assignment_node)

        self.generic_visit(node)
        return_nodes.append(node)
        return return_nodes

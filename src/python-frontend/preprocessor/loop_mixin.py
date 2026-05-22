"""LoopMixin - extracted from preprocessor.

Contains loop/iteration lowering and related helpers:
range/enumerate/items/reversed(range), iterable while-lowering,
heterogeneous dict iteration unrolling, and defaultdict read lowering.

All shared state lives on Preprocessor (set in Preprocessor.__init__);
this mixin only adds methods.
"""
import ast
import copy
from typing import Dict, Optional, Set

__all__ = ["LoopMixin"]


class LoopMixin:
    """Loop/iteration lowering helpers mixed into `Preprocessor`."""
    # Shared state provided by Preprocessor.__init__. These annotations make
    # the mixin contract explicit and improve static readability.
    variable_annotations: Dict[str, ast.AST]
    known_variable_types: Dict[str, str]
    class_attr_annotations: Dict[str, Dict[str, ast.AST]]
    function_return_annotations: Dict[str, ast.AST]
    instance_class_map: Dict[str, str]
    het_dict_literals: Dict[str, ast.Dict]
    het_value_dict_literals: Dict[str, ast.Dict]
    dict_items_vars: Dict[str, ast.AST]
    _defaultdict_factory: Dict[str, ast.AST]
    _with_counter: int
    _unroll_counter: int
    enumerate_loop_counter: int
    range_loop_counter: int
    iterable_loop_counter: int
    nondet_expand_counter: int
    target_name: str
    module_name: str
    dataclasses_module_names: Set[str]

    @staticmethod
    def _name_id_or_none(node: ast.AST) -> Optional[str]:
        """Return `node.id` when node is ast.Name, else None."""
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _pre_annotate_items_loop_vars(self, node):
        """Pre-populate variable_annotations for the loop variables of a dict.items() for loop.

        Called before generic_visit so that nested inner loops can look up
        the type of the outer loop's value variable (e.g. 'inner' for
        dict[str, dict[str, int]]) and resolve their own K/V types correctly.
        """
        dict_expr = node.iter.func.value
        if isinstance(dict_expr, ast.Name):
            key_ann, val_ann = self._get_dict_kv_types(dict_expr.id)
        elif isinstance(dict_expr, ast.Attribute):
            key_ann, val_ann = self._get_kv_types_from_attribute(dict_expr)
        elif isinstance(dict_expr, ast.Subscript):
            key_ann, val_ann = self._get_kv_types_from_subscript(dict_expr)
        else:
            key_ann, val_ann = self._get_kv_types_from_call(dict_expr)

        target = node.target
        if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2:
            k_var, v_var = target.elts[0], target.elts[1]
            # If the key type is still unknown, check the loop body for
            # some_dict[key_var] usage patterns: using a variable as a dict
            # subscript key implies it is a str (the common dict key type).
            if (isinstance(key_ann, ast.Name) and key_ann.id == "Any"
                    and isinstance(k_var, ast.Name)
                    and self._key_used_as_subscript(k_var.id, node.body)):
                key_ann = ast.Name(id="str", ctx=ast.Load())
            # If the value type is still unknown, check the loop body for
            # val["key"] usage patterns: string subscripts imply a dict value.
            if (isinstance(val_ann, ast.Name) and val_ann.id == "Any"
                    and isinstance(v_var, ast.Name)
                    and self._uses_string_subscript(v_var.id, node.body)):
                val_ann = ast.Name(id="dict", ctx=ast.Load())
            if isinstance(k_var, ast.Name):
                self.variable_annotations[k_var.id] = key_ann
            if isinstance(v_var, ast.Name):
                self.variable_annotations[v_var.id] = val_ann
        else:
            # d.items() yields (key, value) tuples regardless of unpacking
            target_name = self._name_id_or_none(target)
            if target_name is not None:
                self.variable_annotations[target_name] = ast.Name(id="tuple", ctx=ast.Load())

    def _pre_annotate_enumerate_loop_vars(self, node):
        """Pre-populate variable_annotations for enumerate() loop value variable.

        Called before generic_visit so that inner expressions (e.g.
        tuple(sorted([elem, elem2]))) can infer the element type from the loop
        variable when the iterable has a known generic annotation like List[float].
        """
        if not self._is_enumerate_preannotation_candidate(node):
            return

        iterable = node.iter.args[0]
        annotation_id = self._get_iterable_type_annotation(iterable)
        element_type = self._get_element_type_from_container(annotation_id, iterable)
        if element_type and element_type != "Any":
            value_elt = node.target.elts[1]
            if isinstance(value_elt, ast.Name):
                ann_node = ast.Name(id=element_type, ctx=ast.Load())
                self.variable_annotations[value_elt.id] = ann_node
                self.known_variable_types[value_elt.id] = element_type

    @staticmethod
    def _is_enumerate_preannotation_candidate(node):
        """Return True when node matches `for i, v in enumerate(iterable, ...)`."""
        if not isinstance(node.iter, ast.Call):
            return False
        if not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "enumerate":
            return False
        if len(node.iter.args) < 1:
            return False
        if not isinstance(node.target, (ast.Tuple, ast.List)):
            return False
        return len(node.target.elts) >= 2

    def _is_reversed_range_call(self, iter_node):
        """Return True if iter_node is reversed(range(...))."""
        return (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
                and iter_node.func.id == "reversed" and len(iter_node.args) == 1
                and not iter_node.keywords and isinstance(iter_node.args[0], ast.Call)
                and isinstance(iter_node.args[0].func, ast.Name)
                and iter_node.args[0].func.id == "range")

    def _transform_reversed_range(self, reversed_call):
        """
        Transform reversed(range(args)) into an equivalent range(new_args) call.

        Python semantics:
          reversed(range(n))             → range(n-1, -1, -1)
          reversed(range(start, stop))   → range(stop-1, start-1, -1)
          reversed(range(start, stop, step))
            → range(ESBMC_reversed_range_start_(start, stop, step),
                    start-step, -step)

        The helper function computes the last element of the original range
        (or start-step for an empty range, keeping the reversed range empty).
        All divisions inside the helper use same-sign operands, so C and
        Python floor-division agree without any adjustment.
        """
        range_call = reversed_call.args[0]
        args = range_call.args

        if len(args) == 1:
            n = args[0]
            new_args = [
                ast.BinOp(left=n, op=ast.Sub(), right=ast.Constant(value=1)),
                ast.Constant(value=-1),
                ast.Constant(value=-1),
            ]
        elif len(args) == 2:
            start, stop = args
            new_args = [
                ast.BinOp(left=stop, op=ast.Sub(), right=ast.Constant(value=1)),
                ast.BinOp(left=start, op=ast.Sub(), right=ast.Constant(value=1)),
                ast.Constant(value=-1),
            ]
        elif len(args) == 3:
            start, stop, step = args
            # new_start = ESBMC_reversed_range_start_(start, stop, step)
            # new_stop  = start - step
            # new_step  = -step
            #
            # The helper function correctly computes the last element of
            # range(start, stop, step) (or start-step for an empty range,
            # which makes the caller's reversed range trivially empty too).
            # It avoids mixed-sign floor-division so C and Python agree.
            new_start = ast.Call(
                func=ast.Name(id="ESBMC_reversed_range_start_", ctx=ast.Load()),
                args=[copy.deepcopy(start),
                      copy.deepcopy(stop),
                      copy.deepcopy(step)],
                keywords=[],
            )
            new_stop = ast.BinOp(left=copy.deepcopy(start), op=ast.Sub(), right=copy.deepcopy(step))
            # Constant-fold -step so that step==0 remains an ast.Constant and
            # _transform_range_for's compile-time ValueError check still fires.
            if isinstance(step, ast.Constant):
                new_step = ast.Constant(value=-step.value)
            else:
                new_step = ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(step))
            new_args = [new_start, new_stop, new_step]
        else:
            # Invalid number of range args — let the existing validator raise.
            return reversed_call

        new_range = ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=new_args,
            keywords=[],
        )
        ast.copy_location(new_range, reversed_call)
        ast.fix_missing_locations(new_range)
        return new_range

    def visit_For(self, node):  # pylint: disable=too-many-branches
        """
        Transform for loops into while loops.
        Handles range() calls, enumerate() calls, dict.items(), and general iterables.
        """
        # Rewrite reversed(range(...)) to an equivalent range(...) call so that
        # the normal range-loop path can handle it without any extra machinery.
        if self._is_reversed_range_call(node.iter):
            node.iter = self._transform_reversed_range(node.iter)

        # Detect range call before generic_visit so we can hoist generator
        # outer_init (e.g. `i = 0`) before the loop.  Without hoisting, the
        # init ends up inside the while body and re-runs every iteration.
        is_range_call = (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                         and node.iter.func.id == "range")

        gen_pre_stmts = []
        if is_range_call:
            gen_pre_stmts = self._hoist_generator_inits(node.body, node)

        # Pre-populate variable_annotations for items() loop variables before
        # generic_visit, so that inner loops can resolve the type of outer loop
        # variables (e.g. 'inner: dict[str, int]') when they are visited.
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr == "items"):
            self._pre_annotate_items_loop_vars(node)

        # Pre-populate variable_annotations for enumerate() loop value variable
        # before generic_visit, so that inner expressions can infer the element
        # type from the loop variable (e.g. elem: float when numbers: List[float]).
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "enumerate"
                and isinstance(node.target, (ast.Tuple, ast.List)) and len(node.target.elts) == 2):
            self._pre_annotate_enumerate_loop_vars(node)

        # First, recursively visit any nested nodes
        node = self.generic_visit(node)

        # Check if iter is a Call to enumerate
        is_enumerate_call = (isinstance(node.iter, ast.Call)
                             and isinstance(node.iter.func, ast.Name)
                             and node.iter.func.id == "enumerate")

        # Check if iter is a Call to dict.items()
        is_items_call = (isinstance(node.iter, ast.Call)
                         and isinstance(node.iter.func, ast.Attribute)
                         and node.iter.func.attr == "items")

        if is_range_call:
            # Handle range-based for loops
            self.is_range_loop = True
            self.helper_functions_added = True  # Mark that we need helper functions
            result = self._transform_range_for(node)
            self.is_range_loop = False
            return gen_pre_stmts + result
        if is_enumerate_call:
            # Handle enumerate-based for loops
            self.is_range_loop = False
            return self._transform_enumerate_for(node)
        if is_items_call:
            # Handle dict.items() for loops
            self.is_range_loop = False
            return self._transform_items_for(node)
        # zip(), reversed(<non-range>), and filter() for-loop iteration.
        # reversed(range(...)) was already rewritten to range(...) above, so
        # _is_reversed_call here only matches reversed() over other sequences.
        if self._is_zip_call(node.iter):
            self.is_range_loop = False
            return self._transform_zip_for(node)
        if self._is_reversed_call(node.iter):
            self.is_range_loop = False
            return self._transform_reversed_for(node)
        if self._is_filter_call(node.iter):
            self.is_range_loop = False
            return self._transform_filter_for(node)
        list_literal = self.list_literal_values.get(node.iter.id) if isinstance(
            node.iter, ast.Name) else None
        if (list_literal is not None
                and self._can_safely_unroll_list_literal_for(node, list_literal)):
            # For direct iteration over a known list literal variable, unroll the loop
            # to avoid introducing len()/index machinery in the generated model.
            # Skip the unroll if the body contains break/continue/return, since
            # straight-line unrolling would leave those statements without a
            # surrounding loop/function context. Skip too when elements are not
            # homogeneous pure literals to preserve runtime isinstance semantics.
            self.is_range_loop = False
            return self._unroll_list_literal_for(node, list_literal)
        # Check if iterating over a generator variable
        if isinstance(node.iter, ast.Name) and node.iter.id in self.generator_vars:
            inlined = self._inline_generator_for(node)
            if inlined is not None:
                return inlined
        # Check if iterating directly over a generator function call, e.g.
        # `for y in gen1(arr): body`.  Without this, _transform_iterable_for
        # would emit `ESBMC_iter: list = gen1(arr)` which assigns a generator
        # object to a list variable — ESBMC cannot model generator objects.
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id in self.generator_funcs):
            inlined = self._inline_generator_call_for(node)
            if inlined is not None:
                return inlined
        # Unwrap explicit d.keys() into d so the heterogeneous-key handler
        # below can pick it up.  `for k in d.keys()` is semantically
        # identical to `for k in d` and must be treated the same way.
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr == "keys" and isinstance(node.iter.func.value, ast.Name)
                and node.iter.func.value.id in self.het_dict_literals):
            node.iter = node.iter.func.value
        # Unroll iteration over dict literals with heterogeneous key types.
        if isinstance(node.iter, ast.Name) and node.iter.id in self.het_dict_literals:
            return self._transform_het_dict_for(node)
        # Unroll d.values() when the dict has heterogeneous value types.
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr == "values" and isinstance(node.iter.func.value, ast.Name)
                and node.iter.func.value.id in self.het_value_dict_literals):
            dict_node = self.het_value_dict_literals[node.iter.func.value.id]
            return self._transform_het_values_for(node, dict_node)
        # Handle general iteration over iterables (strings, lists, etc.)
        self.is_range_loop = False
        return self._transform_iterable_for(node)

    def _can_safely_unroll_list_literal_for(self, node, list_literal):
        """Decide whether a `for` over a tracked list literal is safe to unroll.

        Skip the unroll when:
          * the loop body contains ``break``/``continue``/``return`` (these
            need a surrounding loop/function context);
          * elements are constants of heterogeneous types (e.g. mixed ``int``
            and ``str``), which would silently drop runtime ``isinstance``
            checks during unrolling and constant folding.
        """
        for stmt in node.body:
            for n in ast.walk(stmt):
                if isinstance(n, (ast.Break, ast.Continue, ast.Return)):
                    return False

        const_types = set()
        all_constants = True
        for elt in list_literal.elts:
            if isinstance(elt, ast.Constant):
                const_types.add(type(elt.value).__name__)
            elif (isinstance(elt, ast.UnaryOp) and isinstance(elt.op, (ast.UAdd, ast.USub))
                  and isinstance(elt.operand, ast.Constant)):
                const_types.add(type(elt.operand.value).__name__)
            else:
                all_constants = False
                break
        if all_constants and len(const_types) > 1:
            return False
        return True

    def _unroll_list_literal_for(self, node, list_literal):
        """Unroll `for` over a tracked list literal variable into straight-line code.

        For ``Name`` loop targets, snapshots each list element into a
        per-iteration temp *before* emitting the unrolled body. This preserves
        Python's "list elements are evaluated once at list construction"
        semantics: when the body mutates a name that also appears among the
        list elements (e.g. ``xs = [a, a]; for x in xs: a = ...``), later
        iterations still see the original value via the temp instead of
        re-reading the now-mutated source name.

        For tuple/list unpacking targets (``for a, b in pairs:``), the snapshot
        path is skipped because the converter's tuple-unpacking pipeline
        requires the RHS to be a tuple/list literal — not a symbol load — and
        tuple-literal elements rarely depend on body-mutated names in practice.
        """
        unrolled = []
        counter = self._unroll_counter
        self._unroll_counter += 1
        target_is_name = isinstance(node.target, ast.Name)

        # Snapshot phase (Name targets only): evaluate each element once into
        # a fresh temp so subsequent body mutations cannot retroactively
        # change values seen by later iterations.
        temp_names = []
        if target_is_name:
            for idx, elt in enumerate(list_literal.elts):
                temp_name = f"__esbmc_unrolled_item_{counter}_{idx}"
                temp_names.append(temp_name)
                snap_assign = ast.Assign(
                    targets=[ast.Name(id=temp_name, ctx=ast.Store())],
                    value=copy.deepcopy(elt),
                )
                self.ensure_all_locations(snap_assign, node)
                unrolled.append(snap_assign)

        # Iteration phase: bind the loop target from each snapshot temp (or
        # inline the elt for tuple/list unpacking) and emit the original body
        # once per element.
        for idx, elt in enumerate(list_literal.elts):
            if target_is_name:
                rhs = ast.Name(id=temp_names[idx], ctx=ast.Load())
                self.ensure_all_locations(rhs, node)
                target_assign = ast.Assign(
                    targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
                    value=rhs,
                )
            else:
                # Tuple/list unpacking: keep the RHS as the original literal so
                # the converter's tuple-unpacking path can still extract elts.
                target_assign = ast.Assign(
                    targets=[copy.deepcopy(node.target)],
                    value=copy.deepcopy(elt),
                )
            self.ensure_all_locations(target_assign, node)
            unrolled.append(target_assign)

            for stmt in node.body:
                stmt_copy = copy.deepcopy(stmt)
                self.ensure_all_locations(stmt_copy, node)
                unrolled.append(stmt_copy)

        for stmt in unrolled:
            ast.fix_missing_locations(stmt)
        return unrolled

    def visit_With(self, node):  # pylint: disable=too-many-locals
        """Desugar 'with EXPR as VAR: BODY' into __enter__/__exit__ calls.

        Transforms each context manager item into:
            __esbmc_mgr_N = EXPR              # annotated if class type is known
            VAR = __esbmc_mgr_N.__enter__()   # omitted when there is no 'as' clause
            BODY
            __esbmc_mgr_N.__exit__(0, 0, 0)   # non-exceptional path; zeros for int args

        Multiple items are expanded left-to-right; __exit__ is called in reverse order.
        AsyncWith is handled identically via the class-level alias below.
        """
        node = self.generic_visit(node)
        result = []
        exit_start = self._with_counter

        for item in node.items:
            mgr_name = f"__esbmc_mgr_{self._with_counter}"
            self._with_counter += 1
            ctx_expr = item.context_expr

            if isinstance(ctx_expr, ast.Call) and isinstance(ctx_expr.func, ast.Name):
                class_name = ctx_expr.func.id
                type_ann = ast.Name(id=class_name, ctx=ast.Load())
                mgr_assign = ast.AnnAssign(
                    target=ast.Name(id=mgr_name, ctx=ast.Store()),
                    annotation=type_ann,
                    value=ctx_expr,
                    simple=1,
                )
                self.variable_annotations[mgr_name] = type_ann
                self.instance_class_map[mgr_name] = class_name
            else:
                mgr_assign = ast.Assign(
                    targets=[ast.Name(id=mgr_name, ctx=ast.Store())],
                    value=ctx_expr,
                )
            result.append(self.ensure_all_locations(mgr_assign, node))

            enter_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=mgr_name, ctx=ast.Load()),
                    attr="__enter__",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
            if item.optional_vars is not None:
                result.append(
                    self.ensure_all_locations(
                        ast.Assign(targets=[item.optional_vars], value=enter_call), node))
            else:
                result.append(self.ensure_all_locations(ast.Expr(value=enter_call), node))

        # Build __exit__ calls in reverse order.  The helper is factored out so
        # the same call shape can be re-instantiated for both the success path
        # (statement) and the exception handler (operand of `not`); AST nodes
        # must not be shared across locations because each carries its own
        # location/parent metadata.
        def make_exit_call(i):
            mgr_name = f"__esbmc_mgr_{exit_start + i}"
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=mgr_name, ctx=ast.Load()),
                    attr="__exit__",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=0)] * 3,
                keywords=[],
            )

        exit_calls = [
            self.ensure_all_locations(ast.Expr(value=make_exit_call(i)), node)
            for i in range(len(node.items) - 1, -1, -1)
        ]

        # When every manager's class defines (or inherits) __exit__, lower to
        # CPython's dynamic-suppression form:
        #   try:
        #       BODY
        #       <success-path exit calls>
        #   except BaseException:
        #       if not mgr.__exit__(0, 0, 0): raise
        #       ...                          # one guard per manager, reverse order
        # __exit__'s return value is consulted at runtime: truthy suppresses,
        # falsy re-raises via bare `raise`.  Managers without a tracked class
        # (e.g. `open(...)`) fall back to today's unwrapped lowering.
        wrap = (hasattr(self, "_classes_with_exit") and len(node.items) > 0 and all(
            isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id in self._classes_with_exit for item in node.items))

        if wrap:
            handler_body = [
                ast.If(
                    test=ast.UnaryOp(op=ast.Not(), operand=make_exit_call(i)),
                    body=[ast.Raise(exc=None, cause=None)],
                    orelse=[],
                ) for i in range(len(node.items) - 1, -1, -1)
            ]
            try_node = ast.Try(
                body=list(node.body) + exit_calls,
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id="BaseException", ctx=ast.Load()),
                        name=None,
                        body=handler_body,
                    )
                ],
                orelse=[],
                finalbody=[],
            )
            self.ensure_all_locations(try_node, node)
            ast.fix_missing_locations(try_node)
            result.append(try_node)
        else:
            result.extend(node.body)
            result.extend(exit_calls)

        return result

    visit_AsyncWith = visit_With  # noqa: N815

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
        setup_statements = self._create_enumerate_setup_statements(node, iterable, start_value,
                                                                   loop_id)

        # Step 5: Create the while loop
        while_stmt = self._create_enumerate_while_loop(node, target_info, loop_id)

        # Step 6: Combine everything and ensure proper AST locations
        result = setup_statements + [while_stmt]
        for stmt in result:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        return result

    def _validate_enumerate_call(self, enumerate_call):
        """Validate enumerate() call arguments."""
        if not enumerate_call.args:
            raise TypeError("enumerate() missing required argument 'iterable' (pos 1)")
        if len(enumerate_call.args) > 2:
            raise TypeError(
                f"enumerate() takes at most 2 arguments ({len(enumerate_call.args)} given)")
        for kw in getattr(enumerate_call, "keywords", []) or []:
            if kw.arg is None:
                raise TypeError("enumerate() does not accept **kwargs")
            if kw.arg != "start":
                raise TypeError(f"enumerate() got an unexpected keyword argument '{kw.arg}'")
        if (len(enumerate_call.args) == 2
                and any(kw.arg == "start" for kw in (enumerate_call.keywords or []))):
            raise TypeError("enumerate() got multiple values for argument 'start'")

    def _parse_enumerate_target(self, target):
        """Parse and validate the for loop target, return target information."""
        # Check if this is tuple/list unpacking or single variable assignment
        is_unpacking = (isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2)

        if is_unpacking:
            if not all(isinstance(elt, ast.Name) for elt in target.elts):
                raise ValueError("enumerate unpacking target must contain only names")
            return {
                "type": "unpacking",
                "index_var": target.elts[0].id,
                "value_var": target.elts[1].id,
            }
        if isinstance(target, ast.Name):
            return {"type": "single", "var_name": target.id}
        # Handle error cases
        if isinstance(target, (ast.Tuple, ast.List)):
            expected = len(target.elts)
            if expected > 2:
                raise ValueError(f"not enough values to unpack (expected {expected}, got 2)")
            if expected < 2:
                raise ValueError(f"too many values to unpack (expected {expected})")
        raise ValueError("enumerate target must be a name, tuple, or list")

    def _parse_enumerate_arguments(self, enumerate_call, node):
        """Extract and validate iterable and start value from enumerate call."""
        iterable = enumerate_call.args[0]

        start_value = None
        if len(enumerate_call.args) > 1:
            start_value = enumerate_call.args[1]
        else:
            for kw in (enumerate_call.keywords or []):
                if kw.arg == "start":
                    start_value = kw.value
                    break

        if start_value is None:
            start_value = self.create_constant_node(0, node)
        else:
            self._validate_start_value(start_value)

        return iterable, start_value

    def _validate_start_value(self, start_value):
        """Validate that the start value is an integer (matching Python's behavior)."""
        if isinstance(start_value, ast.Constant):
            start_val = start_value.value
            if isinstance(start_val, bool):
                # Python accepts bool since bool is a subclass of int.
                return
            if isinstance(start_val, (float, str)):
                type_name = type(start_val).__name__
                raise TypeError(f"'{type_name}' object cannot be interpreted as an integer")
            if not isinstance(start_val, int):
                type_name = type(start_val).__name__
                raise TypeError(f"'{type_name}' object cannot be interpreted as an integer")

    def _create_enumerate_setup_statements(self, node, iterable, start_value, loop_id):
        """Create the initial variable assignments for enumerate transformation."""
        annotation_id = self._get_iterable_type_annotation(iterable)

        iter_var = f"ESBMC_iter_{loop_id}"
        index_var = f"ESBMC_index_{loop_id}"
        array_index_var = f"ESBMC_array_index_{loop_id}"
        length_var = f"ESBMC_length_{loop_id}"

        # Create: ESBMC_iter: <type> = iterable
        iter_assign = ast.AnnAssign(
            target=self.create_name_node(iter_var, ast.Store(), node),
            # annotation=annotation_node,
            annotation=self.create_name_node(annotation_id, ast.Load(), node),
            value=iterable,
            simple=1,
        )
        self.ensure_all_locations(iter_assign, node)

        # Create: ESBMC_index: int = start_value (enumeration index)
        index_assign = ast.AnnAssign(
            target=self.create_name_node(index_var, ast.Store(), node),
            annotation=self.create_name_node("int", ast.Load(), node),
            value=start_value,
            simple=1,
        )
        self.ensure_all_locations(index_assign, node)

        # Create: ESBMC_array_index: int = 0 (array access index)
        array_index_assign = ast.AnnAssign(
            target=self.create_name_node(array_index_var, ast.Store(), node),
            annotation=self.create_name_node("int", ast.Load(), node),
            value=self.create_constant_node(0, node),
            simple=1,
        )
        self.ensure_all_locations(array_index_assign, node)

        # Create: ESBMC_length: int = len(ESBMC_iter)
        len_call = ast.Call(
            func=self.create_name_node("len", ast.Load(), node),
            args=[self.create_name_node(iter_var, ast.Load(), node)],
            keywords=[],
        )
        self.ensure_all_locations(len_call, node)
        length_assign = ast.AnnAssign(
            target=self.create_name_node(length_var, ast.Store(), node),
            annotation=self.create_name_node("int", ast.Load(), node),
            value=len_call,
            simple=1,
        )
        self.ensure_all_locations(length_assign, node)

        return [iter_assign, index_assign, array_index_assign, length_assign]

    def _create_enumerate_while_loop(self, node, target_info, loop_id):
        """Create the while loop for enumerate transformation."""
        array_index_var = f"ESBMC_array_index_{loop_id}"
        length_var = f"ESBMC_length_{loop_id}"

        # Create while condition: ESBMC_array_index < ESBMC_length
        while_cond = ast.Compare(
            left=self.create_name_node(array_index_var, ast.Load(), node),
            ops=[ast.Lt()],
            comparators=[self.create_name_node(length_var, ast.Load(), node)],
        )
        self.ensure_all_locations(while_cond, node)

        # Create loop body based on target type
        if target_info["type"] == "unpacking":
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
        iterable_node = node.iter.args[0] if hasattr(node.iter, "args") else None
        annotation_id = self._get_iterable_type_annotation(iterable_node)

        iter_var = f"ESBMC_iter_{loop_id}"
        index_var = f"ESBMC_index_{loop_id}"
        array_index_var = f"ESBMC_array_index_{loop_id}"

        # index_var: int = ESBMC_index
        user_index_assign = ast.AnnAssign(
            target=self.create_name_node(target_info["index_var"], ast.Store(), node),
            annotation=self.create_name_node("int", ast.Load(), node),
            value=self.create_name_node(index_var, ast.Load(), node),
            simple=1,
        )
        self.ensure_all_locations(user_index_assign, node)

        # value_var: <element_type> = ESBMC_iter[ESBMC_array_index]
        subscript = ast.Subscript(
            value=self.create_name_node(iter_var, ast.Load(), node),
            slice=self.create_name_node(array_index_var, ast.Load(), node),
            ctx=ast.Load(),
        )
        self.ensure_all_locations(subscript, node)

        element_type = self._get_element_type_from_container(annotation_id, iterable_node)
        ann_node = self.create_name_node(element_type, ast.Load(), node)
        user_value_assign = ast.AnnAssign(
            target=self.create_name_node(target_info["value_var"], ast.Store(), node),
            annotation=ann_node,
            value=subscript,
            simple=1,
        )
        self.ensure_all_locations(user_value_assign, node)
        # Propagate type so downstream visitors (e.g. _lower_tuple_sorted_pair_call)
        # can infer the scalar type of variables derived from this loop variable.
        self.variable_annotations[target_info["value_var"]] = ann_node
        self.known_variable_types[target_info["value_var"]] = element_type

        return [user_index_assign, user_value_assign]

    def _create_single_var_loop_body(self, node, target_info, loop_id):
        """Create loop body for single variable case: for item in enumerate(...)"""
        iter_var = f"ESBMC_iter_{loop_id}"
        index_var = f"ESBMC_index_{loop_id}"
        array_index_var = f"ESBMC_array_index_{loop_id}"

        # Create tuple: (ESBMC_index, ESBMC_iter[ESBMC_array_index])
        subscript = ast.Subscript(
            value=self.create_name_node(iter_var, ast.Load(), node),
            slice=self.create_name_node(array_index_var, ast.Load(), node),
            ctx=ast.Load(),
        )
        self.ensure_all_locations(subscript, node)

        tuple_value = ast.Tuple(
            elts=[self.create_name_node(index_var, ast.Load(), node), subscript],
            ctx=ast.Load(),
        )
        self.ensure_all_locations(tuple_value, node)

        # single_var: tuple = (ESBMC_index, ESBMC_iter[ESBMC_array_index])
        user_tuple_assign = ast.AnnAssign(
            target=self.create_name_node(target_info["var_name"], ast.Store(), node),
            annotation=self.create_name_node("tuple", ast.Load(), node),
            value=tuple_value,
            simple=1,
        )
        self.ensure_all_locations(user_tuple_assign, node)

        return [user_tuple_assign]

    def _create_increment_statements(self, node, loop_id):
        """Create the increment statements for both indices."""
        index_var = f"ESBMC_index_{loop_id}"
        array_index_var = f"ESBMC_array_index_{loop_id}"

        # ESBMC_index: int = ESBMC_index + 1
        index_increment = ast.AnnAssign(
            target=self.create_name_node(index_var, ast.Store(), node),
            annotation=self.create_name_node("int", ast.Load(), node),
            value=ast.BinOp(
                left=self.create_name_node(index_var, ast.Load(), node),
                op=ast.Add(),
                right=self.create_constant_node(1, node),
            ),
            simple=1,
        )
        self.ensure_all_locations(index_increment, node)

        # ESBMC_array_index: int = ESBMC_array_index + 1
        array_index_increment = ast.AnnAssign(
            target=self.create_name_node(array_index_var, ast.Store(), node),
            annotation=self.create_name_node("int", ast.Load(), node),
            value=ast.BinOp(
                left=self.create_name_node(array_index_var, ast.Load(), node),
                op=ast.Add(),
                right=self.create_constant_node(1, node),
            ),
            simple=1,
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

    def _transform_range_for(self, node):  # pylint: disable=too-many-locals
        """Transform range-based for loops to while loops"""
        # Add validation for range arguments
        if len(node.iter.args) == 0:
            raise SyntaxError(
                "range expected at least 1 argument, got 0",
                (self.module_name, node.lineno, node.col_offset, ""),
            )
        if len(node.iter.args) > 3:
            raise SyntaxError(
                f"range expected at most 3 arguments, got {len(node.iter.args)}",
                (self.module_name, node.lineno, node.col_offset, ""),
            )
        # Check if step (third argument) is zero
        if len(node.iter.args) == 3:
            step = node.iter.args[2]
            if isinstance(step, ast.Constant) and step.value == 0:
                raise ValueError("range() arg 3 must not be zero")
        # Generate unique variable names for this loop level
        loop_id = self.range_loop_counter
        self.range_loop_counter += 1
        start_var = f"start_{loop_id}"
        has_next_var = f"has_next_{loop_id}"
        start = ast.Constant(value=0)
        end = node.iter.args[0]
        if len(node.iter.args) > 1:
            start = node.iter.args[0]  # Start of the range
            end = node.iter.args[1]  # End of the range

        # Check if step is provided in range, otherwise default to 1
        if len(node.iter.args) > 2:
            step = node.iter.args[2]
        else:
            step = ast.Constant(value=1)

        # Step validation - Python raises ValueError if step == 0
        step_validation = ast.Assert(
            test=ast.Compare(left=step, ops=[ast.NotEq()], comparators=[ast.Constant(value=0)]),
            msg=ast.Constant(value="range() arg 3 must not be zero"),
        )

        # Create assignment for the start variable
        start_assign = ast.AnnAssign(
            target=ast.Name(id=start_var, ctx=ast.Store()),
            annotation=ast.Name(id="int", ctx=ast.Load()),
            value=start,
            simple=1,
        )

        # Create call to ESBMC_range_has_next_ function for the range
        has_next_call = ast.Call(
            func=ast.Name(id="ESBMC_range_has_next_", ctx=ast.Load()),
            args=[start, end, step],
            keywords=[],
        )

        # Create assignment for the has_next variable
        has_next_assign = ast.AnnAssign(
            target=ast.Name(id=has_next_var, ctx=ast.Store()),
            annotation=ast.Name(id="bool", ctx=ast.Load()),
            value=has_next_call,
            simple=1,
        )

        # Create condition for the while loop
        has_next_name = ast.Name(id=has_next_var, ctx=ast.Load())
        while_cond = has_next_name

        # Transform the body of the for loop
        transformed_body = []
        old_target_name = self.target_name
        old_start_var = getattr(self, "current_start_var", None)
        target_name = self._name_id_or_none(node.target)
        if target_name is None:
            raise ValueError("range loop target must be a variable name")
        self.target_name = target_name  # Store the target variable name for replacement
        self.current_start_var = (
            start_var  # Store current start variable for Name replacement
        )

        for statement in node.body:
            transformed_statement = self.visit(statement)
            if isinstance(transformed_statement, list):
                transformed_body.extend(transformed_statement)
            else:
                transformed_body.append(transformed_statement)
        self.target_name = old_target_name
        self.current_start_var = old_start_var

        # Assign loop variable = range counter at the start of each iteration.
        # Use AnnAssign with 'int' so the annotation system knows the type;
        # range() always yields integers.  A plain Assign leaves the loop var
        # unannotated, causing pointer-type mismatches in arithmetic operations.
        loop_var_init = ast.AnnAssign(
            target=ast.Name(id=target_name, ctx=ast.Store()),
            annotation=ast.Name(id="int", ctx=ast.Load()),
            value=ast.Name(id=start_var, ctx=ast.Load()),
            simple=1,
        )
        self.ensure_all_locations(loop_var_init, node)
        ast.fix_missing_locations(loop_var_init)

        # Create the body of the while loop, including updating the start and has_next variables
        while_body = ([loop_var_init] + transformed_body + [
            ast.Assign(
                targets=[ast.Name(id=start_var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="ESBMC_range_next_", ctx=ast.Load()),
                    args=[ast.Name(id=start_var, ctx=ast.Load()), step],
                    keywords=[],
                ),
            ),
            ast.Assign(
                targets=[ast.Name(id=has_next_var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="ESBMC_range_has_next_", ctx=ast.Load()),
                    args=[ast.Name(id=start_var, ctx=ast.Load()), end, step],
                    keywords=[],
                ),
            ),
        ])

        # Create the while statement
        while_stmt = ast.While(test=while_cond, body=while_body, orelse=[])

        # Return the transformed statements
        return [step_validation, start_assign, has_next_assign, while_stmt]

    def _transform_items_for(self, node):  # pylint: disable=too-many-locals,too-many-statements
        """
        Transform dict.items() for loops to while loops.

        Tuple-unpacking form (for k, v in d.items()):
            ESBMC_keys_N: list[key_type] = d.keys()
            ESBMC_vals_N: list[val_type] = d.values()
            ESBMC_index_N: int = 0
            ESBMC_length_N: int = len(ESBMC_keys_N)
            while ESBMC_index_N < ESBMC_length_N:
                k: key_type = ESBMC_keys_N[ESBMC_index_N]
                v: val_type = ESBMC_vals_N[ESBMC_index_N]
                ESBMC_index_N: int = ESBMC_index_N + 1
                # body

        Single-variable form (for item in d.items()):
            ESBMC_keys_N: list[key_type] = d.keys()
            ESBMC_vals_N: list[val_type] = d.values()
            ESBMC_index_N: int = 0
            ESBMC_length_N: int = len(ESBMC_keys_N)
            while ESBMC_index_N < ESBMC_length_N:
                item: tuple = (ESBMC_keys_N[ESBMC_index_N], ESBMC_vals_N[ESBMC_index_N])
                ESBMC_index_N: int = ESBMC_index_N + 1
                # body

        Using intermediate annotated list variables lets the C++ list subscript
        handler resolve element types from the AnnAssign annotation.
        """
        loop_id = self.iterable_loop_counter
        self.iterable_loop_counter += 1

        index_var = f"ESBMC_index_{loop_id}"
        length_var = f"ESBMC_length_{loop_id}"
        keys_var = f"ESBMC_keys_{loop_id}"
        vals_var = f"ESBMC_vals_{loop_id}"

        # Get the dict expression (e.g., 'd' in d.items(), or 'make()' in make().items())
        dict_expr = node.iter.func.value
        setup_stmts = []

        if isinstance(dict_expr, ast.Name):
            # Simple variable: use directly and look up its annotation
            dict_node = dict_expr
            key_ann, val_ann = self._get_dict_kv_types(dict_node.id)
        elif isinstance(dict_expr, ast.Attribute):
            # Attribute access (e.g., c.d.items()): materialize into a temp variable
            # and look up K/V types from the class attribute annotation.
            dict_temp_var = f"ESBMC_dict_{loop_id}"
            dict_node = ast.Name(id=dict_temp_var, ctx=ast.Load())
            self.ensure_all_locations(dict_node, node)
            key_ann, val_ann = self._get_kv_types_from_attribute(dict_expr)
            dict_assign = ast.AnnAssign(
                target=ast.Name(id=dict_temp_var, ctx=ast.Store()),
                annotation=ast.Name(id="dict", ctx=ast.Load()),
                value=dict_expr,
                simple=1,
            )
            self.ensure_all_locations(dict_assign, node)
            setup_stmts.append(dict_assign)
        elif isinstance(dict_expr, ast.Subscript):
            # Subscript access (e.g., d["key"].items()): materialize into a temp
            # variable and infer K/V types from the outer dict's value annotation.
            dict_temp_var = f"ESBMC_dict_{loop_id}"
            dict_node = ast.Name(id=dict_temp_var, ctx=ast.Load())
            self.ensure_all_locations(dict_node, node)
            key_ann, val_ann = self._get_kv_types_from_subscript(dict_expr)
            dict_assign = ast.AnnAssign(
                target=ast.Name(id=dict_temp_var, ctx=ast.Store()),
                annotation=ast.Name(id="dict", ctx=ast.Load()),
                value=dict_expr,
                simple=1,
            )
            self.ensure_all_locations(dict_assign, node)
            setup_stmts.append(dict_assign)
        else:
            # Other complex expression (e.g., a function call: make().items()):
            # materialize into a temp symbol so the C++ converter gets a stable
            # lvalue for member access. Accessing a member of an rvalue crashes ESBMC.
            dict_temp_var = f"ESBMC_dict_{loop_id}"
            dict_node = ast.Name(id=dict_temp_var, ctx=ast.Load())
            self.ensure_all_locations(dict_node, node)
            key_ann, val_ann = self._get_kv_types_from_call(dict_expr)
            dict_assign = ast.AnnAssign(
                target=ast.Name(id=dict_temp_var, ctx=ast.Store()),
                annotation=ast.Name(id="dict", ctx=ast.Load()),
                value=dict_expr,
                simple=1,
            )
            self.ensure_all_locations(dict_assign, node)
            setup_stmts.append(dict_assign)

        # If key or val type is still unknown (Any), scan the loop body for
        # usage patterns that reveal the type.
        _tgt = node.target
        if isinstance(_tgt, (ast.Tuple, ast.List)) and len(_tgt.elts) == 2:
            _k_elt, _v_elt = _tgt.elts[0], _tgt.elts[1]
            # some_dict[key_var] in the body => key is str (common dict key type)
            if (isinstance(key_ann, ast.Name) and key_ann.id == "Any"
                    and isinstance(_k_elt, ast.Name)
                    and self._key_used_as_subscript(_k_elt.id, node.body)):
                key_ann = ast.Name(id="str", ctx=ast.Load())
            # val["str_const"] in the body => value is a dict
            if (isinstance(val_ann, ast.Name) and val_ann.id == "Any"
                    and isinstance(_v_elt, ast.Name)
                    and self._uses_string_subscript(_v_elt.id, node.body)):
                val_ann = ast.Name(id="dict", ctx=ast.Load())

        # Intermediate list variables: ESBMC_keys_N: list[base(K)] = d.keys()
        # The list slice uses the BASE type name only (e.g. 'dict' for dict[str,int])
        # so the C++ list subscript handler can call get_typet("dict") correctly.
        keys_assign = self._create_dict_list_assign(node, keys_var, dict_node, "keys", key_ann)
        vals_assign = self._create_dict_list_assign(node, vals_var, dict_node, "values", val_ann)

        # Setup: index = 0 and length = len(ESBMC_keys_N)
        index_assign = self._create_index_assignment(node, index_var)
        length_assign = self._create_length_assignment(node, keys_var, length_var)

        # While condition: ESBMC_index_N < ESBMC_length_N
        while_cond = self._create_while_condition(node, index_var, length_var)

        # Build loop body
        target = node.target
        body = []
        if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2:
            key_var_name = target.elts[0].id
            val_var_name = target.elts[1].id
            body.append(
                self._create_var_subscript_assign(node, key_var_name, keys_var, index_var, key_ann))
            body.append(
                self._create_var_subscript_assign(node, val_var_name, vals_var, index_var, val_ann))
        else:
            # Single variable: d.items() yields (key, value) tuples per Python semantics.
            single_var = self._name_id_or_none(target) or "ESBMC_loop_var"
            key_subscript = ast.Subscript(
                value=ast.Name(id=keys_var, ctx=ast.Load()),
                slice=ast.Name(id=index_var, ctx=ast.Load()),
                ctx=ast.Load(),
            )
            self.ensure_all_locations(key_subscript, node)
            val_subscript = ast.Subscript(
                value=ast.Name(id=vals_var, ctx=ast.Load()),
                slice=ast.Name(id=index_var, ctx=ast.Load()),
                ctx=ast.Load(),
            )
            self.ensure_all_locations(val_subscript, node)
            tuple_value = ast.Tuple(elts=[key_subscript, val_subscript], ctx=ast.Load())
            self.ensure_all_locations(tuple_value, node)
            tuple_assign = ast.AnnAssign(
                target=ast.Name(id=single_var, ctx=ast.Store()),
                annotation=ast.Name(id="tuple", ctx=ast.Load()),
                value=tuple_value,
                simple=1,
            )
            self.ensure_all_locations(tuple_assign, node)
            body.append(tuple_assign)

        body.append(self._create_index_increment(node, index_var))
        body.extend(node.body)
        # Detect modification of the dict during iteration (Python raises RuntimeError).
        # Since ESBMC_keys_N is a pointer alias to d.keys, list_size(ESBMC_keys_N)
        # reflects any list_push/list_pop done by dict assignment in the loop body.
        body.append(self._create_dict_size_assertion(node, keys_var, length_var))

        while_stmt = ast.While(test=while_cond, body=body, orelse=[])
        self.ensure_all_locations(while_stmt, node)

        result = setup_stmts + [
            keys_assign,
            vals_assign,
            index_assign,
            length_assign,
            while_stmt,
        ]
        for stmt in result:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        return result

    def _any_ann(self):
        """Return a fresh ast.Name(id='Any') annotation node."""
        return ast.Name(id="Any", ctx=ast.Load())

    def _uses_string_subscript(self, var_name, body):
        """Return True if var_name is subscripted with a string constant anywhere in body.

        Used to infer that a loop variable annotated as Any is actually a dict,
        because val["key"] access in Python is only valid on mappings.
        """
        module = ast.Module(body=list(body), type_ignores=[])
        for node in ast.walk(module):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id == var_name and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                return True
        return False

    def _key_used_as_subscript(self, var_name, body):
        """Return True if var_name appears as a subscript key anywhere in body.

        Detects patterns like some_dict[var_name] or some_dict[var_name] = value.
        When iterating a plain dict (key type = Any), this implies the key is str,
        since it is being used to index another dict in the loop body.
        """
        module = ast.Module(body=list(body), type_ignores=[])
        for node in ast.walk(module):
            if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Name)
                    and node.slice.id == var_name):
                return True
        return False

    def _kv_types_from_annotation(self, annotation):
        """Extract (key_ann, val_ann) AST nodes from a dict[K, V] annotation node.

        Returns the raw AST slice elements so nested types like dict[str, int]
        are preserved intact (not flattened to a string).
        """
        if (isinstance(annotation, ast.Subscript) and isinstance(annotation.slice, ast.Tuple)
                and len(annotation.slice.elts) >= 2):
            return annotation.slice.elts[0], annotation.slice.elts[1]
        return self._any_ann(), self._any_ann()

    def _get_base_type_name(self, ann_node):
        """Return the base type name string from an annotation node.

        For simple names (int, str, dict) returns the id.
        For subscripts (dict[str, int]) returns the outer name ('dict').
        """
        if isinstance(ann_node, ast.Name):
            return ann_node.id
        if isinstance(ann_node, ast.Subscript) and isinstance(ann_node.value, ast.Name):
            return ann_node.value.id
        return "Any"

    def _get_dict_kv_types(self, dict_var_name):
        """Return (key_ann, val_ann) annotation nodes from a variable's dict[K, V] annotation."""
        if dict_var_name and dict_var_name in self.variable_annotations:
            return self._kv_types_from_annotation(self.variable_annotations[dict_var_name])
        return self._any_ann(), self._any_ann()

    def _get_kv_types_from_call(self, call_node):
        """Return (key_ann, val_ann) annotation nodes from a function call's return annotation."""
        if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
            if func_name in self.function_return_annotations:
                return self._kv_types_from_annotation(self.function_return_annotations[func_name])
        return self._any_ann(), self._any_ann()

    def _get_kv_types_from_attribute(self, attr_node):
        """Return (key_ann, val_ann) annotation nodes from c.d via class attribute lookup."""
        if not (isinstance(attr_node, ast.Attribute) and isinstance(attr_node.value, ast.Name)):
            return self._any_ann(), self._any_ann()
        var_name = attr_node.value.id
        attr_name = attr_node.attr

        # Get class name from explicit annotation (c: C = ...) or from c = C()
        class_name = None
        ann = self.variable_annotations.get(var_name)
        if isinstance(ann, ast.Name):
            class_name = ann.id
        if class_name is None:
            class_name = self.instance_class_map.get(var_name)
        if class_name is None:
            return self._any_ann(), self._any_ann()

        attr_ann = self.class_attr_annotations.get(class_name, {}).get(attr_name)
        if attr_ann is not None:
            return self._kv_types_from_annotation(attr_ann)
        return self._any_ann(), self._any_ann()

    def _get_kv_types_from_subscript(self, subscript_node):
        """Return (key_ann, val_ann) for a subscript dict expression.

        For d["key"].items() where d: dict[str, dict[K, V]], returns (K, V).
        Uses _create_subscript_annotation to find the value type of d at the
        subscript key, then extracts the K/V types from that inner dict type.
        """
        val_ann = self._create_subscript_annotation(subscript_node)
        if val_ann is not None:
            return self._kv_types_from_annotation(val_ann)
        return self._any_ann(), self._any_ann()

    def _create_dict_list_assign(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            self, node, var_name, dict_node, method, elem_ann):
        """Create: var_name: list[base(elem_ann)] = dict_node.method()

        The list annotation uses only the BASE type name (e.g. 'dict' for
        dict[str, int]) so the C++ list subscript handler can call
        get_typet("dict") and correctly extract a dict struct from the PyObj.
        Full nested type info is preserved via the loop variable's own annotation
        (produced by _create_var_subscript_assign).
        """
        base_name = self._get_base_type_name(elem_ann)
        actual_base = base_name if base_name and base_name != "Any" else "Any"
        annotation = ast.Subscript(
            value=ast.Name(id="list", ctx=ast.Load()),
            slice=ast.Name(id=actual_base, ctx=ast.Load()),
            ctx=ast.Load(),
        )
        method_call = ast.Call(
            func=ast.Attribute(value=dict_node, attr=method, ctx=ast.Load()),
            args=[],
            keywords=[],
        )
        self.ensure_all_locations(method_call, node)
        assign = ast.AnnAssign(
            target=ast.Name(id=var_name, ctx=ast.Store()),
            annotation=annotation,
            value=method_call,
            simple=1,
        )
        self.ensure_all_locations(assign, node)
        return assign

    def _create_var_subscript_assign(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            self, node, var_name, list_var, index_var, elem_ann):
        """Create: var_name: elem_ann = list_var[index_var]

        Uses the FULL annotation node (e.g. dict[str, int]) so that
        variable_annotations[var_name] carries nested type information for
        subsequent inner-loop type resolution.
        """
        annotation = elem_ann  # full AST annotation node
        subscript = ast.Subscript(
            value=ast.Name(id=list_var, ctx=ast.Load()),
            slice=ast.Name(id=index_var, ctx=ast.Load()),
            ctx=ast.Load(),
        )
        self.ensure_all_locations(subscript, node)
        assign = ast.AnnAssign(
            target=ast.Name(id=var_name, ctx=ast.Store()),
            annotation=annotation,
            value=subscript,
            simple=1,
        )
        self.ensure_all_locations(assign, node)
        return assign

    def _create_dict_size_assertion(self, node, keys_var, length_var):
        """Create dict-size check to detect mutation during iteration."""
        size_call = ast.Call(
            func=ast.Name(id="len", ctx=ast.Load()),
            args=[ast.Name(id=keys_var, ctx=ast.Load())],
            keywords=[],
        )
        assert_stmt = ast.Assert(
            test=ast.Compare(
                left=size_call,
                ops=[ast.Eq()],
                comparators=[ast.Name(id=length_var, ctx=ast.Load())],
            ),
            msg=ast.Constant(value="RuntimeError: dictionary changed size during iteration"),
        )
        self.ensure_all_locations(assert_stmt, node)
        return assert_stmt

    @staticmethod
    def _is_zip_call(it):
        """Return True if `it` is a zip(...) call with at least one argument."""
        return (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and it.func.id == "zip" and len(it.args) >= 1 and not it.keywords)

    @staticmethod
    def _is_filter_call(it):
        """Return True if `it` is a filter(func, iterable) call."""
        return (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and it.func.id == "filter" and len(it.args) == 2 and not it.keywords)

    @staticmethod
    def _is_reversed_call(it):
        """Return True if `it` is a reversed(seq) call (seq is not range())."""
        return (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and it.func.id == "reversed" and len(it.args) == 1 and not it.keywords)

    def _materialize_for_iter(self, node, seq, loop_id, suffix=""):
        """Bind `seq` to an iterable variable usable by index-based iteration.

        A bare Name is used directly (preserving its type annotation); any other
        expression is copied into a fresh annotated ESBMC_iter variable. Returns
        (iter_var_name, setup_statements, element_type).
        """
        annotation_id = self._get_iterable_type_annotation(seq)
        element_type = self._get_element_type_from_container(annotation_id, seq)
        if isinstance(seq, ast.Name):
            return seq.id, [], element_type
        iter_var_name = f"ESBMC_iter_{loop_id}{suffix}"
        saved = node.iter
        node.iter = seq
        iter_assign = self._create_iter_assignment(node, annotation_id, iter_var_name, element_type)
        node.iter = saved
        return iter_var_name, [iter_assign], element_type

    def _make_target_assign(self, node, target, iter_var_name, index_var, element_type):
        """Build `target = iter_var[index]` plus any tuple/list unpacking assigns."""
        current = ast.Subscript(
            value=ast.Name(id=iter_var_name, ctx=ast.Load()),
            slice=ast.Name(id=index_var, ctx=ast.Load()),
            ctx=ast.Load(),
        )
        self.ensure_all_locations(current, node)
        name = self._name_id_or_none(target) or "ESBMC_loop_var"
        ann = ast.Name(id=(element_type if element_type and element_type != "Any" else "Any"),
                       ctx=ast.Load())
        assign = ast.AnnAssign(
            target=ast.Name(id=name, ctx=ast.Store()), annotation=ann, value=current, simple=1)
        self.ensure_all_locations(assign, node)
        out = [assign]
        if isinstance(target, (ast.Tuple, ast.List)):
            for i, elt in enumerate(target.elts):
                if not isinstance(elt, ast.Name):
                    continue
                unpack = ast.Assign(
                    targets=[ast.Name(id=elt.id, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Name(id=name, ctx=ast.Load()),
                        slice=ast.Constant(value=i),
                        ctx=ast.Load()),
                )
                self.ensure_all_locations(unpack, node)
                out.append(unpack)
        return out

    def _make_index_step(self, node, index_var, step):
        """Build `index = index +/- |step|` as an annotated int assignment."""
        op = ast.Add() if step >= 0 else ast.Sub()
        inc = ast.AnnAssign(
            target=self.create_name_node(index_var, ast.Store(), node),
            annotation=self.create_name_node("int", ast.Load(), node),
            value=ast.BinOp(
                left=self.create_name_node(index_var, ast.Load(), node),
                op=op,
                right=self.create_constant_node(abs(step), node)),
            simple=1,
        )
        self.ensure_all_locations(inc, node)
        return inc

    def _transform_reversed_for(self, node):
        """for x in reversed(seq): -> backward index-based while loop over seq."""
        loop_id = self.iterable_loop_counter
        self.iterable_loop_counter += 1
        seq = node.iter.args[0]
        index_var = f"ESBMC_index_{loop_id}"
        length_var = f"ESBMC_length_{loop_id}"

        iter_var_name, setup, element_type = self._materialize_for_iter(node, seq, loop_id)
        setup.append(self._create_length_assignment(node, iter_var_name, length_var))

        # ESBMC_index = ESBMC_length - 1
        index_assign = ast.AnnAssign(
            target=self.create_name_node(index_var, ast.Store(), node),
            annotation=self.create_name_node("int", ast.Load(), node),
            value=ast.BinOp(
                left=self.create_name_node(length_var, ast.Load(), node),
                op=ast.Sub(),
                right=self.create_constant_node(1, node)),
            simple=1,
        )
        self.ensure_all_locations(index_assign, node)
        setup.append(index_assign)

        # while ESBMC_index >= 0:
        while_cond = ast.Compare(
            left=self.create_name_node(index_var, ast.Load(), node),
            ops=[ast.GtE()],
            comparators=[self.create_constant_node(0, node)])
        self.ensure_all_locations(while_cond, node)

        body = self._make_target_assign(node, node.target, iter_var_name, index_var, element_type)
        body.append(self._make_index_step(node, index_var, -1))
        body.extend(node.body)

        while_stmt = ast.While(test=while_cond, body=body, orelse=[])
        result = setup + [while_stmt]
        for stmt in result:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)
        return result

    def _transform_filter_for(self, node):
        """for x in filter(func, seq): -> while loop over seq guarded by func(x).

        filter(None, seq) keeps truthy elements.
        """
        loop_id = self.iterable_loop_counter
        self.iterable_loop_counter += 1
        func = node.iter.args[0]
        seq = node.iter.args[1]
        index_var = f"ESBMC_index_{loop_id}"
        length_var = f"ESBMC_length_{loop_id}"

        iter_var_name, setup, element_type = self._materialize_for_iter(node, seq, loop_id)
        setup.append(self._create_index_assignment(node, index_var))
        setup.append(self._create_length_assignment(node, iter_var_name, length_var))
        while_cond = self._create_while_condition(node, index_var, length_var)

        body = self._make_target_assign(node, node.target, iter_var_name, index_var, element_type)
        body.append(self._make_index_step(node, index_var, 1))

        name = self._name_id_or_none(node.target) or "ESBMC_loop_var"
        if isinstance(func, ast.Constant) and func.value is None:
            pred = ast.Name(id=name, ctx=ast.Load())
        else:
            pred = ast.Call(
                func=copy.deepcopy(func),
                args=[ast.Name(id=name, ctx=ast.Load())],
                keywords=[])
        self.ensure_all_locations(pred, node)
        guard = ast.If(test=pred, body=list(node.body), orelse=[])
        self.ensure_all_locations(guard, node)
        body.append(guard)

        while_stmt = ast.While(test=while_cond, body=body, orelse=[])
        result = setup + [while_stmt]
        for stmt in result:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)
        return result

    def _transform_zip_for(self, node):
        """for a, b, ... in zip(s0, s1, ...): -> parallel index-based while loop.

        Iterates up to the shortest sequence (Python's zip semantics).
        """
        loop_id = self.iterable_loop_counter
        self.iterable_loop_counter += 1
        seqs = node.iter.args
        index_var = f"ESBMC_index_{loop_id}"
        length_var = f"ESBMC_length_{loop_id}"

        iter_names = []
        elem_types = []
        setup = []
        for i, seq in enumerate(seqs):
            nm, st, et = self._materialize_for_iter(node, seq, loop_id, suffix=f"_{i}")
            iter_names.append(nm)
            elem_types.append(et)
            setup.extend(st)

        # ESBMC_length = min(len(iter0), len(iter1), ...)
        def len_call(nm):
            call = ast.Call(
                func=self.create_name_node("len", ast.Load(), node),
                args=[self.create_name_node(nm, ast.Load(), node)],
                keywords=[])
            self.ensure_all_locations(call, node)
            return call

        length_expr = len_call(iter_names[0])
        for nm in iter_names[1:]:
            length_expr = ast.Call(
                func=self.create_name_node("min", ast.Load(), node),
                args=[length_expr, len_call(nm)],
                keywords=[])
            self.ensure_all_locations(length_expr, node)

        length_assign = ast.AnnAssign(
            target=self.create_name_node(length_var, ast.Store(), node),
            annotation=self.create_name_node("int", ast.Load(), node),
            value=length_expr,
            simple=1,
        )
        self.ensure_all_locations(length_assign, node)
        setup.append(self._create_index_assignment(node, index_var))
        setup.append(length_assign)

        while_cond = self._create_while_condition(node, index_var, length_var)

        body = []
        target = node.target
        targets = target.elts if isinstance(target, (ast.Tuple, ast.List)) else None
        if targets is not None and len(targets) == len(iter_names):
            for tgt, nm, et in zip(targets, iter_names, elem_types):
                if not isinstance(tgt, ast.Name):
                    continue
                cur = ast.Subscript(
                    value=ast.Name(id=nm, ctx=ast.Load()),
                    slice=ast.Name(id=index_var, ctx=ast.Load()),
                    ctx=ast.Load())
                ann = ast.Name(id=(et if et and et != "Any" else "Any"), ctx=ast.Load())
                assign = ast.AnnAssign(
                    target=ast.Name(id=tgt.id, ctx=ast.Store()),
                    annotation=ann, value=cur, simple=1)
                self.ensure_all_locations(assign, node)
                body.append(assign)
        else:
            # Single target variable receives a tuple of the parallel elements.
            name = self._name_id_or_none(target) or "ESBMC_loop_var"
            elts = [
                ast.Subscript(
                    value=ast.Name(id=nm, ctx=ast.Load()),
                    slice=ast.Name(id=index_var, ctx=ast.Load()),
                    ctx=ast.Load()) for nm in iter_names
            ]
            tup = ast.Tuple(elts=elts, ctx=ast.Load())
            assign = ast.AnnAssign(
                target=ast.Name(id=name, ctx=ast.Store()),
                annotation=ast.Name(id="tuple", ctx=ast.Load()),
                value=tup, simple=1)
            self.ensure_all_locations(assign, node)
            body.append(assign)

        body.append(self._make_index_step(node, index_var, 1))
        body.extend(node.body)

        while_stmt = ast.While(test=while_cond, body=body, orelse=[])
        result = setup + [while_stmt]
        for stmt in result:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)
        return result

    def _transform_iterable_for(self, node):  # pylint: disable=too-many-locals
        """
        Transform general iterable for loops to while loops with unique variable names.
        """
        # Generate unique variable names for this loop level
        loop_id = self.iterable_loop_counter
        self.iterable_loop_counter += 1

        index_var = f"ESBMC_index_{loop_id}"
        length_var = f"ESBMC_length_{loop_id}"
        iter_var_base = "ESBMC_iter"

        # Handle the target variable name
        target_var_name = self._name_id_or_none(node.target) or "ESBMC_loop_var"

        # Determine annotation type based on the iterable value
        annotation_id = self._get_iterable_type_annotation(node.iter)

        # Get element type for proper annotation
        element_type = self._get_element_type_from_container(annotation_id, node.iter)

        # Handle dict iteration
        if annotation_id in ["dict", "Dict"]:
            # Transform: for k in d: into for k in d.keys():
            if isinstance(node.iter, ast.Name):
                # Create d.keys() call
                keys_call = ast.Call(
                    func=ast.Attribute(value=node.iter, attr="keys", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
                self.ensure_all_locations(keys_call, node)
                node.iter = keys_call
                annotation_id = "list"  # d.keys() returns list

        # Determine iterator variable name and whether to create ESBMC_iter
        if isinstance(node.iter, ast.Name):
            # For any Name reference (parameter or variable), use it directly
            # This preserves type information for the converter
            iter_var_name = node.iter.id
            setup_statements = []
        else:
            # For other iterables (literals, calls, expressions), create ESBMC_iter copy
            iter_var_name = f"{iter_var_base}_{loop_id}"
            iter_assign = self._create_iter_assignment(node, annotation_id, iter_var_name,
                                                       element_type)
            setup_statements = [iter_assign]

        # Create common setup statements (index and length) with unique names
        index_assign = self._create_index_assignment(node, index_var)
        length_assign = self._create_length_assignment(node, iter_var_name, length_var)
        setup_statements.extend([index_assign, length_assign])

        # Create while loop condition with unique variable names
        while_cond = self._create_while_condition(node, index_var, length_var)

        # Create loop body with unique variable names
        transformed_body = self._create_loop_body(node, target_var_name, iter_var_name, index_var,
                                                  element_type)

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
        if element_type and element_type != "Any":
            # Create Subscript: list[element_type]
            iter_annotation = ast.Subscript(
                value=ast.Name(id="list", ctx=ast.Load()),
                slice=ast.Name(id=element_type, ctx=ast.Load()),
                ctx=ast.Load(),
            )
        elif annotation_id in ("list", "List", "tuple", "Tuple"):
            # Use list[Any] rather than bare Any so the C++ converter treats
            # ESBMC_iter as an indexable list (avoiding the index2t assertion
            # that fires when subscripting a void* variable).  Bare 'list'
            # must be avoided because get_elem_type_from_annotation would then
            # return list_type itself as the element type, causing ptr+ptr
            # arithmetic crashes in arith_2ops.
            iter_annotation = ast.Subscript(
                value=ast.Name(id="list", ctx=ast.Load()),
                slice=ast.Name(id="Any", ctx=ast.Load()),
                ctx=ast.Load(),
            )
        else:
            # Use 'Any' instead of bare 'list' to avoid misinterpreting the
            # container type as the element type in the C++ converter,
            # which causes invalid ptr+ptr arithmetic (crashes in arith_2ops).
            iter_annotation = ast.Name(id="Any", ctx=ast.Load())

        # Create: ESBMC_iter_N: list[element_type] = <iterable>
        iter_assign = ast.AnnAssign(
            target=ast.Name(id=iter_var_name, ctx=ast.Store()),
            annotation=iter_annotation,
            value=node.iter,
            simple=1,
        )
        self.ensure_all_locations(iter_assign, node)
        return iter_assign

    def _create_index_assignment(self, node, index_var="ESBMC_index"):
        """Create ESBMC_index assignment with custom name."""
        index_target = self.create_name_node(index_var, ast.Store(), node)
        index_value = self.create_constant_node(0, node)
        int_annotation = self.create_name_node("int", ast.Load(), node)
        index_assign = ast.AnnAssign(target=index_target,
                                     annotation=int_annotation,
                                     value=index_value,
                                     simple=1)
        self.ensure_all_locations(index_assign, node)
        return index_assign

    def _create_length_assignment(self, node, iter_var_name, length_var="ESBMC_length"):
        """Create ESBMC_length assignment with custom name."""
        length_target = self.create_name_node(length_var, ast.Store(), node)
        int_annotation = self.create_name_node("int", ast.Load(), node)

        # The function_call_builder will map len() to either:
        # - strlen(): string types
        # - __ESBMC_get_object_size(): list/dict/set/sequence types
        len_func = self.create_name_node("len", ast.Load(), node)

        iter_arg = self.create_name_node(iter_var_name, ast.Load(), node)
        len_call = ast.Call(func=len_func, args=[iter_arg], keywords=[])
        self.ensure_all_locations(len_call, node)

        length_assign = ast.AnnAssign(target=length_target,
                                      annotation=int_annotation,
                                      value=len_call,
                                      simple=1)
        self.ensure_all_locations(length_assign, node)
        return length_assign

    def _create_while_condition(self, node, index_var="ESBMC_index", length_var="ESBMC_length"):
        """Create while loop condition with custom variable names."""
        index_left = self.create_name_node(index_var, ast.Load(), node)
        length_right = self.create_name_node(length_var, ast.Load(), node)
        lt_op = ast.Lt()
        self.ensure_all_locations(lt_op, node)
        while_cond = ast.Compare(left=index_left, ops=[lt_op], comparators=[length_right])
        self.ensure_all_locations(while_cond, node)
        return while_cond

    def _create_loop_body(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        node,
        target_var_name,
        iter_var_name,
        index_var,
        element_type,
    ):
        """Create the body of the while loop with proper type annotations."""
        # Current iterable element expression: iter_var[index]
        current_item = ast.Subscript(
            value=ast.Name(id=iter_var_name, ctx=ast.Load()),
            slice=ast.Name(id=index_var, ctx=ast.Load()),
            ctx=ast.Load(),
        )
        self.ensure_all_locations(current_item, node)

        unpack_assigns = []
        # Support tuple/list unpacking targets in for-loops:
        # for a, b in items: ...
        if isinstance(node.target, (ast.Tuple, ast.List)):
            for i, elt in enumerate(node.target.elts):
                if not isinstance(elt, ast.Name):
                    continue
                unpack_assign = ast.Assign(
                    targets=[ast.Name(id=elt.id, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Name(id=target_var_name, ctx=ast.Load()),
                        slice=ast.Constant(value=i),
                        ctx=ast.Load(),
                    ),
                )
                self.ensure_all_locations(unpack_assign, node)
                unpack_assigns.append(unpack_assign)

        # Create target variable annotation
        if element_type and element_type != "Any":
            target_annotation = ast.Name(id=element_type, ctx=ast.Load())
        else:
            target_annotation = ast.Name(id="Any", ctx=ast.Load())

        # Create: target: element_type = iter_var[index]
        target_assign = ast.AnnAssign(
            target=ast.Name(id=target_var_name, ctx=ast.Store()),
            annotation=target_annotation,
            value=current_item,
            simple=1,
        )
        self.ensure_all_locations(target_assign, node)

        # Create: index += 1
        index_increment = ast.AnnAssign(
            target=ast.Name(id=index_var, ctx=ast.Store()),
            annotation=ast.Name(id="int", ctx=ast.Load()),
            value=ast.BinOp(
                left=ast.Name(id=index_var, ctx=ast.Load()),
                op=ast.Add(),
                right=ast.Constant(value=1),
            ),
            simple=1,
        )
        self.ensure_all_locations(index_increment, node)

        # Combine with original body (include unpack assignments when needed)
        if unpack_assigns:
            return [target_assign] + unpack_assigns + [index_increment] + node.body
        return [target_assign, index_increment] + node.body

    def _create_index_increment(self, node, index_var="ESBMC_index"):
        """Create index increment statement with custom index variable name."""
        inc_target = self.create_name_node(index_var, ast.Store(), node)
        inc_left = self.create_name_node(index_var, ast.Load(), node)
        inc_right = self.create_constant_node(1, node)
        add_op = ast.Add()
        self.ensure_all_locations(add_op, node)
        inc_binop = ast.BinOp(left=inc_left, op=add_op, right=inc_right)
        self.ensure_all_locations(inc_binop, node)
        int_annotation = self.create_name_node("int", ast.Load(), node)
        index_increment = ast.AnnAssign(target=inc_target,
                                        annotation=int_annotation,
                                        value=inc_binop,
                                        simple=1)
        self.ensure_all_locations(index_increment, node)
        return index_increment

    def visit_Name(self, node):
        return node

    def _infer_type_from_value(self, value):
        """Infer the type string from an AST value node"""
        # Handle direct AST node types
        node_type_map = {
            ast.List: "list",
            ast.Tuple: "tuple",
            ast.Dict: "dict",
            ast.Set: "set",
        }

        value_type = type(value)
        if value_type in node_type_map:
            return node_type_map[value_type]

        if isinstance(value, ast.Name):
            return self.known_variable_types.get(value.id, "Any")

        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
            return "bool"

        if isinstance(value, ast.BoolOp):
            operand_types = [self._infer_type_from_value(operand) for operand in value.values]
            if operand_types and all(operand_type == operand_types[0]
                                     for operand_type in operand_types[1:]):
                return operand_types[0]
            return "Any"

        if isinstance(value, ast.Compare):
            return "bool"

        # Handle subscript operations (e.g., d["key"], lst[0])
        if isinstance(value, ast.Subscript):
            return self._infer_type_from_subscript(value)

        # Handle constant values
        if isinstance(value, ast.Constant):
            return self._infer_type_from_constant(value)

        # Handle function calls
        if isinstance(value, ast.Call):
            return self._infer_type_from_call(value)

        return "Any"

    def _infer_type_from_constant(self, constant_node):
        """Infer type from ast.Constant node"""
        value = constant_node.value
        constant_type_map = {
            str: "str",
            int: "int",
            float: "float",
            bool: "bool",
            complex: "complex",
        }
        return constant_type_map.get(type(value), "Any")

    def _infer_type_from_call(self, call_node):
        """Infer type from function call nodes"""
        if not isinstance(call_node.func, ast.Name):
            return "Any"

        # Check if this is a class instantiation (constructor call)
        func_name = call_node.func.id

        # If the function name starts with uppercase, it's likely a class constructor
        if func_name and func_name[0].isupper():
            return func_name

        call_type_map = {
            "range": "range",
            "list": "list",
            "dict": "dict",
            "set": "set",
            "tuple": "tuple",
            "nondet_list": "list",
            "nondet_dict": "dict",
        }

        return call_type_map.get(func_name, "Any")

    def _copy_location_info(self, source_node, target_node):
        """Copy all location information from source to target node"""
        target_node.lineno = getattr(source_node, "lineno", 1)
        target_node.col_offset = getattr(source_node, "col_offset", 0)
        if hasattr(source_node, "end_lineno"):
            target_node.end_lineno = source_node.end_lineno
        if hasattr(source_node, "end_col_offset"):
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
        leaf_pairs = []

        def collect_unpacking_pairs(target_node, value_node):
            if isinstance(target_node, ast.Name):
                leaf_pairs.append((target_node, value_node))
                return True

            if not isinstance(target_node, (ast.Tuple, ast.List)):
                return False
            if not isinstance(value_node, (ast.Tuple, ast.List)):
                return False
            if len(target_node.elts) != len(value_node.elts):
                return False

            for target_elem, value_elem in zip(target_node.elts, value_node.elts):
                if not collect_unpacking_pairs(target_elem, value_elem):
                    return False
            return True

        if not collect_unpacking_pairs(target, value):
            # Don't transform unsupported unpacking shapes - let converter handle it
            return source_node

        for target_node, value_node in leaf_pairs:
            target_copy = copy.deepcopy(target_node)
            value_copy = copy.deepcopy(value_node)
            individual_assign = self._create_individual_assignment(target_copy, value_copy,
                                                                   source_node)
            self._update_variable_types_simple(target_copy, value_copy)
            assignments.append(individual_assign)

        return assignments

    def _create_annotation_node_from_value(self, value):
        """Create an annotation AST node from a value node for storage"""
        if isinstance(value, ast.List):
            return self._create_list_annotation(value)
        if isinstance(value, ast.Dict):
            return self._create_dict_annotation(value)
        if isinstance(value, ast.Subscript):
            return self._create_subscript_annotation(value)
        if isinstance(value, ast.Call):
            return self._create_annotation_from_call(value)
        return None

    def _create_annotation_from_call(self, call_node):
        """Create annotation from known function calls (nondet_dict/nondet_list)."""
        if not isinstance(call_node.func, ast.Name):
            return None
        func_name = call_node.func.id

        if func_name == "nondet_dict":
            key_t = "int"
            val_t = "int"
            for kw in call_node.keywords:
                if kw.arg == "key_type" and isinstance(kw.value, ast.Call):
                    key_t = self._nondet_call_to_type(kw.value) or key_t
                elif kw.arg == "value_type" and isinstance(kw.value, ast.Call):
                    val_t = self._nondet_call_to_type(kw.value) or val_t
            return ast.Subscript(
                value=ast.Name(id="dict", ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[
                        ast.Name(id=key_t, ctx=ast.Load()),
                        ast.Name(id=val_t, ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            )

        if func_name == "nondet_list":
            elem_t = "int"
            if len(call_node.args) >= 2 and isinstance(call_node.args[1], ast.Call):
                elem_t = self._nondet_call_to_type(call_node.args[1]) or elem_t
            for kw in call_node.keywords:
                if kw.arg == "elem_type" and isinstance(kw.value, ast.Call):
                    elem_t = self._nondet_call_to_type(kw.value) or elem_t
            return ast.Subscript(
                value=ast.Name(id="list", ctx=ast.Load()),
                slice=ast.Name(id=elem_t, ctx=ast.Load()),
                ctx=ast.Load(),
            )

        return None

    @staticmethod
    def _nondet_call_to_type(call_node):
        """Extract the type name from `nondet_*()` calls."""
        if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
            name = call_node.func.id
            if name.startswith("nondet_"):
                return name[len("nondet_"):]
        return None

    def _expand_nondet_call(self, target, call, source_node):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        """Expand nondet_list && nondet_dict call into an inline loop
        to replace the effect in nondet.py
        e.g.:
            x = nondet_list(3, nondet_bool())  -->
                x: list[bool] = []
                __nd_size_0: int = nondet_int()
                __ESBMC_assume(__nd_size_0 >= 0)
                __ESBMC_assume(__nd_size_0 <= 3)
                __nd_i_0: int = 0
                while __nd_i_0 < __nd_size_0:
                    x.append(nondet_bool())
                    __nd_i_0 = __nd_i_0 + 1

            x = nondet_dict(2, key_type=nondet_str(), value_type=nondet_float())  -->
                x: dict[str, float] = {}
                __nd_size_0: int = nondet_int()
                __ESBMC_assume(__nd_size_0 >= 0)
                __ESBMC_assume(__nd_size_0 <= 2)
                if __nd_size_0 >= 1: x["0"] = nondet_float()
                if __nd_size_0 >= 2: x["1"] = nondet_float()
        """
        uid = self.nondet_expand_counter
        self.nondet_expand_counter += 1
        func_name = call.func.id
        loc = source_node

        # Parse arguments
        max_size_node = ast.Constant(value=8)
        if call.args:
            max_size_node = call.args[0]

        # Determine nondet type functions
        def _get_nondet_func(call_arg):
            """Extract function name'nondet_bool' from a Call node."""
            if isinstance(call_arg, ast.Call) and isinstance(call_arg.func, ast.Name):
                return call_arg.func.id
            return None

        def _get_type_name(call_arg):
            """Extract type name'bool' from nondet_bool() Call node."""
            fn = _get_nondet_func(call_arg)
            if fn and fn.startswith("nondet_"):
                return fn[len("nondet_"):]
            return "int"

        if func_name == "nondet_list":
            elem_func = "nondet_int"
            elem_type_name = "int"
            if len(call.args) >= 2:
                fn = _get_nondet_func(call.args[1])
                if fn:
                    elem_func = fn
                    elem_type_name = _get_type_name(call.args[1])
            for kw in call.keywords:
                if kw.arg == "elem_type":
                    fn = _get_nondet_func(kw.value)
                    if fn:
                        elem_func = fn
                        elem_type_name = _get_type_name(kw.value)
        elif func_name == "nondet_dict":
            val_func = "nondet_int"
            key_type_name = "int"
            val_type_name = "int"
            for kw in call.keywords:
                if kw.arg == "key_type":
                    fn = _get_nondet_func(kw.value)
                    if fn:
                        key_type_name = _get_type_name(kw.value)
                elif kw.arg == "value_type":
                    fn = _get_nondet_func(kw.value)
                    if fn:
                        val_func = fn
                        val_type_name = _get_type_name(kw.value)

        # create AST nodes
        def name(n, ctx=ast.Load()):
            nd = ast.Name(id=n, ctx=ctx)
            self.ensure_all_locations(nd, loc)
            return nd

        def const(v):
            nd = ast.Constant(value=v)
            self.ensure_all_locations(nd, loc)
            return nd

        def call_node(fn, args=None):
            nd = ast.Call(func=name(fn), args=args or [], keywords=[])
            self.ensure_all_locations(nd, loc)
            return nd

        size_var = f"__nd_size_{uid}"
        idx_var = f"__nd_i_{uid}"
        var_name = self._name_id_or_none(target)
        if var_name is None:
            return None
        stmts = []

        # x: list[T] = [] && x: dict[K,V] = {}
        if func_name == "nondet_list":
            init_val = ast.List(elts=[], ctx=ast.Load())
            annotation = ast.Subscript(value=name("list"),
                                       slice=name(elem_type_name),
                                       ctx=ast.Load())
        else:
            init_val = ast.Dict(keys=[], values=[])
            annotation = ast.Subscript(
                value=name("dict"),
                slice=ast.Tuple(elts=[name(key_type_name), name(val_type_name)], ctx=ast.Load()),
                ctx=ast.Load(),
            )
        self.ensure_all_locations(init_val, loc)
        self.ensure_all_locations(annotation, loc)

        init_assign = ast.AnnAssign(
            target=name(var_name, ast.Store()),
            annotation=annotation,
            value=init_val,
            simple=1,
        )
        self.ensure_all_locations(init_assign, loc)
        stmts.append(init_assign)

        # Store annotation for dict iteration support
        self.variable_annotations[var_name] = annotation
        self.known_variable_types[var_name] = ("list" if func_name == "nondet_list" else "dict")

        # size = nondet_int();
        # assume(size >= 0);
        # assume(size <= max_size);
        size_assign = ast.AnnAssign(
            target=name(size_var, ast.Store()),
            annotation=name("int"),
            value=call_node("nondet_int"),
            simple=1,
        )
        self.ensure_all_locations(size_assign, loc)
        stmts.append(size_assign)

        for op_cls, bound in [(ast.GtE, const(0)), (ast.LtE, max_size_node)]:
            assume_call = ast.Expr(value=ast.Call(
                func=name("__ESBMC_assume"),
                args=[ast.Compare(left=name(size_var), ops=[op_cls()], comparators=[bound])],
                keywords=[],
            ))
            self.ensure_all_locations(assume_call, loc)
            stmts.append(assume_call)

        # i = 0
        idx_assign = ast.AnnAssign(
            target=name(idx_var, ast.Store()),
            annotation=name("int"),
            value=const(0),
            simple=1,
        )
        self.ensure_all_locations(idx_assign, loc)
        stmts.append(idx_assign)

        # Build the collection
        if func_name == "nondet_list":
            append_call = ast.Expr(value=ast.Call(
                func=ast.Attribute(value=name(var_name), attr="append", ctx=ast.Load()),
                args=[call_node(elem_func)],
                keywords=[],
            ))
            self.ensure_all_locations(append_call, loc)

            inc = ast.Assign(
                targets=[name(idx_var, ast.Store())],
                value=ast.BinOp(left=name(idx_var), op=ast.Add(), right=const(1)),
            )
            self.ensure_all_locations(inc, loc)

            while_stmt = ast.While(
                test=ast.Compare(left=name(idx_var), ops=[ast.Lt()], comparators=[name(size_var)]),
                body=[append_call, inc],
                orelse=[],
            )
            self.ensure_all_locations(while_stmt, loc)
            stmts.append(while_stmt)
        else:
            # To avoid solver explosion(timeout)
            # when the dict grows large.
            # Dict is using if-chain with
            # concrete sequential keys (0,1,2,... / False,True /..)
            # makes every contains check trivially decidable.
            # values can remain fully nondeterministic.
            # Future improvement:
            # Once the ESBMC dict C model supports efficient
            # symbolic key insertion(would not be such time-consuming),
            # this can be replaced with a simple loop like nondet_list.
            max_entries = 8
            if isinstance(max_size_node, ast.Constant) and isinstance(max_size_node.value, int):
                max_entries = max_size_node.value

            for entry_idx in range(max_entries):
                concrete_key = self._make_concrete_key(key_type_name, entry_idx, loc)
                dict_assign = ast.Assign(
                    targets=[
                        ast.Subscript(value=name(var_name), slice=concrete_key, ctx=ast.Store())
                    ],
                    value=call_node(val_func),
                )
                self.ensure_all_locations(dict_assign, loc)

                if_stmt = ast.If(
                    test=ast.Compare(
                        left=name(size_var),
                        ops=[ast.GtE()],
                        comparators=[const(entry_idx + 1)],
                    ),
                    body=[dict_assign],
                    orelse=[],
                )
                self.ensure_all_locations(if_stmt, loc)
                stmts.append(if_stmt)

        for s in stmts:
            ast.fix_missing_locations(s)

        return stmts

    def _make_concrete_key(self, key_type_name, index, loc):
        """Generate a concrete key AST node for dict if-chain expansion.
        int  → 0, 1, 2, ...
        bool → False, True  (wraps at 2)
        str  → "0", "1", "2", ...
        """
        if key_type_name == "bool":
            val = bool(index % 2)
        elif key_type_name == "str":
            val = str(index)
        else:
            val = index
        nd = ast.Constant(value=val)
        self.ensure_all_locations(nd, loc)
        return nd

    def _create_list_annotation(self, list_node):
        """Create list[T] annotation from a list literal"""
        if list_node.elts:
            elem_type = self._infer_type_from_value(list_node.elts[0])
            if elem_type and elem_type != "Any":
                return ast.Subscript(
                    value=ast.Name(id="list", ctx=ast.Load()),
                    slice=ast.Name(id=elem_type, ctx=ast.Load()),
                    ctx=ast.Load(),
                )
        return ast.Name(id="list", ctx=ast.Load())

    def _create_dict_annotation(self, dict_node):
        """Create dict[K, V] annotation from a dict literal"""
        if not dict_node.keys or not dict_node.values:
            return ast.Name(id="dict", ctx=ast.Load())

        key_type = self._infer_dict_key_type(dict_node.keys[0])
        value_annotation = self._infer_dict_value_annotation(dict_node.values[0])

        if key_type != "Any" and value_annotation:
            return ast.Subscript(
                value=ast.Name(id="dict", ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[ast.Name(id=key_type, ctx=ast.Load()), value_annotation],
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            )

        return ast.Name(id="dict", ctx=ast.Load())

    def _has_heterogeneous_keys(self, dict_node):
        """Return True if a dict literal has keys of more than one ESBMC-representable type.

        ESBMC stores list elements with a type-specific byte size.  When all
        keys share the same type the retrieval is consistent; when they differ
        (e.g. int=8 bytes vs str=strlen+1 bytes) reading with a single fixed
        size causes an array-bounds violation.
        """
        if not dict_node.keys or len(dict_node.keys) < 2:
            return False
        key_types = [self._infer_dict_key_type(k) for k in dict_node.keys]
        return len(set(key_types)) > 1

    def _has_heterogeneous_values(self, dict_node):
        """Return True if a dict literal has values of more than one ESBMC type.

        Even when both types occupy the same number of bytes (e.g. int and
        float are both 8 bytes on 64-bit), retrieving a float element through
        an int-typed pointer gives the raw IEEE 754 bits, not the numeric
        value, producing a spurious counterexample.
        """
        if not dict_node.values or len(dict_node.values) < 2:
            return False
        val_types = [self._infer_constant_type(v) for v in dict_node.values]
        return len(set(val_types)) > 1

    def _infer_constant_type(self, node):
        """Infer the ESBMC-representable Python type name from a constant node.

        Handles bool (must precede int because bool is a subclass of int),
        int, float, and str.  Returns 'Any' for anything else.
        """
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, float):
                return "float"
            if isinstance(node.value, int):
                return "int"
            if isinstance(node.value, str):
                return "str"
        return "Any"

    def _unroll_het_for(self, node, typed_elts):
        """Emit one typed assignment + one body copy per element.

        typed_elts — list of (type_str, ast_value_node) in iteration order.

        The loop variable (node.target) is renamed to a unique per-iteration
        symbol so that ESBMC never tries to hold two incompatible types in the
        same IR symbol.
        """
        target_name = node.target.id if isinstance(node.target, ast.Name) else "ESBMC_het_var"

        class _RenameVar(ast.NodeTransformer):
            """Replace every Load-context Name(old) with Name(new)."""

            def __init__(self, old, new):
                self.old = old
                self.new = new

            def visit_Name(self, n):
                if n.id == self.old and isinstance(n.ctx, ast.Load):
                    return ast.copy_location(ast.Name(id=self.new, ctx=ast.Load()), n)
                return n

        result = []
        for i, (type_str, value_node) in enumerate(typed_elts):
            iter_var = f"{target_name}_het_{i}_"

            assign = ast.AnnAssign(
                target=ast.Name(id=iter_var, ctx=ast.Store()),
                annotation=ast.Name(id=type_str, ctx=ast.Load()),
                value=copy.deepcopy(value_node),
                simple=1,
            )
            self.ensure_all_locations(assign, node)
            ast.fix_missing_locations(assign)
            result.append(assign)

            renamer = _RenameVar(target_name, iter_var)
            for stmt in node.body:
                renamed = renamer.visit(copy.deepcopy(stmt))
                ast.fix_missing_locations(renamed)
                result.append(renamed)

        return result

    def _transform_het_dict_for(self, node):
        """Unroll a for-loop over a dict literal with heterogeneous key types."""
        dict_node = self.het_dict_literals[node.iter.id]
        typed_elts = [(self._infer_dict_key_type(k), k) for k in dict_node.keys]
        return self._unroll_het_for(node, typed_elts)

    def _transform_het_values_for(self, node, dict_node):
        """Unroll a for-loop over d.values() where values have heterogeneous types."""
        typed_elts = [(self._infer_constant_type(v), v) for v in dict_node.values]
        return self._unroll_het_for(node, typed_elts)

    def _infer_dict_key_type(self, key_node):
        """Infer key type from dict literal's first key"""
        if isinstance(key_node, ast.Constant):
            if isinstance(key_node.value, str):
                return "str"
            if isinstance(key_node.value, int):
                return "int"
        return "Any"

    def _infer_dict_value_annotation(self, value_node):
        """Infer value annotation from dict literal's first value"""
        if isinstance(value_node, ast.List):
            return self._create_list_annotation(value_node)
        if isinstance(value_node, ast.Dict):
            return self._create_annotation_node_from_value(value_node)
        if isinstance(value_node, ast.Constant):
            const_type = type(value_node.value).__name__
            return ast.Name(id=const_type, ctx=ast.Load())
        return None

    def _create_subscript_annotation(self, subscript_node):
        """Extract annotation from subscript operation (e.g., d["key"])"""
        if not isinstance(subscript_node.value, ast.Name):
            return None

        base_var = subscript_node.value.id

        if not (hasattr(self, "variable_annotations") and base_var in self.variable_annotations):
            return None

        base_annotation = self.variable_annotations[base_var]

        # Extract value type from dict[K, V] annotation
        if isinstance(base_annotation, ast.Subscript):
            if (isinstance(base_annotation.value, ast.Name) and base_annotation.value.id == "dict"):
                if (isinstance(base_annotation.slice, ast.Tuple)
                        and len(base_annotation.slice.elts) == 2):
                    return base_annotation.slice.elts[1]

        return None

    def _is_defaultdict_call(self, call_node):
        """Return True if call_node is a collections.defaultdict(...) call.

        Matches only when defaultdict was actually imported from collections.
        Handles both:
          from collections import defaultdict        → defaultdict(...)
          from collections import defaultdict as dd  → dd(...)
          import collections                         → collections.defaultdict(...)
          import collections as col                  → col.defaultdict(...)
        """
        if not isinstance(call_node, ast.Call):
            return False

        func = call_node.func
        # from collections import defaultdict [as alias]
        if self.defaultdict_imported and isinstance(func, ast.Name):
            expected = self.defaultdict_alias or "defaultdict"
            return func.id == expected
        # import collections [as alias]
        if self.collections_module_imported and isinstance(func, ast.Attribute):
            module_name = self.collections_module_alias or "collections"
            return (isinstance(func.value, ast.Name) and func.value.id == module_name
                    and func.attr == "defaultdict")
        return False

    def _get_defaultdict_factory(self, call_node):
        """Return the factory node for a collections.defaultdict call, or None.

        Returns None when:
          - call_node is not a defaultdict call (_is_defaultdict_call is False)
          - defaultdict() called with no args (no auto-insertion)
          - defaultdict(None) called with explicit None (no auto-insertion)

        Callers that need to distinguish "not a defaultdict" from "defaultdict
        without a factory" should call _is_defaultdict_call() separately and
        always rewrite the construction to {}, only recording a factory when
        this method returns non-None.
        """
        if not self._is_defaultdict_call(call_node):
            return None

        if call_node.args:
            factory = call_node.args[0]
            # defaultdict(None) means no auto-insertion; treat like no factory.
            if isinstance(factory, ast.Constant) and factory.value is None:
                return None
            return factory
        return None

    def _make_defaultdict_missing_check(self, dict_name, key_node, factory_node, template):
        """Generate: if key not in dict: dict[key] = factory()

        Returns (stmts, key_expr) where:
          - stmts  is the list of AST statements to insert before the original node
          - key_expr is the safe key expression to use in the original subscript

        When key_node is a complex expression (not a bare Name), a temp variable
        is introduced so the expression is evaluated exactly once. The caller must
        replace the original subscript's slice with the returned key_expr to avoid
        a second evaluation.
        """
        # If the key is a complex expression, store it in a temporary variable first
        pre_stmts = []
        if isinstance(key_node, (ast.Name, ast.Constant)):
            key_load = ast.copy_location(
                (ast.Name(id=key_node.id, ctx=ast.Load())
                 if isinstance(key_node, ast.Name) else key_node),
                template,
            )
        else:
            # Create a temporary variable to hold the key expression so that
            # complex expressions (e.g. f()) are evaluated only once.
            tmp_name = f"__defaultdict_key_tmp_{id(key_node)}"
            tmp_assign = ast.Assign(
                targets=[ast.Name(id=tmp_name, ctx=ast.Store())],
                value=key_node,
                type_comment=None,
            )
            ast.copy_location(tmp_assign, template)
            ast.fix_missing_locations(tmp_assign)
            pre_stmts.append(tmp_assign)
            key_load = ast.Name(id=tmp_name, ctx=ast.Load())
            ast.copy_location(key_load, template)

        # if key not in dict_name:
        not_in = ast.Compare(
            left=key_load,
            ops=[ast.NotIn()],
            comparators=[ast.Name(id=dict_name, ctx=ast.Load())],
        )
        ast.copy_location(not_in, template)
        ast.fix_missing_locations(not_in)

        # dict_name[key] = factory()
        subscript = ast.Subscript(
            value=ast.Name(id=dict_name, ctx=ast.Load()),
            slice=key_load,
            ctx=ast.Store(),
        )
        ast.copy_location(subscript, template)
        # Prefer empty container literals over Call() for built-in container
        # factories: dict storage of an empty list literal binds a concrete
        # PyListObject whose mutations are visible at d[k]. A bare `list()`
        # call returns a value that the empty-dict storage cannot accept
        # without an explicit dict-of-list annotation already present.
        if isinstance(factory_node, ast.Name) and factory_node.id == "list":
            factory_value = ast.List(elts=[], ctx=ast.Load())
        elif isinstance(factory_node, ast.Name) and factory_node.id == "dict":
            factory_value = ast.Dict(keys=[], values=[])
        else:
            factory_value = ast.Call(func=factory_node, args=[], keywords=[])
        ast.copy_location(factory_value, template)
        assign = ast.Assign(targets=[subscript], value=factory_value, type_comment=None)
        ast.copy_location(assign, template)
        ast.fix_missing_locations(assign)

        if_stmt = ast.If(test=not_in, body=[assign], orelse=[])
        ast.copy_location(if_stmt, template)
        ast.fix_missing_locations(if_stmt)

        return pre_stmts + [if_stmt], key_load

    def _lower_defaultdict_reads_in_expr(self, expr, template):
        """Walk expr, find all Load-context d[k] where d is a known defaultdict,
        generate missing-key init stmts, and rewrite each subscript slice to use
        the (possibly temp) key expression.

        Returns (init_stmts, new_expr). init_stmts is a (possibly empty) list of
        AST statements that must be prepended before the containing statement.

        This enables correct auto-insertion semantics for defaultdict reads that
        appear inside arbitrary expressions (assert, return, function args, etc.)
        rather than only as the direct RHS of an assignment.
        """
        all_inits = []
        defaultdict_factory = self._defaultdict_factory
        make_missing_check = self._make_defaultdict_missing_check

        class _Lowerer(ast.NodeTransformer):

            def visit_Subscript(self, node):
                # Recurse into children first (handles nested subscripts).
                self.generic_visit(node)
                if not (isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name)
                        and node.value.id in defaultdict_factory):
                    return node
                dict_name = node.value.id
                factory = defaultdict_factory[dict_name]
                stmts, key_expr = make_missing_check(dict_name, node.slice, factory, template)
                all_inits.extend(stmts)
                node.slice = key_expr
                return node

        new_expr = _Lowerer().visit(expr)
        return all_inits, new_expr

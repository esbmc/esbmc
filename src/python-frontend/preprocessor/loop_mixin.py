"""LoopMixin - extracted from preprocessor.

Contains loop/iteration lowering and related helpers:
range/enumerate/items/reversed(range), iterable while-lowering,
heterogeneous dict iteration unrolling, and defaultdict read lowering.

All shared state lives on Preprocessor (set in Preprocessor.__init__);
this mixin only adds methods.
"""
import ast
import copy
import sys
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
    target_name: str
    module_name: str
    dataclasses_module_names: Set[str]

    @staticmethod
    def _name_id_or_none(node: ast.AST) -> Optional[str]:
        """Return `node.id` when node is ast.Name, else None."""
        if isinstance(node, ast.Name):
            return node.id
        return None

    @staticmethod
    def _is_nullary_lambda(node: ast.AST) -> bool:
        """Return True when node is an `ast.Lambda` taking no parameters."""
        if not isinstance(node, ast.Lambda):
            return False
        args = node.args
        return (not args.args and not args.posonlyargs and not args.kwonlyargs
                and args.vararg is None and args.kwarg is None)

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

    def visit_For(self, node):
        """
        Transform for loops into while loops.

        Python's for-else (`else` runs when the loop completes without break)
        is lowered here at the boundary so all sub-transformers can emit
        their `while ... orelse=[]` shape uniformly without each having to
        remember to preserve the original `orelse`.
        """
        self._update_assignment_call_origins([node.target], None)
        for_else_pre, for_else_post = self._lower_for_else(node)
        result = self._visit_for_inner(node)
        if not isinstance(result, list):
            result = [result]
        return for_else_pre + result + for_else_post

    def _lower_for_else(self, node):
        """Lower `for ... else: <orelse>` into a did-not-break flag.

        Returns (pre_statements, post_statements) to bracket the result of
        the for-to-while transform. Pre: init `flag = True`. Post: emit
        `if flag: <orelse>`. Side effect: rewrites every reachable `break`
        in the body to `flag = False; break`, and clears `node.orelse` so
        downstream transformers see a plain for.

        No-op when `node.orelse` is empty.
        """
        if not node.orelse:
            return [], []
        counter = getattr(self, "_for_else_counter", 0)
        self._for_else_counter = counter + 1
        flag_name = f"ESBMC_for_else_{counter}"
        flag_init = ast.AnnAssign(
            target=ast.Name(id=flag_name, ctx=ast.Store()),
            annotation=ast.Name(id="bool", ctx=ast.Load()),
            value=ast.Constant(value=True),
            simple=1,
        )
        self._copy_location_info(node, flag_init)

        # Rewrite Break in the body to set flag=False first. Walk only the
        # current loop's body, not nested loops -- a Break inside a nested
        # for-else belongs to the inner loop, not this one.
        def rewrite_breaks(stmts):
            new_stmts = []
            for s in stmts:
                if isinstance(s, ast.Break):
                    set_flag = ast.Assign(
                        targets=[ast.Name(id=flag_name, ctx=ast.Store())],
                        value=ast.Constant(value=False),
                    )
                    self._copy_location_info(s, set_flag)
                    new_stmts.append(set_flag)
                    new_stmts.append(s)
                    continue
                if isinstance(s, (ast.For, ast.While, ast.AsyncFor)):
                    # Don't descend; nested-loop breaks bind to the inner loop.
                    new_stmts.append(s)
                    continue
                if isinstance(s, ast.If):
                    s.body = rewrite_breaks(s.body)
                    s.orelse = rewrite_breaks(s.orelse)
                elif isinstance(s, ast.Try):
                    s.body = rewrite_breaks(s.body)
                    s.orelse = rewrite_breaks(s.orelse)
                    for handler in s.handlers:
                        handler.body = rewrite_breaks(handler.body)
                    s.finalbody = rewrite_breaks(s.finalbody)
                elif isinstance(s, (ast.With, ast.AsyncWith)):
                    s.body = rewrite_breaks(s.body)
                elif hasattr(ast, "Match") and isinstance(s, ast.Match):
                    # Python 3.10+: `break` inside `match ... case: <body>`
                    # binds to the enclosing loop. Patterns/guards are
                    # expressions, not statements, so only case bodies need
                    # descent.
                    for case in s.cases:
                        case.body = rewrite_breaks(case.body)
                new_stmts.append(s)
            return new_stmts

        node.body = rewrite_breaks(node.body)

        orelse_stmts = node.orelse
        node.orelse = []

        orelse_guard = ast.If(
            test=ast.Name(id=flag_name, ctx=ast.Load()),
            body=orelse_stmts,
            orelse=[],
        )
        self._copy_location_info(node, orelse_guard)
        return [flag_init], [orelse_guard]

    def _visit_for_inner(self, node):  # pylint: disable=too-many-branches
        """Inner dispatch for visit_For after for-else lowering."""
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
        # Inline list-literal iterable with a tuple/list-unpacking target, e.g.
        # `for u, v in [(1, 2), (3, 4)]:`. Unroll like a name-bound list literal
        # so each statically-known element keeps its tuple/list shape and feeds
        # the converter's assignment-unpacking pipeline (`u, v = (1, 2)`). The
        # generic iterable path would instead bind the element to an Any-typed
        # temp and subscript-unpack it, which aborts with type2t::symbolic_type_excp.
        # Shares _unroll_list_literal_for's tuple-target limitation: the RHS must
        # stay a literal for the converter, so element sub-expressions are not
        # snapshotted -- a tuple element naming a body-mutated variable would read
        # the mutated value (evaluate-once divergence). Constant tuples are exact.
        if (isinstance(node.iter, ast.List) and isinstance(node.target, (ast.Tuple, ast.List))
                and self._can_safely_unroll_list_literal_for(node, node.iter)):
            self.is_range_loop = False
            return self._unroll_list_literal_for(node, node.iter)
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
                # Annotate the binding with the element's class when known, so a
                # user-class element keeps its type even if the body never
                # touches its attributes. Without this, the loop variable falls
                # back to Any (void*), and appending it to a list stores a
                # zero type-id, so a later attribute access on a read-back
                # element dereferences an invalid pointer (#4805).
                elem_class = self._element_instance_class(elt)
                if elem_class:
                    target_assign = ast.AnnAssign(
                        target=ast.Name(id=node.target.id, ctx=ast.Store()),
                        annotation=ast.Name(id=elem_class, ctx=ast.Load()),
                        value=rhs,
                        simple=1,
                    )
                else:
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

        # Step 1: Validate the enumerate call
        self._validate_enumerate_call(enumerate_call)

        # Step 2: Parse and validate the target structure
        target_info = self._parse_enumerate_target(node.target)

        if target_info["type"] == "nested":
            return self._unroll_nested_enumerate_for(node, target_info)

        if target_info["type"] == "unpacking":
            literal, start = self._tuple_literal_enumerate_source(node)
            if literal is not None:
                return self._unroll_enumerate_over_tuples(node, target_info, literal, start)

        # Generate unique variable names for this enumerate loop level
        loop_id = self.enumerate_loop_counter
        self.enumerate_loop_counter += 1

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
            index_elt, value_elt = target.elts
            if isinstance(index_elt, ast.Name) and isinstance(value_elt, (ast.Tuple, ast.List)):
                return {
                    "type": "nested",
                    "index_var": index_elt.id,
                    "value_target": value_elt,
                }
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

    def _reject_unsupported_loop(self, node, message):
        """Emit a located parser-stage diagnostic and abort, as threading lowering does."""
        print(f"ERROR: {self.module_name}:{getattr(node, 'lineno', '?')}: {message}")
        sys.exit(4)

    @staticmethod
    def _static_int_value(expr):
        """Return the int an expression denotes statically, or None."""
        if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
            return int(expr.value)
        if (isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub))
                and isinstance(expr.operand, ast.Constant) and isinstance(expr.operand.value, int)):
            sign = -1 if isinstance(expr.op, ast.USub) else 1
            return sign * int(expr.operand.value)
        return None

    def _nested_enumerate_source(self, node, arity):
        """Return the (list literal, start) the unroll needs, or reject with the reason."""
        iterable, start_value = self._parse_enumerate_arguments(node.iter, node)
        literal = self._resolve_list_literal_iterable(iterable)
        start = self._static_int_value(start_value)

        blocker = None
        if literal is None:
            blocker = "the iterable is not a list literal"
        elif start is None:
            blocker = "the start argument is not a constant integer"
        elif not self._can_safely_unroll_list_literal_for(node, literal):
            blocker = "the loop body contains break/continue/return"
        else:
            for elt in literal.elts:
                if not isinstance(elt, (ast.Tuple, ast.List)):
                    blocker = "an element is not a tuple or list literal"
                elif len(elt.elts) != arity:
                    blocker = f"an element does not have {arity} items"
                if blocker:
                    break
        if blocker:
            self._reject_unsupported_loop(
                node, f"enumerate() with a nested unpacking target is unsupported here: {blocker}")
        return literal, start

    def _tuple_literal_enumerate_source(self, node):
        """Return (list literal, start) when unrolling would preserve tuple shape.

        `for i, v in enumerate(pairs)` is lowered to a while loop whose value
        variable is a bare-`tuple`-annotated subscript read, which carries no
        component types, so a later `a, b = v` is handed an untyped value and
        refuses to unpack. Unrolling binds each tuple literal directly and
        keeps its shape, exactly as the nested-target form already does.

        Returns (None, None) when this does not apply, leaving the ordinary
        lowering in place -- unlike the nested form, a plain value target is
        fully supported there.
        """
        iterable, start_value = self._parse_enumerate_arguments(node.iter, node)
        literal = self._resolve_list_literal_iterable(iterable)
        start = self._static_int_value(start_value)
        if (literal is None or start is None
                or not self._can_safely_unroll_list_literal_for(node, literal)):
            return None, None

        # Only the tuple/list-element case loses type information; leave every
        # other element kind to the ordinary lowering.
        if not literal.elts or not all(
                isinstance(elt, (ast.Tuple, ast.List)) for elt in literal.elts):
            return None, None

        return literal, start

    def _unroll_enumerate_over_tuples(self, node, target_info, literal, start):
        """Bind each tuple literal directly, keeping its shape."""
        unrolled = []
        for offset, elt in enumerate(literal.elts):
            index_assign = ast.AnnAssign(
                target=self.create_name_node(target_info["index_var"], ast.Store(), node),
                annotation=self.create_name_node("int", ast.Load(), node),
                value=self.create_constant_node(start + offset, node),
                simple=1,
            )
            value_assign = ast.Assign(
                targets=[self.create_name_node(target_info["value_var"], ast.Store(), node)],
                value=copy.deepcopy(elt),
            )
            unrolled.extend([index_assign, value_assign])
            unrolled.extend(copy.deepcopy(stmt) for stmt in node.body)

        for stmt in unrolled:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)
        return unrolled

    def _unroll_nested_enumerate_for(self, node, target_info):
        """Unroll `for i, (a, b) in enumerate(seq)` over a statically known list.

        Nested patterns need a literal RHS; see _unroll_list_literal_for (#6744).
        Shares its evaluate-once caveat: elements are not snapshotted, so an
        element naming a body-mutated variable reads the mutated value.
        """
        value_target = target_info["value_target"]
        literal, start = self._nested_enumerate_source(node, len(value_target.elts))

        unrolled = []
        for offset, elt in enumerate(literal.elts):
            index_assign = ast.AnnAssign(
                target=self.create_name_node(target_info["index_var"], ast.Store(), node),
                annotation=self.create_name_node("int", ast.Load(), node),
                value=self.create_constant_node(start + offset, node),
                simple=1,
            )
            value_assign = ast.Assign(
                targets=[copy.deepcopy(value_target)],
                value=copy.deepcopy(elt),
            )
            unrolled.extend([index_assign, value_assign])
            unrolled.extend(copy.deepcopy(stmt) for stmt in node.body)

        for stmt in unrolled:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)
        return unrolled

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
        # A container annotation yields only the element's head name, so
        # `List[Tuple[int, int]]` gives a component-less `Tuple`. Annotating the
        # loop value with that types it as an opaque pointer and a later
        # `a, b = v` cannot unpack it -- strictly worse than `Any`, which the
        # unannotated path uses and which recovers the real type.
        if element_type in ("Tuple", "tuple"):
            element_type = "Any"
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

        # Create the body of the while loop. The loop variable is snapshotted
        # from the counter, then the counter and has_next are advanced *before*
        # the loop body — not after — so that a `continue` in the body (which
        # jumps straight to the while condition) does not skip the advance and
        # spin forever. This mirrors the index-increment-before-body layout used
        # by the items/iterable while-lowerings.
        while_body = ([
            loop_var_init,
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
        ] + transformed_body)

        # Create the while statement
        while_stmt = ast.While(test=while_cond, body=while_body, orelse=[])

        # Return the transformed statements
        return [step_validation, start_assign, has_next_assign, while_stmt]

    @staticmethod
    def _body_destructures_name(body, name):
        """True if `body` assigns `name` to a tuple/list target (`u, v = name`).

        Only a direct statement counts: a nested loop or branch may rebind the
        name first, and this only decides whether to keep the key's concrete
        type, so a missed case just falls back to the existing handling.
        """
        if not name:
            return False
        for stmt in body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not (isinstance(stmt.value, ast.Name) and stmt.value.id == name):
                continue
            if any(isinstance(t, (ast.Tuple, ast.List)) for t in stmt.targets):
                return True
        return False

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
            # val added to a float accumulator in the body => value is float.
            # An unannotated-parameter dict erases the value type to void*, so a
            # float value is read as `*(void**)item->value` and the accumulate
            # lowers to `s = IEEE_ADD(s, (double)w)` -- a numeric cast of a
            # pointer-typed term that produces an ill-sorted floating-point node
            # (#5501). Typing the value as float routes the read through the
            # float_buf path, which yields a real-sorted double.
            if (isinstance(val_ann, ast.Name) and val_ann.id == "Any"
                    and isinstance(_v_elt, ast.Name)
                    and self._value_used_in_float_arith(_v_elt.id, node.body)):
                val_ann = ast.Name(id="float", ctx=ast.Load())

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
            self._emit_items_unpack(node, body, target.elts[0], keys_var, index_var, key_ann)
            self._emit_items_unpack(node, body, target.elts[1], vals_var, index_var, val_ann)
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

    def _value_used_in_float_arith(self, var_name, body):
        """Return True if var_name is combined with a float operand in arithmetic.

        Detects `acc += var` (and `-=`, `*=`, ...) where acc is float, and
        `acc + var` / `var * f` where the other operand is a float literal or a
        name known to hold a float. Used to recover the value element type of an
        unannotated-parameter dict whose values are floats (#5501): the loop
        variable is otherwise erased to void*, and reading a float through void*
        produces an ill-sorted IEEE node.
        """
        arith_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)

        def is_float_operand(operand):
            if isinstance(operand, ast.Constant):
                return isinstance(operand.value, float)
            if isinstance(operand, ast.Name):
                return self.known_variable_types.get(operand.id) == "float"
            return False

        def mentions_var(operand):
            return any(isinstance(n, ast.Name) and n.id == var_name for n in ast.walk(operand))

        module = ast.Module(body=list(body), type_ignores=[])
        for node in ast.walk(module):
            if (isinstance(node, ast.AugAssign) and isinstance(node.op, arith_ops)
                    and is_float_operand(node.target) and mentions_var(node.value)):
                return True
            if isinstance(node, ast.BinOp) and isinstance(node.op, arith_ops):
                left, right = node.left, node.right
                if ((mentions_var(left) and is_float_operand(right))
                        or (mentions_var(right) and is_float_operand(left))):
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

    def _is_concrete_tuple_ann(self, ann_node):
        """True for a full tuple[A, B, ...] annotation node (not the bare name
        'tuple'). Such annotations must be preserved end-to-end so the C++ list
        subscript read rebuilds the concrete tuple struct instead of erasing it
        to void* (#5444)."""
        return (isinstance(ann_node, ast.Subscript)
                and self._get_base_type_name(ann_node) in ("tuple", "Tuple"))

    # Member types the C++ list subscript read can size. A str member reaches
    # the solver as a mismatched bitvector width, a tuple member emits a bare
    # inner `tuple` -- the very #5444 erosion this annotation avoids -- and a
    # bool member mismatches the tuple AST, so anything outside this set leaves
    # the literal unannotated and the unpack refused.
    _SIZABLE_TUPLE_MEMBER_TYPES = frozenset({"int", "float"})

    def _literal_tuple_annotation_node(self, iterable_node):
        """tuple[A, B, ...] element annotation of a list literal whose elements
        are all same-arity tuples of one sizable scalar type per position, else
        None.

        A key'd sorted() over a constant dict of tuple keys folds to such a
        literal, which carries no annotation of its own to read (#5444).
        """
        if not (isinstance(iterable_node, ast.List) and iterable_node.elts
                and all(isinstance(elt, ast.Tuple) for elt in iterable_node.elts)):
            return None
        arity = len(iterable_node.elts[0].elts)
        if arity == 0 or any(len(elt.elts) != arity for elt in iterable_node.elts):
            return None
        members = []
        for position in range(arity):
            names = {self._infer_type_from_value(elt.elts[position]) for elt in iterable_node.elts}
            if len(names) != 1 or not names <= self._SIZABLE_TUPLE_MEMBER_TYPES:
                return None
            members.append(ast.Name(id=names.pop(), ctx=ast.Load()))
        return ast.Subscript(value=ast.Name(id="tuple", ctx=ast.Load()),
                             slice=ast.Tuple(elts=members, ctx=ast.Load()),
                             ctx=ast.Load())

    def _full_tuple_annotation_node(self, iterable_node):
        """Return the full tuple[A, B, ...] element annotation of a
        list[tuple[...]]-annotated Name iterable, or None if unavailable.

        `_get_element_type_from_container` collapses a list[tuple[str, str]]
        element type down to the bare string "tuple" (it only ever returns a
        base-type name), which is enough for a scalar loop target but loses
        the tuple's concrete member types for a tuple-unpacking target
        (`for u, v in <list[tuple[...]]>`). This mirrors that lookup but keeps
        the full annotation node so the per-index element assignment stays
        typed as tuple[str, str] instead of eroding to bare tuple, whose
        element_0/element_1 members the C++ converter cannot size (#5444).
        """
        literal_ann = self._literal_tuple_annotation_node(iterable_node)
        if literal_ann is not None:
            return literal_ann
        if not (isinstance(iterable_node, ast.Name) and hasattr(self, "variable_annotations")):
            return None
        annotation = self.variable_annotations.get(iterable_node.id)
        if not isinstance(annotation, ast.Subscript):
            return None
        element_annotation = annotation.slice
        # Iterating a dict yields its keys: for a dict[K, V] annotation the
        # slice is an ast.Tuple of (K, V), so the element is K, not the slice
        # itself (`for u, v in g` over dict[tuple[str, str], int] unpacks a
        # tuple[str, str] key).
        if (self._get_base_type_name(annotation) in ("dict", "Dict")
                and isinstance(element_annotation, ast.Tuple)
                and len(element_annotation.elts) == 2):
            element_annotation = element_annotation.elts[0]
        return element_annotation if self._is_concrete_tuple_ann(element_annotation) else None

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
        # Tuple element types must keep their FULL annotation (list[tuple[A, B]]),
        # not flatten to the base name 'tuple', so the C++ list subscript read
        # rebuilds the concrete tuple struct instead of erasing it to void* and
        # crashing on the unpack (#5444). Other element types (dict, scalars)
        # keep the base-name form so get_typet(base) resolves correctly.
        if self._is_concrete_tuple_ann(elem_ann):
            slice_node = elem_ann
        else:
            actual_base = base_name if base_name and base_name != "Any" else "Any"
            slice_node = ast.Name(id=actual_base, ctx=ast.Load())
        annotation = ast.Subscript(
            value=ast.Name(id="list", ctx=ast.Load()),
            slice=slice_node,
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
        # ast2json serializes non-underscore attributes, so the converter can
        # tell this synthesized annotation apart from a user-written one
        # (an explicit `v: Any` must keep Any semantics; a synthesized one
        # must not override the rhs element type).
        assign.esbmc_synthesized = True
        self.ensure_all_locations(assign, node)
        return assign

    def _emit_items_unpack(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            self, node, body, target_elt, list_var, index_var, elem_ann):
        """Append the assignment that binds the i-th key/value of d.items() to ``target_elt``.

        Simple Name target — emits ``name: elem_ann = list_var[index_var]``.
        Tuple/List target (nested unpacking, e.g. ``for (u, v), w in d.items():``) —
        binds the element to a local tuple temp first, then destructures from that
        temp. Unpacking directly from ``list_var[index_var]`` lowers each component
        to ``(*(tuple*)elem->value).element_i``, which dereferences to an array
        rvalue for string components and aborts ("Can't construct rvalue reference
        to array type"). A whole-struct copy into a local makes each component read
        a local member access instead, reusing the working ``u, v = t`` path.
        """
        name = self._name_id_or_none(target_elt)
        if name is not None:
            body.append(self._create_var_subscript_assign(node, name, list_var, index_var,
                                                          elem_ann))
            return

        tmp_name = f"ESBMC_items_elt_{list_var}"
        # Keep the full tuple annotation (tuple[A, B]) for the temp when known,
        # so the key/value struct type is concrete instead of erased (#5444).
        tuple_ann = (elem_ann if self._is_concrete_tuple_ann(elem_ann) else ast.Name(
            id="tuple", ctx=ast.Load()))
        body.append(
            self._create_var_subscript_assign(node, tmp_name, list_var, index_var, tuple_ann))
        unpack = ast.Assign(
            targets=[target_elt],
            value=ast.Name(id=tmp_name, ctx=ast.Load()),
        )
        self.ensure_all_locations(unpack, node)
        body.append(unpack)

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
        return (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "zip"
                and len(it.args) >= 1 and not it.keywords)

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
        # A key'd sorted() has to be lowered here: this assignment is built
        # after the pass that lowers one, so the call would otherwise reach the
        # frontend with its key= dropped, and be refused.
        setup = []
        lowered = self._lower_sorted_key_iterable(seq)
        if lowered is not None:
            setup, seq = lowered

        annotation_id = self._get_iterable_type_annotation(seq)
        element_type = self._get_element_type_from_container(annotation_id, seq)
        if isinstance(seq, ast.Name):
            return seq.id, setup, element_type
        iter_var_name = f"ESBMC_iter_{loop_id}{suffix}"
        saved = node.iter
        node.iter = seq
        iter_assign = self._create_iter_assignment(node, annotation_id, iter_var_name, element_type)
        node.iter = saved
        return iter_var_name, setup + [iter_assign], element_type

    def _make_target_assign(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            self, node, target, iter_var_name, index_var, element_type):
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
        assign = ast.AnnAssign(target=ast.Name(id=name, ctx=ast.Store()),
                               annotation=ann,
                               value=current,
                               simple=1)
        self.ensure_all_locations(assign, node)
        out = [assign]
        if isinstance(target, (ast.Tuple, ast.List)):
            for i, elt in enumerate(target.elts):
                if not isinstance(elt, ast.Name):
                    continue
                unpack = ast.Assign(
                    targets=[ast.Name(id=elt.id, ctx=ast.Store())],
                    value=ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()),
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
            value=ast.BinOp(left=self.create_name_node(index_var, ast.Load(), node),
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
            value=ast.BinOp(left=self.create_name_node(length_var, ast.Load(), node),
                            op=ast.Sub(),
                            right=self.create_constant_node(1, node)),
            simple=1,
        )
        self.ensure_all_locations(index_assign, node)
        setup.append(index_assign)

        # while ESBMC_index >= 0:
        while_cond = ast.Compare(left=self.create_name_node(index_var, ast.Load(), node),
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

    def _transform_filter_for(self, node):  # pylint: disable=too-many-locals
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
            pred = ast.Call(func=copy.deepcopy(func),
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

    def _transform_zip_for(self, node):  # pylint: disable=too-many-locals
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
            call = ast.Call(func=self.create_name_node("len", ast.Load(), node),
                            args=[self.create_name_node(nm, ast.Load(), node)],
                            keywords=[])
            self.ensure_all_locations(call, node)
            return call

        length_expr = len_call(iter_names[0])
        for nm in iter_names[1:]:
            length_expr = ast.Call(func=self.create_name_node("min", ast.Load(), node),
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
                cur = ast.Subscript(value=ast.Name(id=nm, ctx=ast.Load()),
                                    slice=ast.Name(id=index_var, ctx=ast.Load()),
                                    ctx=ast.Load())
                ann = ast.Name(id=(et if et and et != "Any" else "Any"), ctx=ast.Load())
                assign = ast.AnnAssign(target=ast.Name(id=tgt.id, ctx=ast.Store()),
                                       annotation=ann,
                                       value=cur,
                                       simple=1)
                self.ensure_all_locations(assign, node)
                body.append(assign)
        else:
            # Single target variable receives a tuple of the parallel elements.
            name = self._name_id_or_none(target) or "ESBMC_loop_var"
            elts = [
                ast.Subscript(value=ast.Name(id=nm, ctx=ast.Load()),
                              slice=ast.Name(id=index_var, ctx=ast.Load()),
                              ctx=ast.Load()) for nm in iter_names
            ]
            tup = ast.Tuple(elts=elts, ctx=ast.Load())
            assign = ast.AnnAssign(target=ast.Name(id=name, ctx=ast.Store()),
                                   annotation=ast.Name(id="tuple", ctx=ast.Load()),
                                   value=tup,
                                   simple=1)
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

        # A key'd sorted() has to be lowered here: this loop's iterable
        # assignment is built after the pass that lowers one, so the call would
        # otherwise reach the frontend with its key= dropped, and be refused.
        # Lowered before the annotation is read, so the annotation describes
        # what the loop actually iterates -- a constant fold yields a list
        # literal, whose element type the sorted() call does not carry.
        scan_setup = []
        lowered = self._lower_sorted_key_iterable(node.iter)
        if lowered is not None:
            scan_setup, node.iter = lowered

        # Determine annotation type based on the iterable value
        annotation_id = self._get_iterable_type_annotation(node.iter)

        # Get element type for proper annotation
        element_type = self._get_element_type_from_container(annotation_id, node.iter)

        # Tuple-unpacking iteration over a dict's keys (for u, v in d): the keys
        # are tuples, so materialize d.keys() into an annotated
        # list[<key_ann>] intermediate first. That lets the C++ list subscript
        # read recover the concrete tuple element type from the list annotation
        # (the working list-of-tuples path) instead of erasing it to void* and
        # crashing on the unpack (#5444). Scoped to dicts whose key annotation
        # is concrete; a bare/unknown key type falls through to the existing
        # scalar-key handling below.
        # A single-name target whose body destructures it (`for edge in d:` then
        # `u, v = edge`) needs the concrete key type just as much as unpacking
        # at the target does: without it the key reads as Any and the later
        # unpack is handed a generic pointer it cannot destructure.
        target_destructured = (isinstance(node.target, (ast.Tuple, ast.List))
                               or self._body_destructures_name(node.body,
                                                               self._name_id_or_none(node.target)))

        # A destructured target over a list[tuple[...]] iterable (e.g. the
        # ESBMC_keys_N materialized just below, or any other list[tuple[...]]
        # variable) needs the full tuple[A, B, ...] annotation, not the bare
        # "tuple" name element_type collapses to -- otherwise the per-index
        # element assignment can't size element_0/element_1 (#5444).
        full_element_ann = None
        if target_destructured and element_type in ("tuple", "Tuple"):
            full_element_ann = self._full_tuple_annotation_node(node.iter)

        if (annotation_id in ["dict", "Dict"] and target_destructured
                and isinstance(node.iter, ast.Name)):
            key_ann, _ = self._get_dict_kv_types(node.iter.id)
            if self._get_base_type_name(key_ann) not in ("Any", None):
                keys_var = f"ESBMC_keys_{loop_id}"
                list_ann = ast.Subscript(value=ast.Name(id="list", ctx=ast.Load()),
                                         slice=key_ann,
                                         ctx=ast.Load())
                keys_assign = ast.AnnAssign(
                    target=ast.Name(id=keys_var, ctx=ast.Store()),
                    annotation=list_ann,
                    value=ast.Call(func=ast.Attribute(value=node.iter, attr="keys", ctx=ast.Load()),
                                   args=[],
                                   keywords=[]),
                    simple=1,
                )
                self.ensure_all_locations(keys_assign, node)
                ast.fix_missing_locations(keys_assign)
                self.variable_annotations[keys_var] = list_ann
                node.iter = ast.Name(id=keys_var, ctx=ast.Load())
                self.ensure_all_locations(node.iter, node)
                inner = self._transform_iterable_for(node)
                if not isinstance(inner, list):
                    inner = [inner]
                return [keys_assign] + inner

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
            setup_statements = list(scan_setup)
        else:
            # For other iterables (literals, calls, expressions), create ESBMC_iter copy
            iter_var_name = f"{iter_var_base}_{loop_id}"
            iter_assign = self._create_iter_assignment(node, annotation_id, iter_var_name,
                                                       element_type, full_element_ann)
            setup_statements = scan_setup + [iter_assign]

        # Create common setup statements (index and length) with unique names
        index_assign = self._create_index_assignment(node, index_var)
        length_assign = self._create_length_assignment(node, iter_var_name, length_var)
        setup_statements.extend([index_assign, length_assign])

        # Create while loop condition with unique variable names
        while_cond = self._create_while_condition(node, index_var, length_var)

        # Create loop body with unique variable names
        transformed_body = self._create_loop_body(node, target_var_name, iter_var_name, index_var,
                                                  element_type, full_element_ann)

        # Create the while statement
        while_stmt = ast.While(test=while_cond, body=transformed_body, orelse=[])
        self.ensure_all_locations(while_stmt, node)

        result = setup_statements + [while_stmt]

        # Ensure all nodes have proper location info
        for stmt in result:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        return result

    def _create_iter_assignment(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            self,
            node,
            annotation_id,
            iter_var_name,
            element_type,
            full_element_ann=None):
        """Create assignment for iterator variable with proper type annotation.

        ``full_element_ann`` carries the concrete tuple[A, B, ...] element
        annotation when one is known: the C++ list subscript read sizes the
        tuple from this list annotation, and the bare "tuple" name erases it to
        void* (#5444).
        """
        # str iterables (`for c in str(x)`, `for c in some_str_var`) must be
        # annotated as str so the converter lowers loop bounds via strlen
        # rather than the list-style get_object_size, which overshoots and
        # trips IndexError.
        if annotation_id == "str":
            iter_annotation = ast.Name(id="str", ctx=ast.Load())
        elif full_element_ann is not None:
            iter_annotation = ast.Subscript(
                value=ast.Name(id="list", ctx=ast.Load()),
                slice=copy.deepcopy(full_element_ann),
                ctx=ast.Load(),
            )
        # Create proper list[T] annotation instead of just 'list'
        elif element_type and element_type != "Any":
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
        # Any other annotation binds the constructor's __ESBMC_new_object
        # result to a non-pointer lvalue (#7083).
        elif annotation_id in getattr(self, "module_class_names", set()):
            iter_annotation = ast.Name(id=annotation_id, ctx=ast.Load())
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
        full_element_ann=None,
    ):
        """Create the body of the while loop with proper type annotations.

        ``full_element_ann``, when given, is the full tuple[A, B, ...]
        annotation node for the per-index element and takes precedence over
        the bare ``element_type`` name -- needed so a tuple-unpacking target
        keeps its concrete member types instead of eroding to bare tuple
        (#5444).
        """
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
            all_names = all(isinstance(elt, ast.Name) for elt in node.target.elts)
            if full_element_ann is not None and all_names:
                # Concrete tuple[A, B, ...] element: unpack with a real tuple
                # assignment (u, v = ESBMC_loop_var) so the C++ tuple_handler's
                # member-access unpacking (temp.element_i, sized from the
                # tuple struct components) is used instead of a per-index
                # constant Subscript read (ESBMC_loop_var[0]), which the
                # tuple-subscript path cannot size and erases to void (#5444).
                unpack_assign = ast.Assign(
                    targets=[node.target],
                    value=ast.Name(id=target_var_name, ctx=ast.Load()),
                )
                self.ensure_all_locations(unpack_assign, node)
                unpack_assigns.append(unpack_assign)
            else:
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
        if full_element_ann is not None:
            target_annotation = full_element_ann
        elif element_type and element_type != "Any":
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

        if func_name.startswith("_nondet_list_"):
            return "list"
        if func_name.startswith("_nondet_dict_"):
            return "dict"

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

    def _next_unpack_tmp_id(self):
        """Monotonic counter for staged-temp names emitted by tuple unpacking.

        Avoids using `id(source_node)` (CPython object id), which gets recycled
        after GC and yields non-stable names across runs. The counter lives on
        the preprocessor instance so it persists across visits within a single
        pass.
        """
        current = getattr(self, "_unpack_tmp_counter", 0)
        self._unpack_tmp_counter = current + 1
        return current

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

        # Python's `a, b = b, a % b` evaluates the RHS tuple first and then
        # binds each target, so a swap like `a, b = b, a` works. Lowering to
        # naive sequential `a = b; b = a % b` reads the *new* `a` in the
        # second assignment and is wrong (e.g. gcd loop terminates after one
        # iteration). Stage each RHS into a fresh temp before the binds, so
        # later target writes don't observe earlier ones.
        target_names = {tn.id for tn, _ in leaf_pairs if isinstance(tn, ast.Name)}

        def value_reads_target(value_node):
            for sub in ast.walk(value_node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id in target_names:
                        return True
            return False

        need_staging = bool(target_names) and any(value_reads_target(vn) for _, vn in leaf_pairs)
        if need_staging:
            base = self._next_unpack_tmp_id()
            staged_values = []
            for i, (_, value_node) in enumerate(leaf_pairs):
                tmp_name = f"ESBMC_unpack_tmp_{base}_{i}"
                tmp_assign = self._create_individual_assignment(
                    ast.Name(id=tmp_name, ctx=ast.Store()),
                    copy.deepcopy(value_node),
                    source_node,
                )
                assignments.append(tmp_assign)
                staged_name = ast.Name(id=tmp_name, ctx=ast.Load())
                self._copy_location_info(source_node, staged_name)
                staged_values.append(staged_name)
        else:
            staged_values = [copy.deepcopy(vn) for _, vn in leaf_pairs]

        for (target_node, value_node), staged in zip(leaf_pairs, staged_values):
            target_copy = copy.deepcopy(target_node)
            individual_assign = self._create_individual_assignment(target_copy, staged, source_node)
            self._update_variable_types_simple(target_copy, value_node)
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

    # Element/key/value types covered by the monomorphic builders in
    # models/nondet.py.  A type outside these is rejected rather than silently
    # degraded to int (esbmc/esbmc#7575).
    _NONDET_LIST_ELEM_TYPES = ("int", "float", "bool", "str")
    _NONDET_DICT_KEY_TYPES = ("int", "str", "bool")
    _NONDET_DICT_VALUE_TYPES = ("int", "float", "bool", "str")

    # Suffixes ESBMC synthesises a pointable stub for, so they can be passed as
    # first-class generators (converter_funcdef.cpp, nondet_stub_suffix).
    _NONDET_SCALAR_SUFFIXES = ("int", "float", "bool", "str", "char", "complex")

    # Mirrors `_DEFAULT_NONDET_SIZE` in models/nondet.py; applied here because
    # the rewrite always passes the bound explicitly.
    _DEFAULT_NONDET_COLLECTION_SIZE = 8

    @classmethod
    def _nondet_generator_type(cls, node):
        """Type name behind a scalar nondet generator, else None.

        Accepts a call (``nondet_bool()``) and the bare function reference
        SV-COMP passes (``nondet_int``).  Only the stubbed suffixes qualify, so
        an ordinary variable that happens to be named ``nondet_size`` is not
        mistaken for a generator and swallowed as the element type.
        """
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node, ast.Name):
            name = node.id
        else:
            return None
        for prefix in ("__VERIFIER_nondet_", "nondet_"):
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                if suffix in cls._NONDET_SCALAR_SUFFIXES:
                    return suffix
        return None

    @classmethod
    def _resolve_nondet_type(cls, node, allowed, func_name, role):
        expected = ", ".join(f"nondet_{t}()" for t in allowed)
        type_name = cls._nondet_generator_type(node)
        if type_name is None:
            raise SyntaxError(
                f"{func_name}: {role} must be a nondet generator, got "
                f"{ast.unparse(node)!r}; expected one of {expected}")
        if type_name not in allowed:
            raise SyntaxError(
                f"{func_name}: unsupported {role} 'nondet_{type_name}()'; "
                f"expected one of {expected}")
        return type_name

    @staticmethod
    def _nondet_collection_keywords(call, func_name):
        """Keyword arguments of a nondet collection call, keyed by name.

        A keyword this model does not know is rejected rather than dropped: a
        dropped one silently reverts the element type to int, which is the
        vacuous-proof failure mode of esbmc/esbmc#7575 arriving through a typo.
        ``**kwargs`` (``kw.arg is None``) hides the bound the same way.
        """
        accepted = ("max_size", "elem_type") if func_name == "nondet_list" \
            else ("max_size", "key_type", "value_type")
        keywords = {}
        for kw in call.keywords:
            if kw.arg not in accepted:
                given = repr(kw.arg) if kw.arg else "**kwargs"
                raise SyntaxError(
                    f"{func_name}: unexpected keyword argument {given}; "
                    f"accepts {', '.join(accepted)}")
            keywords[kw.arg] = kw.value
        return keywords

    def _parse_nondet_collection_call(self, call):
        """Resolve a ``nondet_list``/``nondet_dict`` call.

        Returns ``(builder_name, max_size_node)`` naming the monomorphic builder
        in models/nondet.py that produces the requested element types, or None
        when `call` is not one of these generators.
        """
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            return None
        func_name = call.func.id
        if func_name not in ("nondet_list", "nondet_dict"):
            return None
        if func_name in self.shadowed_nondet_collections:
            return None

        positional = list(call.args)
        # SV-COMP passes the generator first -- nondet_list(nondet_int),
        # nondet_dict(nondet_key, nondet_value) -- so a leading generator is an
        # element type, not a size, and the default bound stands.
        leading_generator = bool(positional) and \
            self._nondet_generator_type(positional[0]) is not None

        max_size_node = None
        if positional and not leading_generator:
            max_size_node = positional.pop(0)

        keywords = self._nondet_collection_keywords(call, func_name)
        max_size_node = keywords.get("max_size", max_size_node)
        if max_size_node is None:
            max_size_node = ast.Constant(value=self._DEFAULT_NONDET_COLLECTION_SIZE)

        def slot(index, kw_name):
            fallback = positional[index] if len(positional) > index else None
            return keywords.get(kw_name, fallback)

        def resolve(node, allowed, role):
            if node is None:
                return "int"
            return self._resolve_nondet_type(node, allowed, func_name, role)

        if func_name == "nondet_list":
            elem = resolve(slot(0, "elem_type"),
                           self._NONDET_LIST_ELEM_TYPES, "elem_type")
            return f"_nondet_list_{elem}", max_size_node

        key = resolve(slot(0, "key_type"),
                      self._NONDET_DICT_KEY_TYPES, "key_type")
        val = resolve(slot(1, "value_type"),
                      self._NONDET_DICT_VALUE_TYPES, "value_type")
        return f"_nondet_dict_{key}_{val}", max_size_node

    def _rewrite_nondet_collection_call(self, call):
        """Rewrite a nondet collection call to its monomorphic builder.

        The builder calls ``nondet_T()`` fresh per element, so indices hold
        independent values.  Rewriting the callee rather than expanding the call
        into statements keeps this valid in every expression position -- an
        annotated assignment, a return, a call argument (esbmc/esbmc#7575).
        """
        parsed = self._parse_nondet_collection_call(call)
        if parsed is None:
            return None
        builder, max_size_node = parsed
        self.ensure_all_locations(max_size_node, call)
        new_call = ast.Call(
            func=ast.Name(id=builder, ctx=ast.Load()),
            args=[max_size_node],
            keywords=[],
        )
        self.ensure_all_locations(new_call, call)
        ast.fix_missing_locations(new_call)
        return new_call

    @staticmethod
    def _nondet_builder_annotation(func_name):
        """``list[T]``/``dict[K, V]`` annotation for a rewritten builder name."""
        if func_name.startswith("_nondet_list_"):
            return ast.Subscript(
                value=ast.Name(id="list", ctx=ast.Load()),
                slice=ast.Name(id=func_name[len("_nondet_list_"):], ctx=ast.Load()),
                ctx=ast.Load(),
            )
        if func_name.startswith("_nondet_dict_"):
            key, _, val = func_name[len("_nondet_dict_"):].partition("_")
            return ast.Subscript(
                value=ast.Name(id="dict", ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[ast.Name(id=key, ctx=ast.Load()), ast.Name(id=val, ctx=ast.Load())],
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            )
        return None

    def _create_annotation_from_call(self, call_node):
        """Create annotation from a nondet collection call or its builder.

        Both spellings are handled because the module pre-pass runs before the
        callee rewrite and the per-assignment pass runs after it.
        """
        if not isinstance(call_node.func, ast.Name):
            return None
        annotation = self._nondet_builder_annotation(call_node.func.id)
        if annotation is not None:
            return annotation
        parsed = self._parse_nondet_collection_call(call_node)
        if parsed is None:
            return None
        return self._nondet_builder_annotation(parsed[0])

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
        """Return True if a dict literal needs per-key loop unrolling at iteration.

        Triggers on two cases:

        * Mixed key types (e.g. ``{"a": 1, 2: "b"}``) — ESBMC stores keys in a
          flat list with a type-specific byte size, so a single retrieval
          stride cannot read both an ``int`` (8 bytes) and a ``str`` (variable
          width) without tripping an array-bounds violation.

        * Any tuple key (e.g. ``{(1, 2): 10}``) — the key is a struct rather
          than a scalar, so when ``for k in d:`` is lowered to indexing the
          synthesised keys list, the loop variable's runtime value is a
          generic pointer.  Downstream destructuring (``u, v = k``) then fails
          in ``converter_stmt.cpp`` with "Cannot unpack pointer".  Unrolling
          inlines each tuple literal directly so the converter sees a struct.
        """
        if not dict_node.keys:
            return False
        key_types = [self._infer_dict_key_type(k) for k in dict_node.keys]
        if "tuple" in key_types:
            return True
        if len(dict_node.keys) < 2:
            return False
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

    def _unroll_het_for(self, node, typed_elts):  # pylint: disable=too-many-locals
        """Emit one typed assignment + one body copy per element.

        typed_elts — list of (type_str, ast_value_node) in iteration order.

        For a Name loop target the per-iteration symbol is renamed so ESBMC
        never holds two incompatible types in the same IR symbol.  For a
        Tuple/List loop target (e.g. ``for u, v in d:`` over a tuple-keyed
        dict) we assign each key into the destructuring pattern directly —
        the per-element types are uniform across iterations, so no rename
        is needed and the converter handles the unpack through its normal
        tuple-assignment path.
        """
        target = node.target

        if isinstance(target, (ast.Tuple, ast.List)):
            result = []
            for _type_str, value_node in typed_elts:
                assign = ast.Assign(
                    targets=[copy.deepcopy(target)],
                    value=copy.deepcopy(value_node),
                )
                self.ensure_all_locations(assign, node)
                ast.fix_missing_locations(assign)
                result.append(assign)
                for stmt in node.body:
                    result.append(copy.deepcopy(stmt))
            for stmt in result:
                ast.fix_missing_locations(stmt)
            return result

        target_name = self._name_id_or_none(target) or "ESBMC_het_var"

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
        if isinstance(key_node, ast.Tuple):
            return "tuple"
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
        elif self._is_nullary_lambda(factory_node):
            # Nullary lambda factory: emit the body expression directly so it
            # routes through the same dict-subscript-assignment path as a
            # literal. The C++ frontend cannot currently invoke
            # `(<lambda>)()` correctly — build_function_id only handles Name
            # and Attribute func types and otherwise resolves to the
            # enclosing function — so inlining the body avoids the misrouted
            # call entirely and is semantically identical for a thunk.
            factory_value = factory_node.body
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

    @staticmethod
    def _defaultdict_key_signature(key_node):
        """Return a stable signature for literal keys whose value is known."""
        if isinstance(key_node, ast.Constant):
            return ast.dump(key_node, include_attributes=False)
        if isinstance(key_node, ast.Tuple) and all(
                isinstance(elt, ast.Constant) for elt in key_node.elts):
            return ast.dump(key_node, include_attributes=False)
        return None

    def _is_defaultdict_key_initialized(self, dict_name, key_node):
        signature = self._defaultdict_key_signature(key_node)
        return (signature is not None
                and signature in self._defaultdict_initialized_keys.get(dict_name, set()))

    def _record_defaultdict_key_initialized(self, dict_name, key_node):
        signature = self._defaultdict_key_signature(key_node)
        if signature is not None:
            self._defaultdict_initialized_keys.setdefault(dict_name, set()).add(signature)

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
        is_key_initialized = self._is_defaultdict_key_initialized

        class _Lowerer(ast.NodeTransformer):

            def visit_Subscript(self, node):
                # Recurse into children first (handles nested subscripts).
                self.generic_visit(node)
                if not (isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name)
                        and node.value.id in defaultdict_factory):
                    return node
                dict_name = node.value.id
                if is_key_initialized(dict_name, node.slice):
                    return node
                factory = defaultdict_factory[dict_name]
                stmts, key_expr = make_missing_check(dict_name, node.slice, factory, template)
                all_inits.extend(stmts)
                node.slice = key_expr
                return node

        new_expr = _Lowerer().visit(expr)
        return all_inits, new_expr

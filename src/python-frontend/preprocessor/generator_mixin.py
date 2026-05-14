"""GeneratorMixin — extracted from preprocessor.

Holds the lowering of comprehensions / generator expressions and the
inlining of generator function calls. All shared state lives on
Preprocessor (set in Preprocessor.__init__); this mixin only adds
methods.
"""
import ast
import copy
import sys

class GeneratorMixin:
    def _lower_listcomp(self, node):
        """Lower a list comprehension into prefix statements and result expression.

        A comprehension with multiple `for` clauses is not a nested comprehension:
        it is semantically equivalent to nested for-loops:
            [f(i,j) for i in A for j in B]  =>  for i in A: for j in B: tmp.append(f(i,j))
        """
        for generator in node.generators:
            if len(getattr(generator, "ifs", [])) > 1:
                raise NotImplementedError(
                    "Only a single if-condition is supported in list comprehensions"
                )
            if getattr(generator, "is_async", False):
                raise NotImplementedError("Async list comprehensions are not supported")

        # Create a unique temporary list that will collect results.
        tmp_name = f"ESBMC_listcomp_{self.listcomp_counter}"
        self.listcomp_counter += 1
        self.known_variable_types[tmp_name] = "list"

        # Step 1: initialise the result list literal.
        init_assign = ast.Assign(
            targets=[self.create_name_node(tmp_name, ast.Store(), node)],
            value=ast.List(elts=[], ctx=ast.Load()),
        )
        self.ensure_all_locations(init_assign, node)
        ast.fix_missing_locations(init_assign)

        # Step 2: build the append expression that pushes each produced element.
        append_expr = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=self.create_name_node(tmp_name, ast.Load(), node),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[self.visit(node.elt)],
                keywords=[],
            )
        )
        self.ensure_all_locations(append_expr, node.elt)

        # Step 3: build nested for-loops from innermost generator outward.
        loop_body = [append_expr]
        for generator in reversed(node.generators):
            if generator.ifs:
                cond = self.visit(generator.ifs[0])
                self.ensure_all_locations(cond, generator.ifs[0])
                if_stmt = ast.If(test=cond, body=loop_body, orelse=[])
                self.ensure_all_locations(if_stmt, generator.ifs[0])
                ast.fix_missing_locations(if_stmt)
                loop_body = [if_stmt]
            for_stmt = ast.For(
                target=generator.target,
                iter=self.visit(generator.iter),
                body=loop_body,
                orelse=[],
            )
            self.ensure_all_locations(for_stmt, node)
            loop_body = [for_stmt]

        transformed_for = self.visit_For(loop_body[0])
        if not isinstance(transformed_for, list):
            transformed_for = [transformed_for]

        for stmt in transformed_for:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        # The comprehension evaluates to the temporary list, so expose it to callers.
        result_name = self.create_name_node(tmp_name, ast.Load(), node)
        self.ensure_all_locations(result_name, node)

        return [init_assign] + transformed_for, result_name
    def _isolate_genexp_targets(self, genexp_node):
        isolated = copy.deepcopy(genexp_node)

        for index, generator in enumerate(isolated.generators):
            if not isinstance(generator.target, ast.Name):
                continue

            old_name = generator.target.id
            new_name = f"ESBMC_gen_{self.listcomp_counter}_{old_name}"
            self.listcomp_counter += 1

            generator.target = ast.copy_location(
                ast.Name(id=new_name, ctx=ast.Store()), generator.target
            )
            generator.ifs = [
                self._rename_loads(cond, old_name, new_name) for cond in generator.ifs
            ]

            shadowed = False
            for later_generator in isolated.generators[index + 1 :]:
                # Comprehension iterables are evaluated before the later target is bound.
                later_generator.iter = self._rename_loads(
                    later_generator.iter, old_name, new_name
                )

                if old_name in self._bound_target_names(later_generator.target):
                    shadowed = True
                    break

                later_generator.ifs = [
                    self._rename_loads(cond, old_name, new_name)
                    for cond in later_generator.ifs
                ]

            if not shadowed:
                isolated.elt = self._rename_loads(isolated.elt, old_name, new_name)

        ast.fix_missing_locations(isolated)
        return isolated
    def _prepare_genexp(self, genexp_node):
        genexp_node = self._isolate_genexp_targets(genexp_node)

        for generator in genexp_node.generators:
            if len(getattr(generator, "ifs", [])) > 1:
                raise NotImplementedError(
                    "Only a single if-condition is supported in generator expressions"
                )
            if getattr(generator, "is_async", False):
                raise NotImplementedError(
                    "Async generator expressions are not supported"
                )

        return genexp_node
    def _build_reduction_guard(self, tmp_name, source_node, negated):
        guard_expr = self.create_name_node(tmp_name, ast.Load(), source_node)
        if negated:
            guard_expr = ast.UnaryOp(op=ast.Not(), operand=guard_expr)
        self.ensure_all_locations(guard_expr, source_node)
        ast.fix_missing_locations(guard_expr)
        return guard_expr
    def _build_genexp_for_body(self, generator, loop_body, guard, source_node):
        if not generator.ifs:
            guarded_body = ast.If(test=guard, body=loop_body, orelse=[])
            self.ensure_all_locations(guarded_body, source_node)
            ast.fix_missing_locations(guarded_body)
            return [guarded_body]

        cond_tmp_name = f"ESBMC_genif_{self.listcomp_counter}"
        self.listcomp_counter += 1
        self.known_variable_types[cond_tmp_name] = "bool"

        cond = self.visit(generator.ifs[0])
        self.ensure_all_locations(cond, generator.ifs[0])
        cond_init = self._create_bool_ann_assign(
            cond_tmp_name, self.create_constant_node(False, source_node), source_node
        )
        cond_update = self._create_bool_ann_assign(
            cond_tmp_name, cond, generator.ifs[0]
        )
        eval_cond = ast.If(test=guard, body=[cond_update], orelse=[])
        self.ensure_all_locations(eval_cond, source_node)
        ast.fix_missing_locations(eval_cond)
        run_body = ast.If(
            test=self.create_name_node(cond_tmp_name, ast.Load(), source_node),
            body=loop_body,
            orelse=[],
        )
        self.ensure_all_locations(run_body, source_node)
        ast.fix_missing_locations(run_body)
        return [cond_init, eval_cond, run_body]
    def _finalize_lowered_genexp(self, for_stmt, source_node):
        transformed_for = self.visit_For(for_stmt)
        if not isinstance(transformed_for, list):
            transformed_for = [transformed_for]

        for stmt in transformed_for:
            self.ensure_all_locations(stmt, source_node)
            ast.fix_missing_locations(stmt)

        return transformed_for
    def _lower_reduction_genexp(
        self, genexp_node, tmp_name, initial_value, reduction_stmt, negated_guard
    ):
        init_assign = ast.Assign(
            targets=[self.create_name_node(tmp_name, ast.Store(), genexp_node)],
            value=self.create_constant_node(initial_value, genexp_node),
        )
        self.ensure_all_locations(init_assign, genexp_node)
        ast.fix_missing_locations(init_assign)

        loop_body = [reduction_stmt]
        for generator in reversed(genexp_node.generators):
            guard = self._build_reduction_guard(tmp_name, genexp_node, negated_guard)
            for_body = self._build_genexp_for_body(
                generator, loop_body, guard, genexp_node
            )
            for_stmt = ast.For(
                target=generator.target,
                iter=self.visit(generator.iter),
                body=for_body,
                orelse=[],
            )
            self.ensure_all_locations(for_stmt, genexp_node)
            ast.fix_missing_locations(for_stmt)
            loop_body = [for_stmt]

        transformed_for = self._finalize_lowered_genexp(loop_body[0], genexp_node)
        result_name = self.create_name_node(tmp_name, ast.Load(), genexp_node)
        return [init_assign] + transformed_for, result_name
    def _lower_any_genexp(self, genexp_node):
        """Lower any(elt for target in iter [if cond]) to prefix stmts + boolean result.

        Transforms:
            any(elt for target in iter if cond)
        Into:
            ESBMC_any_N = False
            for target in iter:
                if cond:          # only when ifs are present
                    if elt:
                        ESBMC_any_N = True
            <result: ESBMC_any_N>
        """
        genexp_node = self._prepare_genexp(genexp_node)

        # if not ESBMC_any_N and elt: ESBMC_any_N = True
        # Guard with `not ESBMC_any_N` so the element expression is not
        # evaluated on remaining iterations once a truthy value is found,
        # approximating Python's short-circuit semantics without break.
        tmp_name = f"ESBMC_any_{self.listcomp_counter}"
        self.listcomp_counter += 1
        self.known_variable_types[tmp_name] = "bool"
        set_true = ast.Assign(
            targets=[self.create_name_node(tmp_name, ast.Store(), genexp_node)],
            value=self.create_constant_node(True, genexp_node),
        )
        self.ensure_all_locations(set_true, genexp_node)
        ast.fix_missing_locations(set_true)

        if_true = ast.If(test=self.visit(genexp_node.elt), body=[set_true], orelse=[])
        self.ensure_all_locations(if_true, genexp_node)
        ast.fix_missing_locations(if_true)

        return self._lower_reduction_genexp(genexp_node, tmp_name, False, if_true, True)
    def _lower_all_genexp(self, genexp_node):
        """Lower all(elt for target in iter [if cond]) to prefix stmts + boolean result.

        Transforms:
            all(elt for target in iter if cond)
        Into:
            ESBMC_all_N = True
            for target in iter:
                if cond:          # only when ifs are present
                    if ESBMC_all_N:
                        if not elt:
                            ESBMC_all_N = False
            <result: ESBMC_all_N>

        Uses guard-based short-circuit (checking ESBMC_all_N before evaluating)
        instead of break to avoid ESBMC's break+empty-list type inference issue.
        """
        genexp_node = self._prepare_genexp(genexp_node)

        # if ESBMC_all_N: if not elt: ESBMC_all_N = False
        # Guard-based short-circuit instead of break to avoid ESBMC break+empty-list bug.
        tmp_name = f"ESBMC_all_{self.listcomp_counter}"
        self.listcomp_counter += 1
        self.known_variable_types[tmp_name] = "bool"
        set_false = ast.Assign(
            targets=[self.create_name_node(tmp_name, ast.Store(), genexp_node)],
            value=self.create_constant_node(False, genexp_node),
        )
        self.ensure_all_locations(set_false, genexp_node)
        ast.fix_missing_locations(set_false)

        not_elt = ast.UnaryOp(op=ast.Not(), operand=self.visit(genexp_node.elt))
        self.ensure_all_locations(not_elt, genexp_node)
        ast.fix_missing_locations(not_elt)

        if_falsy = ast.If(test=not_elt, body=[set_false], orelse=[])
        self.ensure_all_locations(if_falsy, genexp_node)
        ast.fix_missing_locations(if_falsy)

        return self._lower_reduction_genexp(
            genexp_node, tmp_name, True, if_falsy, False
        )
    def _lower_list_gen_call(self, gen_call_node, parent_node):
        """Lower list(gen_func(args...)) to an inline list construction.

        Transforms list(gen_func(a, b)) into:
            param_0 = a
            param_1 = b
            ESBMC_list_gen_N: list = []
            # generator body with yield val -> ESBMC_list_gen_N.append(val)

        Returns (prefix_stmts, result_name_expr), or (None, gen_call_node) if
        inlining is not possible (body has return/yield-from, keyword args, or
        positional-arg count mismatch).
        """
        func_name = gen_call_node.func.id
        # Defensive: generator_funcs and generator_func_defs are kept in sync
        # by visit_FunctionDef, so this guard should never fire in practice.
        body_stmts = self.generator_func_defs.get(func_name)
        if body_stmts is None:
            return None, gen_call_node

        # Keyword arguments on the generator call are not handled.
        if gen_call_node.keywords:
            return None, gen_call_node

        # Generators with return or yield-from cannot be safely inlined at the
        # call site: 'return' would exit the enclosing scope, and 'yield from'
        # is not transformed by _YieldToAppend.
        if self._body_has_node_shallow(
            body_stmts, ast.Return
        ) or self._body_has_node_shallow(body_stmts, ast.YieldFrom):
            return None, gen_call_node

        # Emit parameter assignments so inlined body can reference them.
        param_names = self.functionParams.get(func_name, [])
        call_args = gen_call_node.args
        if len(param_names) != len(call_args):
            return None, gen_call_node

        result_var = f"ESBMC_list_gen_{self.listcomp_counter}"
        self.listcomp_counter += 1

        stmts = []
        for param, arg in zip(param_names, call_args):
            assign = ast.Assign(
                targets=[ast.Name(id=param, ctx=ast.Store())],
                value=copy.deepcopy(arg),
                type_comment=None,
            )
            self.ensure_all_locations(assign, parent_node)
            ast.fix_missing_locations(assign)
            stmts.append(assign)

        # result_var: list = []
        init = ast.AnnAssign(
            target=ast.Name(id=result_var, ctx=ast.Store()),
            annotation=ast.Name(id="list", ctx=ast.Load()),
            value=ast.List(elts=[], ctx=ast.Load()),
            simple=1,
        )
        self.ensure_all_locations(init, parent_node)
        ast.fix_missing_locations(init)
        stmts.append(init)

        # Replace every `yield val` in the body with `result_var.append(val)`.
        transformer = self._YieldToAppend(result_var, parent_node)
        for stmt in body_stmts:
            transformed = transformer.visit(copy.deepcopy(stmt))
            if isinstance(transformed, list):
                stmts.extend(transformed)
            else:
                stmts.append(transformed)

        for stmt in stmts:
            self.ensure_all_locations(stmt, parent_node)
            ast.fix_missing_locations(stmt)

        self.known_variable_types[result_var] = "list"

        result_expr = ast.Name(id=result_var, ctx=ast.Load())
        self.ensure_all_locations(result_expr, parent_node)
        ast.fix_missing_locations(result_expr)

        return stmts, result_expr
    def _has_early_return_before_yield(self, body):
        """Return True if body has a Return statement before any Yield (linear top-level scan)."""
        for stmt in body:
            if isinstance(stmt, ast.Return):
                return True
            if isinstance(stmt, ast.Expr) and isinstance(
                stmt.value, (ast.Yield, ast.YieldFrom)
            ):
                return False
        return False
    @staticmethod
    def _is_recursive_call(func_name, body):
        """Return True if any Call node in body has func.id == func_name."""
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == func_name
            ):
                return True
        return False
    def _transform_recursive_generator(self, node):
        """Transform a recursive generator function to accumulate-and-return.

        Converts:
            def f(args):
                ...
                yield val
                ...

        Into:
            def f(args) -> list:
                ESBMC_gen_result: list = []
                ...
                ESBMC_gen_result.append(val)
                ...
                return ESBMC_gen_result
        """
        result_var = "ESBMC_gen_result"
        template = node.body[0] if node.body else node

        # Annotate unannotated parameters as list[Any].  Without this the C++
        # annotator infers Any from the recursive call site (flatten(x) where
        # x: Any), which types the parameter as void*.  Subscripting void*
        # then crashes the index2t IR constructor.  Recursive generators always
        # recurse on a list-like iterable, so list[Any] is the right type.
        for arg in node.args.args:
            if arg.annotation is None:
                ann = ast.Subscript(
                    value=ast.Name(id="list", ctx=ast.Load()),
                    slice=ast.Name(id="Any", ctx=ast.Load()),
                    ctx=ast.Load(),
                )
                ast.copy_location(ann, template)
                ast.fix_missing_locations(ann)
                arg.annotation = ann

        # Add result list initialisation at the start of the body
        init = ast.AnnAssign(
            target=ast.Name(id=result_var, ctx=ast.Store()),
            annotation=ast.Name(id="list", ctx=ast.Load()),
            value=ast.List(elts=[], ctx=ast.Load()),
            simple=1,
        )
        ast.copy_location(init, template)
        ast.fix_missing_locations(init)

        # Replace all yield statements with append calls
        new_body = [
            self._YieldToAppend(result_var, template).visit(s) for s in node.body
        ]

        # Append return statement
        ret = ast.Return(value=ast.Name(id=result_var, ctx=ast.Load()))
        ast.copy_location(ret, template)
        ast.fix_missing_locations(ret)

        node.body = [init] + new_body + [ret]
        node.returns = ast.Name(id="list", ctx=ast.Load())
        ast.copy_location(node.returns, template)
        ast.fix_missing_locations(node.returns)
        return node
    def _inline_generator_for(self, node):
        """
        Inline a generator-based for loop.

        Transforms:
            for x in g:       # where g = gen_func()
                body

        Into the generator body with each `yield val` replaced by:
            x = val
            body

        Returns the list of inlined statements, or None if inlining is not possible.
        """

        if not isinstance(node.iter, ast.Name):
            return None
        gen_var = node.iter.id
        func_name = self.generator_vars.get(gen_var)
        if func_name is None:
            return None
        body_stmts = self.generator_func_defs.get(func_name)
        if body_stmts is None:
            return None

        if not hasattr(node.target, "id"):
            return None  # Only handle simple name targets
        target_name = node.target.id

        inlined = copy.deepcopy(body_stmts)
        replacer = self._YieldReplacer(target_name, node.body, node)
        result = []
        try:
            for stmt in inlined:
                out = replacer.visit(stmt)
                if isinstance(out, list):
                    result.extend(out)
                elif out is not None:
                    result.append(out)
        except NotImplementedError as e:
            print(
                f"warning: cannot inline generator '{func_name}': {e}", file=sys.stderr
            )
            return None

        for stmt in result:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        return result
    def _inline_generator_call_for(self, node):
        """Inline a for loop over a direct generator function call.

        Transforms:
            for x in gen_func(a, b):
                body

        Into the generator body with each `yield val` replaced by:
            param_0 = a
            param_1 = b
            x = val
            body

        Returns the list of inlined statements, or None if inlining is not
        possible (unknown generator, keyword args, arg count mismatch,
        non-simple loop target, or generator body contains return/yield-from).
        """

        gen_call = node.iter
        func_name = gen_call.func.id

        body_stmts = self.generator_func_defs.get(func_name)
        if body_stmts is None:
            return None

        if gen_call.keywords:
            return None

        # Generators with early return or yield-from cannot be safely inlined:
        # a bare `return` inlined into the enclosing scope would prematurely
        # exit it instead of just stopping the inner generator's iteration.
        if self._body_has_node_shallow(
            body_stmts, ast.Return
        ) or self._body_has_node_shallow(body_stmts, ast.YieldFrom):
            return None

        param_names = self.functionParams.get(func_name, [])
        call_args = gen_call.args
        if len(param_names) != len(call_args):
            return None

        if not hasattr(node.target, "id"):
            return None

        target_name = node.target.id

        stmts = []
        for param, arg in zip(param_names, call_args):
            assign = ast.Assign(
                targets=[ast.Name(id=param, ctx=ast.Store())],
                value=copy.deepcopy(arg),
                type_comment=None,
            )
            stmts.append(assign)

        inlined = copy.deepcopy(body_stmts)
        replacer = self._YieldReplacer(target_name, node.body, node)
        try:
            for stmt in inlined:
                out = replacer.visit(stmt)
                if isinstance(out, list):
                    stmts.extend(out)
                elif out is not None:
                    stmts.append(out)
        except NotImplementedError as e:
            print(
                f"warning: cannot inline generator '{func_name}': {e}", file=sys.stderr
            )
            return None

        for stmt in stmts:
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        return stmts
    @staticmethod
    def _has_yield(node):
        """Return True if node contains a Yield or YieldFrom expression."""
        return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))
    @staticmethod
    def _collect_post(stmts, start):
        """Collect stmts[start:] up to (not including) the first yield-containing statement."""
        post = []
        j = start
        while j < len(stmts):
            if GeneratorMixin._has_yield(stmts[j]):
                break
            post.append(stmts[j])
            j += 1
        return post, j
    def _find_generator_next_call(self, node):
        """Return (gen_var, func_name) if node contains next(g) for a tracked generator, else None."""
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "next"
                and len(child.args) == 1
                and isinstance(child.args[0], ast.Name)
            ):
                gen_var = child.args[0].id
                func_name = self.generator_vars.get(gen_var)
                if func_name is not None:
                    return (gen_var, func_name)
        return None
    def _collect_yields(self, stmts, in_loop=False):
        """
        Collect yield points from a generator body.

        Returns (outer_init, yields) where:
          outer_init : top-level statements before the first yield/loop-with-yield
                       (generator initialisation -- emitted once per generator var).
          yields     : list of (pre_stmts, yield_val, post_stmts, is_repeating)
            pre_stmts : statements inside the innermost scope before this yield.
                        For while-loop yields the first item is a guard:
                        `if not (loop_cond): raise StopIteration`
            yield_val : the yielded expression (may be an IfExp ternary for if/else)
            post_stmts: statements after this yield until the next yield
                        (e.g. `i += 1` after `yield i`)
            is_repeating: True when the yield is inside a loop
        """

        outer_init = []
        yields = []
        current_pre = []
        found_yield = False
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                post, j = self._collect_post(stmts, i + 1)
                yields.append((current_pre[:], stmt.value.value, post, in_loop))
                current_pre = []
                found_yield = True
                i = j
            elif isinstance(stmt, (ast.While, ast.For)):
                loop_init, loop_yields = self._collect_yields(stmt.body, in_loop=True)
                if loop_yields:
                    # For while loops, prepend a guard so _inline_next_call raises
                    # StopIteration when the loop condition becomes false.
                    if isinstance(stmt, ast.While):
                        guard = ast.If(
                            test=ast.UnaryOp(
                                op=ast.Not(), operand=copy.deepcopy(stmt.test)
                            ),
                            body=[self._make_stop_iteration_raise(stmt)],
                            orelse=[],
                        )
                        self.ensure_all_locations(guard, stmt)
                        ast.fix_missing_locations(guard)
                        loop_init = [guard] + loop_init
                    combined = loop_init + loop_yields[0][0]
                    _, iv, ipo, ir = loop_yields[0]
                    loop_yields[0] = (combined, iv, ipo, ir)
                    yields.extend(loop_yields)
                    current_pre = []
                    found_yield = True
                else:
                    if not found_yield:
                        outer_init.append(stmt)
                    else:
                        current_pre.append(stmt)
                i += 1
            elif isinstance(stmt, ast.If):
                if_init, if_yields = self._collect_yields(stmt.body, in_loop=in_loop)
                _, else_yields = (
                    self._collect_yields(stmt.orelse, in_loop=in_loop)
                    if stmt.orelse
                    else ([], [])
                )
                if if_yields and else_yields:
                    # Both branches yield -> combine into a ternary yield value
                    # and capture post_stmts from the outer scope.
                    _, if_val, _, _ = if_yields[0]
                    _, else_val, _, _ = else_yields[0]
                    ternary_val = ast.IfExp(
                        test=copy.deepcopy(stmt.test),
                        body=copy.deepcopy(if_val),
                        orelse=copy.deepcopy(else_val),
                    )
                    self.ensure_all_locations(ternary_val, stmt)
                    ast.fix_missing_locations(ternary_val)
                    post, j = self._collect_post(stmts, i + 1)
                    yields.append((current_pre[:], ternary_val, post, in_loop))
                    current_pre = []
                    found_yield = True
                    i = j
                elif if_yields:
                    # Only if-branch yields; also grab outer post_stmts.
                    combined = if_init + if_yields[0][0]
                    _, iv, ipo, ir = if_yields[0]
                    post, j = self._collect_post(stmts, i + 1)
                    if_yields[0] = (combined, iv, ipo + post, ir)
                    yields.extend(if_yields)
                    current_pre = []
                    found_yield = True
                    i = j
                else:
                    if not found_yield:
                        outer_init.append(stmt)
                    else:
                        current_pre.append(stmt)
                    i += 1
            else:
                if not found_yield:
                    outer_init.append(stmt)
                else:
                    current_pre.append(stmt)
                i += 1
        return outer_init, yields
    def _make_stop_iteration_raise(self, template_node):
        """Build `raise StopIteration('StopIteration')` AST node."""
        raise_node = ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="StopIteration", ctx=ast.Load()),
                args=[ast.Constant(value="StopIteration")],
                keywords=[],
            ),
            cause=None,
        )
        ast.copy_location(raise_node, template_node)
        ast.fix_missing_locations(raise_node)
        return raise_node
    def _inline_next_call(self, targets, func_name, gen_var, template_node):
        """
        Inline `x = next(g)` for a normal generator.

        Emits outer_init (generator initialisation) on the first call for
        gen_var, then per-call: pre_stmts + assignment + post_stmts.
        For yields inside loops (is_repeating=True) the index is not advanced.
        Pass targets=None for a standalone next(g) with no assignment target.
        Returns list of statements, or None if inlining is not possible.
        """

        body_stmts = self.generator_func_defs.get(func_name)
        if body_stmts is None:
            return None
        outer_init, yields = self._collect_yields(body_stmts)
        if not yields:
            return None

        idx = self.generator_next_index.get(gen_var, 0)
        if idx >= len(yields):
            return [self._make_stop_iteration_raise(template_node)]

        pre_stmts, yield_val, post_stmts, is_repeating = yields[idx]

        if not is_repeating:
            self.generator_next_index[gen_var] = idx + 1

        result = []
        # Emit init code once per generator variable
        if outer_init and gen_var not in self.generator_emitted_init:
            result.extend([copy.deepcopy(s) for s in outer_init])
            self.generator_emitted_init.add(gen_var)

        result.extend([copy.deepcopy(s) for s in pre_stmts])
        if targets is not None:
            assign = ast.Assign(
                targets=targets, value=copy.deepcopy(yield_val), type_comment=None
            )
            ast.copy_location(assign, template_node)
            ast.fix_missing_locations(assign)
            result.append(assign)
        result.extend([copy.deepcopy(s) for s in post_stmts])
        for stmt in result:
            self.ensure_all_locations(stmt, template_node)
            ast.fix_missing_locations(stmt)
        return result
    def _lower_listcomp_in_expr(self, expr):
        """Lower all list comprehensions inside an expression node.

        Returns (prefix_stmts, new_expr, result_type) where result_type
        is inferred from the transformed root expression.
        """
        if expr is None:
            return [], expr, "Any"
        lowerer = self._ListCompExpressionLowerer(self)
        new_expr = lowerer.visit(expr)
        result_type = self._infer_type_from_value(new_expr)
        return lowerer.statements, new_expr, result_type
    def _lower_sorted_with_key_call(self, call_node):
        """Lower sorted(iterable, key=lambda x: x[K]) for literal-list iterables."""
        if not (
            isinstance(call_node, ast.Call)
            and isinstance(call_node.func, ast.Name)
            and call_node.func.id == "sorted"
            and len(call_node.args) == 1
        ):
            return None

        key_kw = None
        for kw in call_node.keywords:
            if kw.arg == "key":
                if key_kw is not None:
                    return None
                key_kw = kw
            else:
                return None

        if key_kw is None or not isinstance(key_kw.value, ast.Lambda):
            return None

        key_lambda = key_kw.value
        if len(key_lambda.args.args) != 1:
            return None

        param_name = key_lambda.args.args[0].arg
        body = key_lambda.body
        if not (
            isinstance(body, ast.Subscript)
            and isinstance(body.value, ast.Name)
            and body.value.id == param_name
            and isinstance(body.slice, ast.Constant)
            and isinstance(body.slice.value, int)
            and body.slice.value >= 0
        ):
            return None

        key_index = body.slice.value
        iterable_expr = call_node.args[0]

        iterable_literal = None
        if isinstance(iterable_expr, ast.List):
            iterable_literal = iterable_expr
        elif isinstance(iterable_expr, ast.Name):
            iterable_literal = self.list_literal_values.get(iterable_expr.id)

        if iterable_literal is None:
            return None

        key_values = []
        for elt in iterable_literal.elts:
            if not (isinstance(elt, ast.Tuple) and key_index < len(elt.elts)):
                return None
            key_node = elt.elts[key_index]
            if not isinstance(key_node, ast.Constant):
                return None
            key_values.append(key_node.value)

        order = sorted(range(len(iterable_literal.elts)), key=lambda i: key_values[i])
        folded_sorted = ast.List(
            elts=[copy.deepcopy(iterable_literal.elts[i]) for i in order],
            ctx=ast.Load(),
        )
        self.ensure_all_locations(folded_sorted, call_node)
        ast.fix_missing_locations(folded_sorted)
        return [], folded_sorted
    def _lower_assert_eq_literal(self, test_node, source_node):
        # Disabled by default: this optimization introduced broad semantic/type
        # inference drift across regression suites. Keep original assert shape
        # unless explicitly enabled for focused experiments.
        if not getattr(self, "_enable_assert_eq_literal_lowering", False):
            return [], test_node

        if not (
            isinstance(test_node, ast.Compare)
            and len(test_node.ops) == 1
            and isinstance(test_node.ops[0], ast.Eq)
            and len(test_node.comparators) == 1
        ):
            return [], test_node

        left = test_node.left
        right = test_node.comparators[0]
        left = self._resolve_known_literal_expr(left)
        right = self._resolve_known_literal_expr(right)

        if self._is_assert_literal_shape(left) and self._is_assert_literal_shape(right):
            try:
                result = ast.literal_eval(left) == ast.literal_eval(right)
                return [], ast.Constant(value=result)
            except (ValueError, SyntaxError, TypeError):
                pass

        literal_node = None
        expr_node = None
        if self._is_assert_literal_shape(right) and self._is_pure_assert_expr(left):
            literal_node = right
            expr_node = left
        elif self._is_assert_literal_shape(left) and self._is_pure_assert_expr(right):
            literal_node = left
            expr_node = right
        else:
            return [], test_node

        # String equality lowering through a synthetic temporary has shown
        # semantic drift on dataclass attribute reads; keep native equality.
        if isinstance(literal_node, ast.Constant) and isinstance(
            literal_node.value, str
        ):
            return [], test_node

        # Keep non-trivial expressions untouched to avoid semantic/runtime drift
        # (e.g. subscripts/attributes that may involve model-specific lowering).
        if not isinstance(expr_node, ast.Name):
            return [], test_node

        tmp_name = f"__esbmc_assert_eq_tmp_{self._assert_eq_counter}"
        self._assert_eq_counter += 1
        tmp_assign = ast.Assign(
            targets=[ast.Name(id=tmp_name, ctx=ast.Store())],
            value=copy.deepcopy(expr_node),
        )
        self.ensure_all_locations(tmp_assign, source_node)

        tmp_load = ast.Name(id=tmp_name, ctx=ast.Load())
        checks = self._build_assert_literal_checks(tmp_load, literal_node, source_node)
        if not checks:
            return [], test_node
        if len(checks) == 1:
            new_test = checks[0]
        else:
            new_test = ast.BoolOp(op=ast.And(), values=checks)
            self.ensure_all_locations(new_test, source_node)
        ast.fix_missing_locations(new_test)
        return [tmp_assign], new_test
    def _hoist_generator_inits(self, body, template_node):
        """
        Scan a loop body for direct `var = next(gen_var)` assignments.
        For each normal generator whose outer_init hasn't been emitted yet,
        deep-copy the outer_init statements and return them (to be placed
        before the loop), and mark the generator as initialized so that
        _inline_next_call won't re-emit them inside the loop body.
        """

        pre_stmts = []
        for stmt in body:
            if not isinstance(stmt, ast.Assign):
                continue
            info = self._find_generator_next_call(stmt.value)
            if info is None:
                continue
            gen_var, func_name = info
            if func_name in self.early_return_generator_funcs:
                continue
            if gen_var in self.generator_emitted_init:
                continue
            body_stmts = self.generator_func_defs.get(func_name)
            if body_stmts is None:
                continue
            outer_init, _ = self._collect_yields(body_stmts)
            for s in outer_init:
                s_copy = copy.deepcopy(s)
                self.ensure_all_locations(s_copy, template_node)
                ast.fix_missing_locations(s_copy)
                pre_stmts.append(s_copy)
            self.generator_emitted_init.add(gen_var)
        return pre_stmts

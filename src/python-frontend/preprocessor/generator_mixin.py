"""GeneratorMixin — extracted from preprocessor.

Holds the lowering of comprehensions / generator expressions and the
inlining of generator function calls. All shared state lives on
Preprocessor (set in Preprocessor.__init__); this mixin only adds
methods.
"""
import ast
import copy
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
import sys


class GeneratorMixin:

    class _YieldToAppend(ast.NodeTransformer):

        def __init__(self, result_var, source_node):
            self.result_var = result_var
            self.source_node = source_node

        def visit_YieldFrom(self, node):
            raise NotImplementedError("yield from is not supported for generator inlining")

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Yield):
                append_expr = ast.Expr(value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=self.result_var, ctx=ast.Load()),
                        attr="append",
                        ctx=ast.Load(),
                    ),
                    args=[
                        node.value.value if node.value.value is not None else ast.Constant(
                            value=None)
                    ],
                    keywords=[],
                ))
                ast.copy_location(append_expr, node)
                ast.fix_missing_locations(append_expr)
                return append_expr
            return self.generic_visit(node)

    class _YieldReplacer(ast.NodeTransformer):

        def __init__(self, target_name, body_dest, source_node):
            self.target_name = target_name
            self.body_dest = body_dest
            self.source_node = source_node

        def visit_YieldFrom(self, node):
            raise NotImplementedError("yield from is not supported for generator inlining")

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Yield):
                assign = ast.Assign(
                    targets=[ast.Name(id=self.target_name, ctx=ast.Store())],
                    value=node.value.value if node.value.value is not None else ast.Constant(
                        value=None),
                )
                ast.copy_location(assign, node)
                ast.fix_missing_locations(assign)
                body_copy = [copy.deepcopy(stmt) for stmt in self.body_dest]
                for stmt in body_copy:
                    ast.copy_location(stmt, self.source_node)
                    ast.fix_missing_locations(stmt)
                return [assign] + body_copy
            return self.generic_visit(node)

    @staticmethod
    def _extract_min_max_key_index(key_lambda):
        if len(key_lambda.args.args) != 1:
            return None
        param_name = key_lambda.args.args[0].arg
        body = key_lambda.body
        if not (isinstance(body, ast.Subscript) and isinstance(body.value, ast.Name)
                and body.value.id == param_name and isinstance(body.slice, ast.Constant)
                and isinstance(body.slice.value, int) and body.slice.value >= 0):
            return None
        return body.slice.value

    def _resolve_list_literal_iterable(self, iterable_expr):
        if isinstance(iterable_expr, ast.List):
            return iterable_expr
        if isinstance(iterable_expr, ast.Name):
            return self.list_literal_values.get(iterable_expr.id)
        return None

    @staticmethod
    def _select_min_max_index(key_values, is_min):
        best_idx = 0
        for i in range(1, len(key_values)):
            if (is_min and key_values[i] < key_values[best_idx]) or (not is_min and key_values[i]
                                                                     > key_values[best_idx]):
                best_idx = i
        return best_idx

    def _lower_listcomp(self, node):
        """Lower a list comprehension into prefix statements and result expression.

        A comprehension with multiple `for` clauses is not a nested comprehension:
        it is semantically equivalent to nested for-loops:
            [f(i,j) for i in A for j in B]  =>  for i in A: for j in B: tmp.append(f(i,j))
        """
        for generator in node.generators:
            if len(getattr(generator, "ifs", [])) > 1:
                raise NotImplementedError(
                    "Only a single if-condition is supported in list comprehensions")
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
        append_expr = ast.Expr(value=ast.Call(
            func=ast.Attribute(
                value=self.create_name_node(tmp_name, ast.Load(), node),
                attr="append",
                ctx=ast.Load(),
            ),
            args=[self.visit(node.elt)],
            keywords=[],
        ))
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

            generator.target = ast.copy_location(ast.Name(id=new_name, ctx=ast.Store()),
                                                 generator.target)
            generator.ifs = [self._rename_loads(cond, old_name, new_name) for cond in generator.ifs]

            shadowed = False
            for later_generator in isolated.generators[index + 1:]:
                # Comprehension iterables are evaluated before the later target is bound.
                later_generator.iter = self._rename_loads(later_generator.iter, old_name, new_name)

                if old_name in self._bound_target_names(later_generator.target):
                    shadowed = True
                    break

                later_generator.ifs = [
                    self._rename_loads(cond, old_name, new_name) for cond in later_generator.ifs
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
                    "Only a single if-condition is supported in generator expressions")
            if getattr(generator, "is_async", False):
                raise NotImplementedError("Async generator expressions are not supported")

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
        cond_init = self._create_bool_ann_assign(cond_tmp_name,
                                                 self.create_constant_node(False, source_node),
                                                 source_node)
        cond_update = self._create_bool_ann_assign(cond_tmp_name, cond, generator.ifs[0])
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

    def _lower_reduction_genexp(self, genexp_node, reduction_stmt, state):
        # state bundles (tmp_name, initial_value, negated_guard) so the helper
        # stays under pylint's positional-argument limit.
        tmp_name, initial_value, negated_guard = state
        init_assign = ast.Assign(
            targets=[self.create_name_node(tmp_name, ast.Store(), genexp_node)],
            value=self.create_constant_node(initial_value, genexp_node),
        )
        self.ensure_all_locations(init_assign, genexp_node)
        ast.fix_missing_locations(init_assign)

        loop_body = [reduction_stmt]
        for generator in reversed(genexp_node.generators):
            guard = self._build_reduction_guard(tmp_name, genexp_node, negated_guard)
            for_body = self._build_genexp_for_body(generator, loop_body, guard, genexp_node)
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

        return self._lower_reduction_genexp(genexp_node, if_true, (tmp_name, False, True))

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

        return self._lower_reduction_genexp(genexp_node, if_falsy, (tmp_name, True, False))

    def _build_param_assigns(self, param_names, call_args, parent_node):
        """Build ``param = arg`` assignments used when inlining generator calls."""
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
        return stmts

    def _apply_yield_replacer(self, body_stmts, replacer_spec, func_name):
        """Inline a generator body, replacing ``yield val`` via ``_YieldReplacer``.

        ``replacer_spec`` is ``(target_name, body_dest, template_node)``.

        Returns the transformed statement list, or ``None`` if the body uses a
        construct ``_YieldReplacer`` cannot rewrite.
        """

        target_name, body_dest, template_node = replacer_spec
        inlined = copy.deepcopy(body_stmts)
        replacer = self._YieldReplacer(target_name, body_dest, template_node)
        result = []
        try:
            for stmt in inlined:
                out = replacer.visit(stmt)
                if isinstance(out, list):
                    result.extend(out)
                elif out is not None:
                    result.append(out)
        except NotImplementedError as e:
            print(f"warning: cannot inline generator '{func_name}': {e}", file=sys.stderr)
            return None
        return result

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
        if self._body_has_node_shallow(body_stmts, ast.Return) or self._body_has_node_shallow(
                body_stmts, ast.YieldFrom):
            return None, gen_call_node

        # Emit parameter assignments so inlined body can reference them.
        param_names = self.functionParams.get(func_name, [])
        if len(param_names) != len(gen_call_node.args):
            return None, gen_call_node

        result_var = f"ESBMC_list_gen_{self.listcomp_counter}"
        self.listcomp_counter += 1

        stmts = self._build_param_assigns(param_names, gen_call_node.args, parent_node)

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
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Yield, ast.YieldFrom)):
                return False
        return False

    @staticmethod
    def _is_recursive_call(func_name, body):
        """Return True if any Call node in body has func.id == func_name."""
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == func_name):
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
        new_body = [self._YieldToAppend(result_var, template).visit(s) for s in node.body]

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

        result = self._apply_yield_replacer(body_stmts, (node.target.id, node.body, node),
                                            func_name)
        if result is None:
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
        if self._body_has_node_shallow(body_stmts, ast.Return) or self._body_has_node_shallow(
                body_stmts, ast.YieldFrom):
            return None

        param_names = self.functionParams.get(func_name, [])
        if len(param_names) != len(gen_call.args):
            return None

        if not hasattr(node.target, "id"):
            return None

        stmts = self._build_param_assigns(param_names, gen_call.args, node)

        inlined_body = self._apply_yield_replacer(body_stmts, (node.target.id, node.body, node),
                                                  func_name)
        if inlined_body is None:
            return None
        stmts.extend(inlined_body)

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
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id == "next" and len(child.args) == 1
                    and isinstance(child.args[0], ast.Name)):
                gen_var = child.args[0].id
                func_name = self.generator_vars.get(gen_var)
                if func_name is not None:
                    return (gen_var, func_name)
        return None

    def _collect_loop_yields(self, stmt, yields):
        """Process a ``While``/``For`` body. Append yields and return ``True`` if any."""
        loop_init, loop_yields = self._collect_yields(stmt.body, in_loop=True)
        if not loop_yields:
            return False
        # For while loops, prepend a guard so _inline_next_call raises
        # StopIteration when the loop condition becomes false.
        if isinstance(stmt, ast.While):
            guard = ast.If(
                test=ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(stmt.test)),
                body=[self._make_stop_iteration_raise(stmt)],
                orelse=[],
            )
            self.ensure_all_locations(guard, stmt)
            ast.fix_missing_locations(guard)
            loop_init = [guard] + loop_init
        first_pre, iv, ipo, ir = loop_yields[0]
        loop_yields[0] = (loop_init + first_pre, iv, ipo, ir)
        yields.extend(loop_yields)
        return True

    def _make_ternary_yield_value(self, stmt, if_val, else_val):
        """Combine if/else yield values into a single ternary expression."""
        ternary_val = ast.IfExp(
            test=copy.deepcopy(stmt.test),
            body=copy.deepcopy(if_val),
            orelse=copy.deepcopy(else_val),
        )
        self.ensure_all_locations(ternary_val, stmt)
        ast.fix_missing_locations(ternary_val)
        return ternary_val

    def _absorb_if_only_yields(self, if_init, if_yields, post, yields):
        """Splice if-branch yields into the parent yields list, including outer post."""
        first_pre, iv, ipo, ir = if_yields[0]
        if_yields[0] = (if_init + first_pre, iv, ipo + post, ir)
        yields.extend(if_yields)

    def _collect_if_yields(self, stmts, i, accum, in_loop):
        """Process an ``If``. Returns the next index if yields were absorbed, else ``None``.

        ``accum`` is ``(current_pre, yields)``.
        """
        current_pre, yields = accum
        stmt = stmts[i]
        if_init, if_yields = self._collect_yields(stmt.body, in_loop=in_loop)
        else_yields = (self._collect_yields(stmt.orelse, in_loop=in_loop)[1] if stmt.orelse else [])
        if not if_yields:
            return None
        post, j = self._collect_post(stmts, i + 1)
        if else_yields:
            ternary = self._make_ternary_yield_value(stmt, if_yields[0][1], else_yields[0][1])
            yields.append((current_pre[:], ternary, post, in_loop))
        else:
            self._absorb_if_only_yields(if_init, if_yields, post, yields)
        return j

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
                continue
            if isinstance(stmt, (ast.While, ast.For)):
                if self._collect_loop_yields(stmt, yields):
                    current_pre = []
                    found_yield = True
                else:
                    (current_pre if found_yield else outer_init).append(stmt)
                i += 1
                continue
            if isinstance(stmt, ast.If):
                next_i = self._collect_if_yields(stmts, i, (current_pre, yields), in_loop)
                if next_i is not None:
                    current_pre = []
                    found_yield = True
                    i = next_i
                else:
                    (current_pre if found_yield else outer_init).append(stmt)
                    i += 1
                continue
            (current_pre if found_yield else outer_init).append(stmt)
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

    def _resolve_next_yield(self, gen_var, func_name):
        """Look up the next yield to inline for ``gen_var``.

        Returns ``(outer_init, entry)`` where ``entry`` is one of:
          * ``None``       -- generator unknown or yields nothing (cannot inline)
          * ``"stop"``     -- generator is exhausted; caller should raise StopIteration
          * tuple          -- ``(pre_stmts, yield_val, post_stmts, is_repeating)``
        """
        body_stmts = self.generator_func_defs.get(func_name)
        if body_stmts is None:
            return None, None
        outer_init, yields = self._collect_yields(body_stmts)
        if not yields:
            return None, None
        idx = self.generator_next_index.get(gen_var, 0)
        if idx >= len(yields):
            return outer_init, "stop"
        entry = yields[idx]
        if not entry[3]:  # not is_repeating
            self.generator_next_index[gen_var] = idx + 1
        return outer_init, entry

    def _inline_next_call(self, targets, func_name, gen_var, template_node):
        """
        Inline `x = next(g)` for a normal generator.

        Emits outer_init (generator initialisation) on the first call for
        gen_var, then per-call: pre_stmts + assignment + post_stmts.
        For yields inside loops (is_repeating=True) the index is not advanced.
        Pass targets=None for a standalone next(g) with no assignment target.
        Returns list of statements, or None if inlining is not possible.
        """

        outer_init, entry = self._resolve_next_yield(gen_var, func_name)
        if entry is None:
            return None
        if entry == "stop":
            return [self._make_stop_iteration_raise(template_node)]

        pre_stmts, yield_val, post_stmts, _ = entry

        result = []
        # Emit init code once per generator variable
        if outer_init and gen_var not in self.generator_emitted_init:
            result.extend([copy.deepcopy(s) for s in outer_init])
            self.generator_emitted_init.add(gen_var)

        result.extend([copy.deepcopy(s) for s in pre_stmts])
        if targets is not None:
            assign = ast.Assign(targets=targets, value=copy.deepcopy(yield_val), type_comment=None)
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

    def _lower_sorted_with_key_call(self, call_node):  # pylint: disable=too-many-branches
        """Lower sorted(iterable, key=lambda x: x[K]) for literal-list iterables."""
        if not (isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name)
                and call_node.func.id == "sorted" and len(call_node.args) == 1):
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
        if not (isinstance(body, ast.Subscript) and isinstance(body.value, ast.Name)
                and body.value.id == param_name and isinstance(body.slice, ast.Constant)
                and isinstance(body.slice.value, int) and body.slice.value >= 0):
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

    def _lower_min_max_with_key_call(self, call_node):  # pylint: disable=too-many-return-statements,too-many-locals,too-many-branches
        """Lower min/max(iterable, key=lambda x: x[K]) for literal-list iterables.

        Mirrors _lower_sorted_with_key_call: handles only the narrow pattern of
        a list literal of tuples plus a one-arg lambda body of the form
        ``param[K]`` with a constant integer index. Returns (prefix, expr) on
        success, or None when the pattern does not apply (caller falls back to
        the regular dispatch, which today drops the key= keyword).
        """
        if not (isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name)
                and call_node.func.id in ("min", "max") and len(call_node.args) == 1):
            return None

        key_kw = None
        default_kw = None
        for kw in call_node.keywords:
            if kw.arg == "key":
                if key_kw is not None:
                    return None
                key_kw = kw
            elif kw.arg == "default":
                # default= is honoured by the typed _default model variants
                # added in #4360; keep it on the call so the regular dispatch
                # forwards it.
                default_kw = kw
            else:
                return None

        if key_kw is None or not isinstance(key_kw.value, ast.Lambda):
            return None

        key_lambda = key_kw.value
        key_index = self._extract_min_max_key_index(key_lambda)
        if key_index is None:
            return None
        iterable_expr = call_node.args[0]
        iterable_literal = self._resolve_list_literal_iterable(iterable_expr)

        if iterable_literal is None:
            return None

        if not iterable_literal.elts:
            # Empty iterable — defer to the regular dispatch so the empty
            # case (default= or ValueError) is handled uniformly.
            return None

        key_values = []
        for elt in iterable_literal.elts:
            if not (isinstance(elt, ast.Tuple) and key_index < len(elt.elts)):
                return None
            key_node = elt.elts[key_index]
            if not isinstance(key_node, ast.Constant):
                return None
            key_values.append(key_node.value)

        is_min = call_node.func.id == "min"
        # Pick the index whose key is the minimum / maximum, breaking ties
        # toward the first occurrence (matches CPython semantics).
        best_idx = self._select_min_max_index(key_values, is_min)

        # Suppress the unused default_kw warning while keeping the variable
        # available for future extension (e.g. empty iterable + default=).
        del default_kw

        result = copy.deepcopy(iterable_literal.elts[best_idx])
        self.ensure_all_locations(result, call_node)
        ast.fix_missing_locations(result)
        return [], result

    def _lower_tuple_sorted_pair_call(self, call_node):  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
        """Lower tuple(sorted([a, b])) to a conditional pair assignment.

        Instead of ``(a, b) if a <= b else (b, a)`` (which ESBMC encodes as a
        pointer to a temporary struct — a known crash pattern), we emit:

            _lo = a if a <= b else b
            _hi = b if a <= b else a
            (_lo, _hi)

        The result is a 2-tuple whose elements are plain scalar variables.
        ESBMC can handle named-scalar tuple construction without the
        pointer-to-temporary issue.
        """
        if not (isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name)
                and call_node.func.id == "tuple" and len(call_node.args) == 1
                and not call_node.keywords):
            return None

        sorted_call = call_node.args[0]
        if not (isinstance(sorted_call, ast.Call) and isinstance(sorted_call.func, ast.Name)
                and sorted_call.func.id == "sorted" and len(sorted_call.args) == 1
                and not sorted_call.keywords):
            return None

        iterable = sorted_call.args[0]
        if not (isinstance(iterable, ast.List) and len(iterable.elts) == 2):
            return None

        left = iterable.elts[0]
        right = iterable.elts[1]

        # Avoid duplicating side effects by only rewriting pure expressions.
        if not (self._is_pure_assert_expr(left) and self._is_pure_assert_expr(right)):
            return None

        # Produce scalar temporaries and fill them via an explicit if/else
        # (instead of IfExp) to avoid irep2 branch-type mismatches.
        counter = self.listcomp_counter
        self.listcomp_counter += 1
        lo_name = f"ESBMC_sorted_lo_{counter}"
        hi_name = f"ESBMC_sorted_hi_{counter}"

        cond = ast.Compare(
            left=copy.deepcopy(left),
            ops=[ast.LtE()],
            comparators=[copy.deepcopy(right)],
        )
        ast.copy_location(cond, call_node)
        ast.fix_missing_locations(cond)

        # Try to determine the element type so ESBMC can type the temporaries
        # correctly (e.g. float instead of void*).
        def _infer_scalar_type(node):
            if isinstance(node, ast.Constant):
                return type(node.value).__name__
            if isinstance(node, ast.Name):
                ann = self.variable_annotations.get(node.id)
                if ann is not None and isinstance(ann, ast.Name):
                    return ann.id
            return None

        elem_type = _infer_scalar_type(left) or _infer_scalar_type(right)
        if elem_type not in {"int", "float", "bool"}:
            elem_type = None

        lo_store = ast.Name(id=lo_name, ctx=ast.Store())
        ast.copy_location(lo_store, call_node)
        if elem_type:
            lo_assign = ast.AnnAssign(
                target=lo_store,
                annotation=ast.Name(id=elem_type, ctx=ast.Load()),
                value=copy.deepcopy(left),
                simple=1,
            )
        else:
            lo_assign = ast.Assign(targets=[lo_store], value=copy.deepcopy(left), type_comment=None)
        ast.copy_location(lo_assign, call_node)
        ast.fix_missing_locations(lo_assign)

        hi_store = ast.Name(id=hi_name, ctx=ast.Store())
        ast.copy_location(hi_store, call_node)
        if elem_type:
            hi_assign = ast.AnnAssign(
                target=hi_store,
                annotation=ast.Name(id=elem_type, ctx=ast.Load()),
                value=copy.deepcopy(right),
                simple=1,
            )
        else:
            hi_assign = ast.Assign(targets=[hi_store],
                                   value=copy.deepcopy(right),
                                   type_comment=None)
        ast.copy_location(hi_assign, call_node)
        ast.fix_missing_locations(hi_assign)

        then_lo = ast.Assign(
            targets=[ast.Name(id=lo_name, ctx=ast.Store())],
            value=copy.deepcopy(left),
            type_comment=None,
        )
        then_hi = ast.Assign(
            targets=[ast.Name(id=hi_name, ctx=ast.Store())],
            value=copy.deepcopy(right),
            type_comment=None,
        )
        else_lo = ast.Assign(
            targets=[ast.Name(id=lo_name, ctx=ast.Store())],
            value=copy.deepcopy(right),
            type_comment=None,
        )
        else_hi = ast.Assign(
            targets=[ast.Name(id=hi_name, ctx=ast.Store())],
            value=copy.deepcopy(left),
            type_comment=None,
        )
        for stmt in (then_lo, then_hi, else_lo, else_hi):
            ast.copy_location(stmt, call_node)
            ast.fix_missing_locations(stmt)

        cond_stmt = ast.If(test=copy.deepcopy(cond),
                           body=[then_lo, then_hi],
                           orelse=[else_lo, else_hi])
        ast.copy_location(cond_stmt, call_node)
        ast.fix_missing_locations(cond_stmt)

        result_tuple = ast.Tuple(
            elts=[
                ast.Name(id=lo_name, ctx=ast.Load()),
                ast.Name(id=hi_name, ctx=ast.Load()),
            ],
            ctx=ast.Load(),
        )
        self.ensure_all_locations(result_tuple, call_node)
        ast.fix_missing_locations(result_tuple)

        return [lo_assign, hi_assign, cond_stmt], result_tuple

    def _maybe_rewrite_list_sort_with_key(self, expr_node):
        """If expr_node is `name.sort(key=lambda ...)` (with optional reverse=),
        rewrite to `name = sorted(name, key=..., reverse=...)`. Returns the
        replacement Assign, or None when the pattern does not apply."""
        call = expr_node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "sort" and isinstance(call.func.value, ast.Name)
                and not call.args):
            return None
        has_key = any(kw.arg == "key" for kw in call.keywords)
        if not has_key:
            return None  # plain reverse= keeps today's path
        target_name = call.func.value.id
        sorted_call = ast.Call(
            func=ast.Name(id="sorted", ctx=ast.Load()),
            args=[ast.Name(id=target_name, ctx=ast.Load())],
            keywords=[copy.deepcopy(kw) for kw in call.keywords],
        )
        assign = ast.Assign(
            targets=[ast.Name(id=target_name, ctx=ast.Store())],
            value=sorted_call,
        )
        ast.copy_location(sorted_call, expr_node)
        ast.copy_location(assign, expr_node)
        ast.fix_missing_locations(assign)
        return self.visit(assign)

    def _lower_assert_eq_literal(self, test_node, source_node):
        # Disabled by default: this optimization introduced broad semantic/type
        # inference drift across regression suites. Keep original assert shape
        # unless explicitly enabled for focused experiments.
        if not getattr(self, "_enable_assert_eq_literal_lowering", False):
            return [], test_node

        if not (isinstance(test_node, ast.Compare) and len(test_node.ops) == 1
                and isinstance(test_node.ops[0], ast.Eq) and len(test_node.comparators) == 1):
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
        if isinstance(literal_node, ast.Constant) and isinstance(literal_node.value, str):
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

    def _try_lower_expr_tuple_literal_eq(self, expr_side, tuple_side, source_node):
        """Lower ``expr == (c0, c1, ...)`` where *expr* is not a bare Name.

        Instead of a struct-to-struct equality (which requires identical Z3
        sorts and fails when the function-return struct type differs from the
        literal tuple struct type), we **unpack** the tuple and compare each
        element individually:

            _u0, _u1, ... = expr
            assert _u0 == c0 and _u1 == c1 and ...

        Tuple unpacking is implemented in ESBMC via struct member access
        (``element_0``, ``element_1``, ...), so each unpacked variable carries
        the correct element type (e.g. ``double_floatbv``).  Element-wise scalar
        comparisons then avoid the struct-sort mismatch entirely.

        Returns ``(prefix_stmts, new_test)`` when the pattern matches, or
        ``(None, None)`` when it does not.  Only applies when *tuple_side* is a
        tuple literal whose elements are all ``_is_assert_literal_shape`` values
        and *expr_side* is not already a ``Name``.
        """
        if isinstance(expr_side, ast.Name):
            return None, None
        if not isinstance(tuple_side, ast.Tuple) or not tuple_side.elts:
            return None, None
        if not all(self._is_assert_literal_shape(e) for e in tuple_side.elts):
            return None, None

        n = len(tuple_side.elts)
        counter = self.listcomp_counter
        self.listcomp_counter += 1

        # Generate unique names for the unpacked elements.
        unpack_names = [f"ESBMC_assert_unpack_{counter}_{i}" for i in range(n)]

        # Build: ESBMC_assert_unpack_N_0, ESBMC_assert_unpack_N_1, ... = expr
        unpack_targets = [ast.Name(id=name, ctx=ast.Store()) for name in unpack_names]
        unpack_target_tuple = ast.Tuple(elts=unpack_targets, ctx=ast.Store())
        ast.copy_location(unpack_target_tuple, source_node)

        unpack_assign = ast.Assign(
            targets=[unpack_target_tuple],
            value=copy.deepcopy(expr_side),
            type_comment=None,
        )
        ast.copy_location(unpack_assign, source_node)
        ast.fix_missing_locations(unpack_assign)

        # Build element-wise comparisons: _u0 == c0 and _u1 == c1 ...
        comparisons = []
        for i, elt in enumerate(tuple_side.elts):
            cmp = ast.Compare(
                left=ast.Name(id=unpack_names[i], ctx=ast.Load()),
                ops=[ast.Eq()],
                comparators=[copy.deepcopy(elt)],
            )
            ast.copy_location(cmp, source_node)
            ast.fix_missing_locations(cmp)
            comparisons.append(cmp)

        if len(comparisons) == 1:
            new_test = comparisons[0]
        else:
            new_test = ast.BoolOp(op=ast.And(), values=comparisons)
            ast.copy_location(new_test, source_node)
            ast.fix_missing_locations(new_test)

        return [unpack_assign], new_test

    def _is_assert_literal_shape(self, node):
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (str, int, float, bool, type(None)))
        if isinstance(node, (ast.List, ast.Tuple)):
            return all(self._is_assert_literal_shape(elt) for elt in node.elts)
        return False

    def _resolve_known_literal_expr(self, node):
        if isinstance(node, ast.Name) and node.id in self._known_literal_values:
            return copy.deepcopy(self._known_literal_values[node.id])

        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id in self._known_literal_values
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)):
            base = self._known_literal_values[node.value.id]
            idx = node.slice.value
            if isinstance(base, (ast.List, ast.Tuple)) and 0 <= idx < len(base.elts):
                return copy.deepcopy(base.elts[idx])

        return node

    def _is_pure_assert_expr(self, node):
        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Attribute):
            return self._is_pure_assert_expr(node.value)
        if isinstance(node, ast.Subscript):
            return self._is_pure_assert_expr(node.value) and self._is_assert_literal_shape(
                node.slice)
        return isinstance(node, (ast.List, ast.Tuple)) and all(
            self._is_pure_assert_expr(elt) or self._is_assert_literal_shape(elt)
            for elt in node.elts)

    def _build_assert_literal_checks(self, actual_expr, literal_node, source_node):
        if isinstance(literal_node, ast.Constant):
            if isinstance(literal_node.value, str):
                cmp_node = ast.Compare(
                    left=copy.deepcopy(actual_expr),
                    ops=[ast.Eq()],
                    comparators=[copy.deepcopy(literal_node)],
                )
                self.ensure_all_locations(cmp_node, source_node)
                return [cmp_node]
            cmp_node = ast.Compare(
                left=copy.deepcopy(actual_expr),
                ops=[ast.Eq()],
                comparators=[copy.deepcopy(literal_node)],
            )
            self.ensure_all_locations(cmp_node, source_node)
            return [cmp_node]

        if not isinstance(literal_node, (ast.List, ast.Tuple)):
            return None

        checks = [
            ast.Compare(
                left=ast.Call(
                    func=self.create_name_node("len", ast.Load(), source_node),
                    args=[copy.deepcopy(actual_expr)],
                    keywords=[],
                ),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=len(literal_node.elts))],
            )
        ]
        self.ensure_all_locations(checks[0], source_node)

        for idx, elt in enumerate(literal_node.elts):
            sub = ast.Subscript(
                value=copy.deepcopy(actual_expr),
                slice=ast.Constant(value=idx),
                ctx=ast.Load(),
            )
            self.ensure_all_locations(sub, source_node)
            sub_checks = self._build_assert_literal_checks(sub, elt, source_node)
            if sub_checks is None:
                return None
            checks.extend(sub_checks)
        return checks

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

    def _get_dict_expr_from_items_call(self, call_node):
        """If call_node is d.items() on a known dict, return the dict expression. Else None."""
        if not (isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Attribute)
                and call_node.func.attr == "items" and not call_node.args
                and not getattr(call_node, "keywords", [])):
            return None
        base = call_node.func.value
        if isinstance(base, ast.Name):
            known_type = self.known_variable_types.get(base.id)
            if known_type is not None and known_type != "dict":
                return None
        return base

    def _get_items_dict_expr(self, node, wrappers):
        """Return (dict_expr, wrapper) if node is W(X) where W in `wrappers` and X is a dict_items source.

        Returns (None, None) on no match. The wrapper name lets the caller
        apply soundness checks that depend on which wrapper is in use
        (list/sorted/set differ in ordering semantics). ``sorted(list(...))``
        folds to ``sorted(...)`` (list() is identity on a view) so the
        cascade matches both idioms.
        """
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id
                in wrappers and len(node.args) == 1 and not getattr(node, "keywords", [])):
            return None, None
        wrapper = node.func.id
        arg = node.args[0]
        if wrapper == "sorted":
            if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                    and arg.func.id == "list" and len(arg.args) == 1
                    and not getattr(arg, "keywords", [])):
                arg = arg.args[0]
        if isinstance(arg, ast.Name) and arg.id in self.dict_items_vars:
            return self.dict_items_vars[arg.id], wrapper
        dict_expr = self._get_dict_expr_from_items_call(arg)
        return (dict_expr, wrapper) if dict_expr is not None else (None, None)

    @staticmethod
    def _all_constants_distinct(elts):
        """Return True iff all elts are ast.Constant with statically distinct hashable values."""
        if not all(isinstance(e, ast.Constant) for e in elts):
            return False
        values = [e.value for e in elts]
        try:
            return len(set(values)) == len(values)
        except TypeError:
            return False

    @staticmethod
    def _is_sorted_const_list(elts, *, by="self"):
        """Return True iff elts is statically known to be sorted (ascending).

        by="self": each elt is ast.Constant, sort by elt.value.
        by="first": each elt is ast.Tuple of Constants, sort by first element.

        Returns False if any elt is non-Constant or has incomparable types
        (conservative — bail when sortedness cannot be statically verified).
        """
        if len(elts) <= 1:
            return True
        keys = []
        for e in elts:
            if by == "self":
                if not isinstance(e, ast.Constant):
                    return False
                keys.append(e.value)
            else:
                if not (isinstance(e, ast.Tuple) and e.elts
                        and isinstance(e.elts[0], ast.Constant)):
                    return False
                keys.append(e.elts[0].value)
        try:
            return all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))
        except TypeError:
            return False

    def _get_dict_view_call(self, node, attr):
        """If node is W(d.<attr>()) where W in {list, sorted, set}, return (dict_expr, wrapper).

        Returns (None, None) on no match. The wrapper name lets the caller
        enforce wrapper-vs-literal-type compatibility (e.g. set wrapper
        requires a Set literal). ``sorted(list(...))`` is peeled to
        ``sorted(...)`` since list() is identity on a view.
        """
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("list", "sorted", "set")
                and len(node.args) == 1 and not getattr(node, "keywords", [])):
            return None, None
        wrapper = node.func.id
        inner = node.args[0]
        if wrapper == "sorted":
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                    and inner.func.id == "list" and len(inner.args) == 1
                    and not getattr(inner, "keywords", [])):
                inner = inner.args[0]
        if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == attr and not inner.args
                and not getattr(inner, "keywords", [])):
            return None, None
        base = inner.func.value
        if isinstance(base, ast.Name):
            known_type = self.known_variable_types.get(base.id)
            if known_type is not None and known_type != "dict":
                return None, None
        return base, wrapper

    def _try_transform_items_set_eq(self, set_side, literal_side, source_node):
        """Transform set(d.items()) == {(k,v),...} into dict membership checks.

        Rewrites to: set(d.keys()) == {k,...} and d[k1] == v1 and d[k2] == v2 ...
        This avoids tuple struct comparison and uses only proven-working primitives.
        Returns the new AST node, or None if the pattern doesn't match.
        """
        dict_expr, _ = self._get_items_dict_expr(set_side, ("set",))
        if dict_expr is None:
            return None
        if not isinstance(literal_side, ast.Set) or not literal_side.elts:
            return None
        pairs = []
        for elt in literal_side.elts:
            if not (isinstance(elt, ast.Tuple) and len(elt.elts) == 2):
                return None
            pairs.append((elt.elts[0], elt.elts[1]))
        # Pair keys must be statically distinct Constants. Dict keys are
        # distinct, and set literals dedupe by value at runtime; without
        # this guard, e.g. {(1,'a'),(1,'a')} (CPython len 1) would rewrite
        # to len(d)==2, turning a True assertion on {1:'a'} into False.
        if not self._all_constants_distinct([k for k, _ in pairs]):
            return None

        # Avoid set-equality backend path: prove same keys via size + membership.
        len_eq = ast.Compare(
            left=ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[copy.deepcopy(dict_expr)],
                keywords=[],
            ),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=len(pairs))],
        )

        # Build: (k in d) and d[k] == v for each pair.
        value_checks = [len_eq]
        for k, v in pairs:
            key_in_dict = ast.Compare(
                left=copy.deepcopy(k),
                ops=[ast.In()],
                comparators=[copy.deepcopy(dict_expr)],
            )
            subscript = ast.Subscript(value=dict_expr, slice=k, ctx=ast.Load())
            val_eq = ast.Compare(left=subscript, ops=[ast.Eq()], comparators=[v])
            value_checks.append(key_in_dict)
            value_checks.append(val_eq)

        result = ast.BoolOp(op=ast.And(), values=value_checks)
        self.ensure_all_locations(result, source_node)
        ast.fix_missing_locations(result)
        return result

    def _try_transform_keys_view_eq(self, view_side, literal_side, source_node):
        """Transform W(d.keys()) == literal into membership checks, where W is one of
        list/sorted/set.

        Rewrites to ``len(d) == N and k1 in d and k2 in d ...`` — set-equality
        semantics. Soundness guards (each bails when violated):
        - wrapper-vs-literal type: set ↔ ast.Set; list/sorted ↔ ast.List
          (CPython makes ``list == set`` always False otherwise).
        - literal keys must be statically distinct Constants. Dict keys are
          distinct, so any literal with duplicate keys mismatches CPython
          semantics: for set wrapper, ``{1,1}`` dedupes to ``{1}`` (length
          mismatch); for list/sorted, ``[1,1]`` never equals any view of a
          real dict. Bailing here is sound.
        - sorted wrapper: literal must be statically sorted ascending
          (otherwise CPython is always False but the rewrite would say True).
        - list wrapper: literal must have at most one element
          (ESBMC's dict model does not preserve insertion order).
        """
        dict_expr, wrapper = self._get_dict_view_call(view_side, "keys")
        if dict_expr is None:
            return None
        if wrapper == "set":
            if not isinstance(literal_side, ast.Set):
                return None
        else:
            if not isinstance(literal_side, ast.List):
                return None
        keys = list(literal_side.elts)
        if keys and not self._all_constants_distinct(keys):
            return None
        if wrapper == "sorted" and not self._is_sorted_const_list(keys):
            return None
        if wrapper == "list" and len(keys) > 1:
            return None
        len_eq = ast.Compare(
            left=ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[copy.deepcopy(dict_expr)],
                keywords=[],
            ),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=len(keys))],
        )
        if not keys:
            self.ensure_all_locations(len_eq, source_node)
            ast.fix_missing_locations(len_eq)
            return len_eq
        checks = [len_eq]
        for k in keys:
            checks.append(
                ast.Compare(
                    left=copy.deepcopy(k),
                    ops=[ast.In()],
                    comparators=[copy.deepcopy(dict_expr)],
                ))
        result = ast.BoolOp(op=ast.And(), values=checks)
        self.ensure_all_locations(result, source_node)
        ast.fix_missing_locations(result)
        return result

    def _try_transform_values_view_eq(self, view_side, literal_side, source_node):
        """Transform list/sorted(d.values()) == [literal] into membership checks.

        Rewrites to ``len(d) == N and v1 in d.values() and ...``. Soundness
        guards (each bails when violated):
        - set wrapper is rejected entirely: dict values may repeat, so the
          rewrite cannot soundly relate ``len(d)`` to ``len(literal_set)``.
        - literal must be ast.List of distinct Constants (duplicates would
          collapse to fewer ``in`` checks but pass the length test, turning
          a False assertion into True).
        - sorted wrapper: literal must be statically sorted ascending.
        - list wrapper: literal must have at most one element (insertion
          order is not modelled).
        """
        dict_expr, wrapper = self._get_dict_view_call(view_side, "values")
        if dict_expr is None or wrapper == "set":
            return None
        if not isinstance(literal_side, ast.List):
            return None
        values = list(literal_side.elts)
        if values and not self._all_constants_distinct(values):
            return None
        if wrapper == "sorted" and not self._is_sorted_const_list(values):
            return None
        if wrapper == "list" and len(values) > 1:
            return None
        len_eq = ast.Compare(
            left=ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[copy.deepcopy(dict_expr)],
                keywords=[],
            ),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=len(values))],
        )
        if not values:
            self.ensure_all_locations(len_eq, source_node)
            ast.fix_missing_locations(len_eq)
            return len_eq
        checks = [len_eq]
        for v in values:
            values_call = ast.Call(
                func=ast.Attribute(
                    value=copy.deepcopy(dict_expr),
                    attr="values",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
            checks.append(
                ast.Compare(
                    left=copy.deepcopy(v),
                    ops=[ast.In()],
                    comparators=[values_call],
                ))
        result = ast.BoolOp(op=ast.And(), values=checks)
        self.ensure_all_locations(result, source_node)
        ast.fix_missing_locations(result)
        return result

    def _try_transform_items_list_eq(self, list_side, literal_side, source_node):
        """Transform list/sorted(d.items()) == [(k,v),...] into dict membership checks.

        Rewrites to ``len(d) == N and k_i in d and d[k_i] == v_i`` per pair.
        Soundness guards:
        - pair keys must be statically distinct Constants. Dict keys are
          distinct, so any literal pair list with duplicate keys never
          equals a real dict's items view; without this guard, e.g.
          ``[(1,'a'),(1,'a')]`` would rewrite to a satisfiable formula
          (``{1:'a', 2:anything}``) — unsound.
        - sorted wrapper: literal must be statically sorted by first tuple
          element (CPython sorts tuples by key first; an unsorted literal
          would compare False but the rewrite would say True).
        - list wrapper: literal must have at most one pair (ESBMC's dict
          model does not preserve insertion order).
        """
        dict_expr, wrapper = self._get_items_dict_expr(list_side,
                                                       ("list", "sorted"))
        if dict_expr is None:
            return None
        if not isinstance(literal_side, ast.List):
            return None
        pairs = []
        for elt in literal_side.elts:
            if not (isinstance(elt, ast.Tuple) and len(elt.elts) == 2):
                return None
            pairs.append((elt.elts[0], elt.elts[1]))
        if pairs and not self._all_constants_distinct([k for k, _ in pairs]):
            return None
        if wrapper == "sorted" and not self._is_sorted_const_list(
                literal_side.elts, by="first"):
            return None
        if wrapper == "list" and len(pairs) > 1:
            return None

        len_eq = ast.Compare(
            left=ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[copy.deepcopy(dict_expr)],
                keywords=[],
            ),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=len(pairs))],
        )

        if not pairs:
            self.ensure_all_locations(len_eq, source_node)
            ast.fix_missing_locations(len_eq)
            return len_eq

        value_checks = [len_eq]
        for k, v in pairs:
            key_in_dict = ast.Compare(
                left=copy.deepcopy(k),
                ops=[ast.In()],
                comparators=[copy.deepcopy(dict_expr)],
            )
            subscript = ast.Subscript(
                value=copy.deepcopy(dict_expr),
                slice=copy.deepcopy(k),
                ctx=ast.Load(),
            )
            val_eq = ast.Compare(left=subscript,
                                 ops=[ast.Eq()],
                                 comparators=[copy.deepcopy(v)])
            value_checks.append(key_in_dict)
            value_checks.append(val_eq)

        result = ast.BoolOp(op=ast.And(), values=value_checks)
        self.ensure_all_locations(result, source_node)
        ast.fix_missing_locations(result)
        return result

    def _try_transform_list_tuple_eq(self, left_side, literal_side, source_node):
        """Transform x == [(a,b), ...] into len/index comparisons for x."""
        if not isinstance(left_side, ast.Name):
            return None
        if not isinstance(literal_side, ast.List):
            return None

        tuple_rows = []
        for elt in literal_side.elts:
            if not isinstance(elt, ast.Tuple):
                return None
            tuple_rows.append(elt)

        checks = [
            ast.Compare(
                left=ast.Call(
                    func=ast.Name(id="len", ctx=ast.Load()),
                    args=[ast.Name(id=left_side.id, ctx=ast.Load())],
                    keywords=[],
                ),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=len(tuple_rows))],
            )
        ]

        for row_idx, tuple_node in enumerate(tuple_rows):
            for col_idx, value_node in enumerate(tuple_node.elts):
                lhs = ast.Subscript(
                    value=ast.Subscript(
                        value=ast.Name(id=left_side.id, ctx=ast.Load()),
                        slice=ast.Constant(value=row_idx),
                        ctx=ast.Load(),
                    ),
                    slice=ast.Constant(value=col_idx),
                    ctx=ast.Load(),
                )
                checks.append(
                    ast.Compare(
                        left=lhs,
                        ops=[ast.Eq()],
                        comparators=[copy.deepcopy(value_node)],
                    ))

        result = ast.BoolOp(op=ast.And(), values=checks)
        self.ensure_all_locations(result, source_node)
        ast.fix_missing_locations(result)
        return result

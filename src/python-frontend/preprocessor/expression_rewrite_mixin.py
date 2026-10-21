import ast
import copy
from collections import Counter
# pylint: disable=protected-access,too-many-locals,too-many-boolean-expressions,too-many-statements


class ExpressionRewriteMixin:

    @staticmethod
    def replace_filtered_genexp_arg(node):
        """Swap a filtered generator argument for the equivalent list comp.

        An eager consumer of a generator drops the generator's `if` clauses on
        the C++ side, so `sum(x for x in xs if x > 2)` silently summed every
        element. The comprehension lowering keeps the filter, so hand it the
        list comprehension instead, as the str.join rewrite already does.
        """
        if not (isinstance(node.func, ast.Name) and node.func.id in ("sum", "min", "max", "sorted")
                and node.args and isinstance(node.args[0], ast.GeneratorExp)
                and any(gen.ifs for gen in node.args[0].generators)):
            return

        gen = node.args[0]
        listcomp = ast.ListComp(elt=copy.deepcopy(gen.elt),
                                generators=copy.deepcopy(gen.generators))
        ast.copy_location(listcomp, gen)
        ast.fix_missing_locations(listcomp)
        node.args[0] = listcomp

    @staticmethod
    def _is_map_call(node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "map" and len(node.args) == 2 and not node.keywords
                and not any(isinstance(a, ast.Starred) for a in node.args))

    def _map_as_listcomp(self, call):
        """map(f, xs) -> [f(t) for t in xs], for a map consumed whole, once."""
        func, iterable = call.args
        if isinstance(func, ast.Lambda) and len(func.args.args) == 1:
            target = ast.Name(id=func.args.args[0].arg, ctx=ast.Store())
            elt = func.body
        else:
            tmp_id = f"ESBMC_map_elt_{self.listcomp_counter}"
            self.listcomp_counter += 1
            target = ast.Name(id=tmp_id, ctx=ast.Store())
            elt = ast.Call(func=func, args=[ast.Name(id=tmp_id, ctx=ast.Load())], keywords=[])
        listcomp = ast.ListComp(
            elt=elt,
            generators=[ast.comprehension(target=target, iter=iterable, ifs=[], is_async=0)])
        ast.copy_location(listcomp, call)
        return ast.fix_missing_locations(listcomp)

    def _map_for_lowerable(self, node, call):
        """Whether `for ... in call` can call f per element: f must be a name
        the loop does not rebind."""
        func = call.args[0]
        return isinstance(func, ast.Name) and not self._binds_name(node, func.id)

    def _map_for_as_plain_for(self, node):
        """for v in map(f, xs): body -> for t in xs: v = f(t); body.

        f is called once per element, when the element is reached, as the
        lazy map calls it."""
        func, iterable = node.iter.args
        tmp_id = f"ESBMC_map_elt_{self.listcomp_counter}"
        self.listcomp_counter += 1
        call = ast.Call(func=func, args=[ast.Name(id=tmp_id, ctx=ast.Load())], keywords=[])
        bind = ast.copy_location(ast.Assign(targets=[node.target], value=call), node)
        node.target = ast.copy_location(ast.Name(id=tmp_id, ctx=ast.Store()), node.target)
        node.iter = iterable
        node.body.insert(0, ast.fix_missing_locations(bind))

    @staticmethod
    def _is_pure_operand(node):
        if isinstance(node, (ast.Name, ast.Constant, ast.Lambda)):
            return True
        if isinstance(node, ast.Attribute):
            return ExpressionRewriteMixin._is_pure_operand(node.value)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return all(map(ExpressionRewriteMixin._is_pure_operand, node.elts))
        return False

    @staticmethod
    def _is_unpacking(node):
        return (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], (ast.Tuple, ast.List)))

    @staticmethod
    def _names(node, name):
        return isinstance(node, ast.Name) and node.id == name

    def _map_consumer_slot(self, stmt, name, call):
        """A setter that puts @p call where @p stmt consumes the iterator
        @p name whole, once and in a form lowered here or in visit_Call;
        None when @p stmt does not."""
        if isinstance(stmt, ast.For):
            if self._names(stmt.iter, name) and self._map_for_lowerable(stmt, call):
                return lambda: setattr(stmt, "iter", call)
            return None
        if self._is_unpacking(stmt) and self._names(stmt.value, name):
            return lambda: setattr(stmt, "value", call)
        call_site = self._list_call_on(stmt, name)
        if call_site is None:
            return None
        return lambda: call_site.args.__setitem__(0, call)

    def _list_call_on(self, stmt, name):
        """The `list(name)` call @p stmt evaluates at most once, or None."""
        if not isinstance(
                stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Assert, ast.Return)):
            return None
        nested = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        stack = [stmt]
        while stack:
            n = stack.pop()
            if (isinstance(n, ast.Call) and self._names(n.func, "list") and len(n.args) == 1
                    and not n.keywords and self._names(n.args[0], name)):
                return n
            if not isinstance(n, nested):
                stack.extend(ast.iter_child_nodes(n))
        return None

    def _inline_single_use_map(self, stmts, i, name_counts):
        """Move `r = map(f, xs)` into the next statement when that is r's only
        use, so the map is consumed where it is built. A map is an iterator:
        lowering a bound one to a list would let a second pass over it see the
        elements again, where CPython sees none."""
        stmt = stmts[i]
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and self._is_map_call(stmt.value)
                and all(map(self._is_pure_operand, stmt.value.args))
                and name_counts[stmt.targets[0].id] == 2):
            return False
        put = self._map_consumer_slot(stmts[i + 1], stmt.targets[0].id, stmt.value)
        if put is None:
            return False
        put()
        del stmts[i]
        return True

    def _inline_single_use_maps(self, module):
        name_counts = Counter(n.id for n in ast.walk(module) if isinstance(n, ast.Name))
        for n in ast.walk(module):
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                name_counts.update({g: 2 for g in n.names})
        for parent in list(ast.walk(module)):
            for field in ("body", "orelse", "finalbody"):
                stmts = getattr(parent, field, None)
                i = 0
                while isinstance(stmts, list) and i + 1 < len(stmts):
                    if not self._inline_single_use_map(stmts, i, name_counts):
                        i += 1

    def _rewrite_map_values(self, module):
        """Lower the map() uses whose result is consumed once, in full: as the
        value of an unpacking, as a for-loop iterable, and through a name used
        only by the next statement. list(map(...)) is lowered in visit_Call."""
        if self._binds_name(module, "map"):
            return
        self._inline_single_use_maps(module)
        for n in ast.walk(module):
            if (self._is_unpacking(n) and self._is_map_call(n.value)
                    and self._is_pure_operand(n.value.args[0])):
                n.value = self._map_as_listcomp(n.value)
            elif (isinstance(n, ast.For) and self._is_map_call(n.iter)
                  and self._map_for_lowerable(n, n.iter)):
                self._map_for_as_plain_for(n)

    _REDUCE_FIRST = """\
{seq} = list(__iterable)
if len({seq}) == 0:
    raise TypeError("reduce() of empty iterable with no initial value")
{acc} = {seq}[0]
for {i} in range(1, len({seq})):
"""

    _REDUCE_INITIAL = """\
{seq} = list(__iterable)
{acc} = __initial
for {i} in range(0, len({seq})):
"""

    _REDUCE_CALL_STEP = "    {acc} = __function({acc}, {seq}[{i}])\n"

    # The converter cannot call a lambda in place, so its body is inlined.
    _REDUCE_LAMBDA_STEP = "    {x} = {acc}\n    {y} = {seq}[{i}]\n    {acc} = __function\n"

    @staticmethod
    def _functools_binding(node, alias):
        """"reduce" when @p alias of import @p node binds functools.reduce,
        "module" when it binds functools itself, else None."""
        if (isinstance(node, ast.ImportFrom) and node.module == "functools" and not node.level
                and alias.name == "reduce"):
            return "reduce"
        if isinstance(node, ast.Import) and alias.name == "functools":
            return "module"
        return None

    def _scan_functools_reduce_names(self, module):
        """Names that can only mean functools.reduce, and names that can only
        mean the functools module: each bound once, by its import."""
        sites = self._binding_sites(module)
        found = {"reduce": set(), "module": set(), None: set()}
        for n in ast.walk(module):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for alias in n.names:
                    name = alias.asname or alias.name
                    if sites[name] == 1:
                        found[self._functools_binding(n, alias)].add(name)
        self.functools_reduce_names = found["reduce"]
        self.functools_module_names = found["module"]

    def _is_functools_reduce_call(self, node):
        """Whether @p node calls functools.reduce in a form the fold lowers."""
        func = node.func
        named = isinstance(func, ast.Name) and func.id in self.functools_reduce_names
        qualified = (isinstance(func, ast.Attribute) and func.attr == "reduce"
                     and isinstance(func.value, ast.Name)
                     and func.value.id in self.functools_module_names)
        return ((named or qualified) and len(node.args) in (2, 3) and not node.keywords
                and (isinstance(node.args[0], ast.Name) or self._is_inlinable_lambda(node.args[0]))
                and not any(isinstance(a, ast.Starred) for a in node.args))

    @staticmethod
    def _is_inlinable_lambda(node):
        """A two-parameter lambda whose body opens no scope of its own, so
        renaming its parameters' loads renames every use of them."""
        if not isinstance(node, ast.Lambda):
            return False
        args = node.args
        scopes = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                  ast.NamedExpr)
        return (len(args.args) == 2 and not args.posonlyargs and not args.vararg
                and not args.kwonlyargs and not args.kwarg and not args.defaults
                and not any(isinstance(n, scopes) for n in ast.walk(node.body)))

    def _lower_functools_reduce(self, call):
        """reduce(f, xs[, initial]) -> the left fold functools.reduce performs,
        raising its TypeError on an empty sequence without an initial value.
        f is called directly at each step, so its argument and result types
        are the caller's."""
        n = self.listcomp_counter
        self.listcomp_counter += 1
        names = {
            "seq": f"ESBMC_reduce_seq_{n}",
            "acc": f"ESBMC_reduce_acc_{n}",
            "i": f"ESBMC_reduce_i_{n}",
            "x": f"ESBMC_reduce_x_{n}",
            "y": f"ESBMC_reduce_y_{n}"
        }
        function = call.args[0]
        step = self._REDUCE_CALL_STEP
        if isinstance(function, ast.Lambda):
            step = self._REDUCE_LAMBDA_STEP
            function = copy.deepcopy(function.body)
            for param, name in zip(call.args[0].args.args, (names["x"], names["y"])):
                function = self._rename_loads(function, param.arg, name)
        template = (self._REDUCE_FIRST if len(call.args) == 2 else self._REDUCE_INITIAL) + step
        holes = dict(zip(("__function", "__iterable", "__initial"), [function, *call.args[1:]]))

        class _Fill(ast.NodeTransformer):

            def visit_Name(self, node):
                """The argument a template placeholder stands for, else @p node."""
                hole = holes.get(node.id)
                return copy.deepcopy(hole) if hole is not None else node

        tree = ast.parse(template.format(**names))
        for node in ast.walk(tree):
            if isinstance(node, (ast.stmt, ast.expr)):
                ast.copy_location(node, call)
        prefix = []
        for stmt in _Fill().visit(tree).body:
            visited = self.visit(stmt)
            prefix.extend(visited if isinstance(visited, list) else [visited])
        return prefix, self.create_name_node(names["acc"], ast.Load(), call)

    class _ListCompExpressionLowerer(ast.NodeTransformer):
        """Utility transformer that lowers list comprehensions, any(genexpr), and all(genexpr) inside an expression."""

        def __init__(self, preprocessor, root):
            super().__init__()
            self.preprocessor = preprocessor
            self.root = root
            self.statements = []
            # Prefix statements run unconditionally, so an expression that
            # might not be evaluated must not be lowered into them.
            self.conditional_depth = 0

        def _visit_conditionally(self, node):
            """Visit @p node, whose operands may go unevaluated."""
            self.conditional_depth += 1
            try:
                return self.generic_visit(node)
            finally:
                self.conditional_depth -= 1

        visit_IfExp = visit_BoolOp = visit_Lambda = _visit_conditionally

        def _lower_functools_reduce_call(self, node):
            """The lowered fold for a functools.reduce call, else None."""
            if self.conditional_depth or not self.preprocessor._is_functools_reduce_call(node):
                return None
            return self.preprocessor._lower_functools_reduce(self.generic_visit(node))

        def visit_ListComp(self, node):
            # pylint: disable=protected-access
            prefix, result_expr = self.preprocessor._lower_listcomp(node)
            self.statements.extend(prefix)
            return result_expr

        def _lower_as_listcomp(self, node):
            """Build a ListComp over node's elt/generators and lower it,
            extending self.statements. Shared by SetComp and GeneratorExp,
            which differ only in what they do with the resulting list."""
            # pylint: disable=protected-access
            listcomp = ast.ListComp(elt=node.elt, generators=node.generators)
            ast.copy_location(listcomp, node)
            ast.fix_missing_locations(listcomp)
            prefix, result_expr = self.preprocessor._lower_listcomp(listcomp)
            self.statements.extend(prefix)
            return result_expr

        def visit_SetComp(self, node):
            list_name = self._lower_as_listcomp(node)
            set_call = ast.Call(
                func=ast.Name(id="set", ctx=ast.Load()),
                args=[list_name],
                keywords=[],
            )
            ast.copy_location(set_call, node)
            ast.fix_missing_locations(set_call)
            return set_call

        def visit_GeneratorExp(self, node):
            """Lower a genexp not already handled above (any/all/join/...)."""
            return self._lower_as_listcomp(node)

        def _other_code_may_run(self, comp):
            """True when the expression may run code outside @p comp other than
            the calls enclosing it, which hoisting @p comp would reorder."""
            parents = {
                id(child): parent
                for parent in ast.walk(self.root)
                for child in ast.iter_child_nodes(parent)
            }
            exempt = {id(n) for n in ast.walk(comp)}
            node = comp
            while id(node) in parents:
                node = parents[id(node)]
                exempt.add(id(node))
            return any(
                isinstance(n, (ast.Call, ast.NamedExpr, ast.Await, ast.Yield,
                               ast.YieldFrom)) and id(n) not in exempt for n in ast.walk(self.root))

        def visit_DictComp(self, node):
            """Hoist a dict comprehension over a dict into a temp, so it takes
            the for-loop lowering an assigned one gets (#5444)."""
            if self.conditional_depth or self._other_code_may_run(node):
                return self.generic_visit(node)
            # pylint: disable=protected-access
            tmp_id = f"ESBMC_dictcomp_{self.preprocessor.listcomp_counter}"
            assign = ast.Assign(targets=[ast.Name(id=tmp_id, ctx=ast.Store())], value=node)
            ast.copy_location(assign, node)
            lowered = self.preprocessor._maybe_lower_dictcomp_over_items(assign)
            if lowered is None:
                return self.generic_visit(node)
            self.preprocessor.listcomp_counter += 1
            self.statements.extend(lowered)
            result = ast.Name(id=tmp_id, ctx=ast.Load())
            ast.copy_location(result, node)
            return result

        def visit_Call(self, node):  # pylint: disable=protected-access,too-many-locals,too-many-boolean-expressions,too-many-statements
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

            self.preprocessor.replace_filtered_genexp_arg(node)

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

            is_list_map_call = (isinstance(node.func, ast.Name) and node.func.id == "list"
                                and len(node.args) == 1 and not node.keywords
                                and isinstance(node.args[0], ast.Call)
                                and isinstance(node.args[0].func, ast.Name)
                                and node.args[0].func.id == "map" and len(node.args[0].args) == 2)
            if is_list_map_call:
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

            # list(filter(pred, seq)) -> [x for x in seq if pred(x)], the exact
            # CPython desugaring (filter keeps the elements for which pred is
            # truthy, in order). filter(None, seq) keeps the truthy elements,
            # so the guard is the element itself. The for-loop form is handled
            # separately by loop_mixin._transform_filter_for.
            is_list_filter_call = (isinstance(node.func, ast.Name) and node.func.id == "list"
                                   and len(node.args) == 1 and not node.keywords
                                   and isinstance(node.args[0], ast.Call)
                                   and isinstance(node.args[0].func, ast.Name)
                                   and node.args[0].func.id == "filter"
                                   and len(node.args[0].args) == 2)
            if is_list_filter_call:
                filter_call = node.args[0]
                func_expr = filter_call.args[0]
                iterable_expr = filter_call.args[1]
                if isinstance(func_expr, ast.Lambda) and len(func_expr.args.args) == 1:
                    param = func_expr.args.args[0]
                    target = ast.Name(id=param.arg, ctx=ast.Store())
                    elt = ast.Name(id=param.arg, ctx=ast.Load())
                    test = func_expr.body
                else:
                    tmp_id = f"ESBMC_filter_elt_{self.preprocessor.listcomp_counter}"
                    target = ast.Name(id=tmp_id, ctx=ast.Store())
                    elt = ast.Name(id=tmp_id, ctx=ast.Load())
                    if isinstance(func_expr, ast.Constant) and func_expr.value is None:
                        test = ast.Name(id=tmp_id, ctx=ast.Load())
                    else:
                        test = ast.Call(
                            func=func_expr,
                            args=[ast.Name(id=tmp_id, ctx=ast.Load())],
                            keywords=[],
                        )
                listcomp = ast.ListComp(
                    elt=elt,
                    generators=[
                        ast.comprehension(target=target, iter=iterable_expr, ifs=[test], is_async=0)
                    ],
                )
                ast.copy_location(listcomp, node)
                ast.fix_missing_locations(listcomp)
                return self.visit(listcomp)

            gen_call, arity_check = self.preprocessor._unguard_starred_call(node)
            if (isinstance(node.func, ast.Name) and node.func.id == "list" and len(node.args) == 1
                    and not node.keywords and isinstance(gen_call, ast.Call)
                    and isinstance(gen_call.func, ast.Name)
                    and gen_call.func.id in self.preprocessor.generator_funcs):
                prefix, result = self.preprocessor._lower_list_gen_call(gen_call, node)
                if prefix is not None:
                    self.statements.extend(arity_check + prefix)
                    return result

            # sum(range(EXPR)) -> (EXPR * (EXPR - 1) // 2 if EXPR > 0 else 0).
            # Without this, range(EXPR) creates a PyListObj with size=EXPR but
            # unpopulated items; sum() then reads nondet via list[i], so
            # `sum_to_n(1) = sum(range(2))` returns nondet instead of 1.
            # Limited to single-argument range and single-argument sum.
            # EXPR is evaluated multiple times in the rewritten tree; this
            # matches the convention of the surrounding ListComp / genexpr
            # rewrites and is safe for the side-effect-free expressions seen
            # in practice (variables, integer arithmetic).
            if (isinstance(node.func, ast.Name) and node.func.id == "sum" and len(node.args) == 1
                    and not node.keywords and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Name) and node.args[0].func.id == "range"
                    and len(node.args[0].args) == 1):
                stop = node.args[0].args[0]
                # Build: stop * (stop - 1) // 2
                product = ast.BinOp(
                    left=copy.deepcopy(stop),
                    op=ast.Mult(),
                    right=ast.BinOp(
                        left=copy.deepcopy(stop),
                        op=ast.Sub(),
                        right=ast.Constant(value=1),
                    ),
                )
                gauss = ast.BinOp(left=product, op=ast.FloorDiv(), right=ast.Constant(value=2))
                # Python: sum(range(n)) == 0 for n <= 0. Guard the formula so
                # the rewrite is exact for the full integer domain, not just
                # the positive case.
                formula = ast.IfExp(
                    test=ast.Compare(
                        left=copy.deepcopy(stop),
                        ops=[ast.Gt()],
                        comparators=[ast.Constant(value=0)],
                    ),
                    body=gauss,
                    orelse=ast.Constant(value=0),
                )
                ast.copy_location(formula, node)
                ast.fix_missing_locations(formula)
                return self.visit(formula)

            node = self.preprocessor._pack_min_max_key_args(node)
            # Ordered: each constant fold gets first refusal, then the scan
            # that handles what it could not fold.
            for lower in (
                    self.preprocessor._lower_sorted_with_key_call,
                    self.preprocessor._lower_min_max_with_key_call,
                    self.preprocessor._lower_min_max_key_scan,
                    self.preprocessor._lower_sorted_key_scan,
                    self.preprocessor._lower_tuple_sorted_pair_call,
                    self._lower_functools_reduce_call,
            ):
                lowered = lower(node)
                if lowered is not None:
                    prefix, result = lowered
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

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if node.attr == "flat" and isinstance(node.ctx, ast.Load):
            ravel_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="ravel",
                    ctx=ast.Load(),
                ),
                args=[node.value],
                keywords=[],
            )
            ast.copy_location(ravel_call, node)
            ast.fix_missing_locations(ravel_call)
            return ravel_call
        return node

    def visit_Subscript(self, node):
        node = self.generic_visit(node)

        # Only constant-fold subscript *reads* (Load context). A subscript in a
        # Store/Del context (e.g. `del a[1]`, `a[1] = x`) is an lvalue target;
        # replacing it with the element's literal value corrupts the statement
        # (a `del a[1]` would become `del 2`).
        if not isinstance(getattr(node, "ctx", None), ast.Load):
            return node

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

        rewritten_seq_next = self._maybe_rewrite_seq_next_expr(node)
        if rewritten_seq_next is not None:
            return rewritten_seq_next

        prefix, new_value, _ = self._lower_listcomp_in_expr(node.value)
        node.value = new_value
        dd_inits, node.value = self._lower_defaultdict_reads_in_expr(node.value, node)
        prefix = dd_inits + prefix
        if prefix:
            return prefix + [node]
        return node

    def visit_If(self, node):
        # A literal one arm binds is not what the other arm, or the code after
        # the join, reads: visit the else arm from the literals known before
        # the if, then forget every name either arm changes (#8617).
        before = (dict(self.list_literal_values), dict(self.dict_literal_values))
        orelse, node.orelse = node.orelse, []
        node = self.generic_visit(node)
        self.list_literal_values, self.dict_literal_values = (dict(before[0]), dict(before[1]))
        else_arm = ast.Module(body=orelse, type_ignores=[])
        node.orelse = self.generic_visit(else_arm).body
        self._forget_literals_changed_in_loop(node)
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

    def visit_Try(self, node):
        # A handler runs after any prefix of the body, and the code after the
        # statement follows either, so no literal the statement changes is
        # known inside it or after it (#8617).
        self._forget_literals_changed_in_loop(node)
        node = self.generic_visit(node)
        self._forget_literals_changed_in_loop(node)
        return node

    def visit_While(self, node):
        # Hoist generator initialisation (e.g. `j = 0`) for `var = next(gen)`
        # loop bodies out of the loop *before* visiting it, so the init runs
        # once and the generator's state persists across iterations. Without
        # this the init is inlined inside the loop body and resets every pass,
        # so `next(gen)` always yields the first value and the loop never makes
        # progress (wedges into an unbounded loop). Mirrors the range-for path
        # in _visit_for_inner.
        gen_pre_stmts = self._hoist_generator_inits(node.body, node)
        self._forget_literals_changed_in_loop(node)
        node = self.generic_visit(node)
        # A break skips the else arm, so neither the body's literals nor the
        # else arm's are known after the loop (#8617).
        self._forget_literals_changed_in_loop(node)
        # Lower `while ... else: <orelse>` (the else runs when the loop ends
        # without break) into a did-not-break flag, the same desugaring used for
        # for-else. Without this the while node keeps its orelse, which the
        # converter emits as an invalid third operand of the GOTO `while`
        # ("while takes two operands"). _lower_for_else is a no-op when there is
        # no orelse and clears node.orelse otherwise.
        while_else_pre, while_else_post = self._lower_for_else(node)
        prefix, new_test, _ = self._lower_listcomp_in_expr(node.test)
        node.test = self._transform_list_truthiness(new_test, node)
        if prefix:
            self._move_test_prefix_into_body(node, prefix)
        result = [node]
        if while_else_pre or while_else_post:
            result = while_else_pre + result + while_else_post
        result = gen_pre_stmts + result
        return result if len(result) > 1 else node

    def _move_test_prefix_into_body(self, node, prefix):
        """Rewrite `while <test>` as `while True` whose body starts with
        `<prefix>; if not <test>: break`, so the statements a comprehension in
        the test lowers to run before every evaluation of the test, not once
        before the loop. Breaks were already rewritten for while-else, so this
        one still lets the else clause run."""
        exit_test = ast.If(test=ast.UnaryOp(op=ast.Not(), operand=node.test),
                           body=[ast.Break()],
                           orelse=[])
        self.ensure_all_locations(exit_test, node)
        node.body = prefix + [exit_test] + node.body
        node.test = self.create_constant_node(True, node)

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

    # Symmetric equality-shape rewrites attempted on `assert L == R`. Each
    # returns the rewritten test or None; the cascade stops at the first hit.
    _SYMMETRIC_EQ_TRANSFORMS = (
        "_try_transform_items_set_eq",
        "_try_transform_items_list_eq",
        "_try_transform_keys_view_eq",
        "_try_transform_values_view_eq",
        "_try_transform_list_tuple_eq",
    )

    def _resolve_call_origin(self, node):
        """Return a deepcopy of ``x``'s tracked Call origin when ``node`` is
        the Name ``x``, else return ``node`` unchanged. Refuse the inline
        for items-view origins on non-dict receivers — the downstream
        cascade only rejects receivers explicitly recorded as non-dict, so
        an unannotated user class with .items() would otherwise be rewritten
        into a dict-membership check.
        """
        if isinstance(node, ast.Name):
            origin = self._assignment_call_origins.get(node.id)
            if origin is None:
                return node
            if self._is_items_view_call(origin):
                recv = self._items_view_receiver_name(origin)
                if recv is None or not self._is_known_dict_name(recv):
                    return node
            return copy.deepcopy(origin)
        return node

    def _apply_assert_eq_rewrites(self, node):
        if not (isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq) and len(node.test.comparators) == 1):
            return [], None
        left, right = node.test.left, node.test.comparators[0]
        left_inlined = self._resolve_call_origin(left)
        right_inlined = self._resolve_call_origin(right)
        # Cross-product of raw and origin-substituted sides so transforms
        # that match Name on one side and Call on the other still fire.
        candidate_pairs = [(left, right)]
        if left_inlined is not left:
            candidate_pairs.append((left_inlined, right))
        if right_inlined is not right:
            candidate_pairs.append((left, right_inlined))
        if left_inlined is not left and right_inlined is not right:
            candidate_pairs.append((left_inlined, right_inlined))
        for name in self._SYMMETRIC_EQ_TRANSFORMS:
            fn = getattr(self, name)
            for pair_left, pair_right in candidate_pairs:
                for lhs, rhs in ((pair_left, pair_right), (pair_right, pair_left)):
                    rewritten = fn(lhs, rhs, node)
                    if rewritten is not None:
                        return [], rewritten
        for lhs, rhs in ((left, right), (right, left)):
            prefix, rewritten = self._try_lower_expr_tuple_literal_eq(lhs, rhs, node)
            if rewritten is not None:
                return prefix or [], rewritten
        return [], None

    def visit_Assert(self, node):
        node = self.generic_visit(node)
        tuple_eq_prefix, rewritten = self._apply_assert_eq_rewrites(node)
        if rewritten is not None:
            node.test = rewritten
            # The rewrite deep-copies part of the test into its prefix, and only
            # node.test is lowered below, so a comprehension carried into the
            # prefix would reach the converter raw (#7692).
            hoisted = []
            for stmt in tuple_eq_prefix:
                comp_prefix, stmt.value, _ = self._lower_listcomp_in_expr(stmt.value)
                hoisted.extend(comp_prefix)
                hoisted.append(stmt)
            tuple_eq_prefix = hoisted
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

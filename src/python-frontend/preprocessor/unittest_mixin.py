"""Expansion of unittest.main() and inlining of TestCase assertions."""

import ast
import copy

_CASE_VAR_PREFIX = "__ESBMC_unittest_case_"
_ARG_VAR_PREFIX = "__ESBMC_unittest_arg_"
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _compare(op):
    return lambda a, b: [ast.Compare(left=a, ops=[op()], comparators=[b])]


def _compare_none(op):
    return lambda a: _compare(op)(a, ast.Constant(value=None))


def _isinstance(obj, cls):
    return ast.Call(func=ast.Name(id="isinstance", ctx=ast.Load()), args=[obj, cls], keywords=[])


def _typed_equal(type_name):
    # assertListEqual and the like fail unless both operands have the type.
    def build(a, b):
        cls = ast.Name(id=type_name, ctx=ast.Load())
        checks = [_isinstance(a, cls), _isinstance(b, copy.deepcopy(cls))]
        return checks + _compare(ast.Eq)(copy.deepcopy(a), copy.deepcopy(b))

    return build


def _not(expr):
    return ast.UnaryOp(op=ast.Not(), operand=expr)


# assertX -> (CPython's parameters before msg, the claims TestCase checks).
_ASSERTIONS = {
    "assertEqual": (("first", "second"), _compare(ast.Eq)),
    "assertNotEqual": (("first", "second"), _compare(ast.NotEq)),
    "assertTrue": (("expr", ), lambda e: [e]),
    "assertFalse": (("expr", ), lambda e: [_not(e)]),
    "assertIs": (("expr1", "expr2"), _compare(ast.Is)),
    "assertIsNot": (("expr1", "expr2"), _compare(ast.IsNot)),
    "assertIsNone": (("obj", ), _compare_none(ast.Is)),
    "assertIsNotNone": (("obj", ), _compare_none(ast.IsNot)),
    "assertIn": (("member", "container"), _compare(ast.In)),
    "assertNotIn": (("member", "container"), _compare(ast.NotIn)),
    "assertLess": (("a", "b"), _compare(ast.Lt)),
    "assertLessEqual": (("a", "b"), _compare(ast.LtE)),
    "assertGreater": (("a", "b"), _compare(ast.Gt)),
    "assertGreaterEqual": (("a", "b"), _compare(ast.GtE)),
    "assertIsInstance": (("obj", "cls"), lambda o, c: [_isinstance(o, c)]),
    "assertNotIsInstance": (("obj", "cls"), lambda o, c: [_not(_isinstance(o, c))]),
    "assertListEqual": (("list1", "list2"), _typed_equal("list")),
    "assertTupleEqual": (("tuple1", "tuple2"), _typed_equal("tuple")),
    "assertDictEqual": (("d1", "d2"), _typed_equal("dict")),
    "assertMultiLineEqual": (("first", "second"), _typed_equal("str")),
    "fail": ((), lambda: [ast.Constant(value=False)]),
}


def _is_names(expr):
    if isinstance(expr, ast.Tuple):
        return all(isinstance(e, ast.Name) for e in expr.elts)
    return isinstance(expr, ast.Name)


def _bind_arguments(call, params):
    """Map *call*'s arguments to *params* and msg, or None if they do not fit."""
    names = params + ("msg", )
    if len(call.args) > len(names) or any(isinstance(a, ast.Starred) for a in call.args):
        return None
    bound = dict(zip(names, call.args))
    for keyword in call.keywords:
        if keyword.arg not in names or keyword.arg in bound:
            return None
        bound[keyword.arg] = keyword.value
    return bound if all(p in bound for p in params) else None


class UnittestMixin:
    """Run the test methods `unittest.main()` would discover.

    The unittest operational model has no runner, so main() is a no-op: a file
    whose only entry point is `unittest.main()` reaches verification with zero
    VCCs and reports a vacuous VERIFICATION SUCCESSFUL (#6745). Each discovered
    test method gets its own instance, as CPython does, so state a test writes
    in setUp cannot leak into the next one.
    """

    def expand_unittest_main(self, module_node):
        """Replace every unittest.main() with its discovered test-method calls."""
        modules, mains, case_names = self._unittest_bindings(module_node)
        self._inline_assertions(module_node, modules, case_names)
        if not modules and not mains:
            return
        cases = self._collect_test_cases(module_node, modules, case_names)
        if cases:
            self._expand_main_calls(module_node.body, (modules, mains), cases, 0)

    @staticmethod
    def _unittest_bindings(module_node):
        """Names bound to the unittest module, to its main() and its TestCase."""
        modules, mains, cases = set(), set(), set()
        for stmt in ast.walk(module_node):
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name == "unittest":
                        modules.add(alias.asname or alias.name)
            elif isinstance(stmt, ast.ImportFrom) and stmt.module == "unittest":
                for alias in stmt.names:
                    bound = alias.asname or alias.name
                    if alias.name == "main":
                        mains.add(bound)
                    elif alias.name == "TestCase":
                        cases.add(bound)
        return modules, mains, cases

    @staticmethod
    def _is_test_case_base(base, modules, case_names):
        if isinstance(base, ast.Name):
            return base.id in case_names
        return (isinstance(base, ast.Attribute) and base.attr == "TestCase"
                and isinstance(base.value, ast.Name) and base.value.id in modules)

    @staticmethod
    def _is_test_method(node):
        # A decorated method is skipped: @unittest.skip and @expectedFailure
        # both mean the runner does not count a failure from this method.
        if not isinstance(node, ast.FunctionDef) or node.decorator_list:
            return False
        args = node.args
        return (node.name.startswith("test") and len(args.args) == 1 and not args.posonlyargs
                and not args.vararg and not args.kwonlyargs and not args.kwarg)

    @classmethod
    def _collect_test_cases(cls, module_node, modules, case_names):
        """(class, sorted test methods) per TestCase subclass, in CPython's order.

        A subclass runs the tests it inherits as well as its own, so a base
        class's method is collected again for each subclass -- its setUp may
        differ. A name redefined in the subclass body is not inherited twice.
        """
        known = dict.fromkeys(case_names, frozenset())
        cases = []
        for stmt in module_node.body:
            if not isinstance(stmt, ast.ClassDef):
                continue
            bases = [b for b in stmt.bases if cls._is_test_case_base(b, modules, known)]
            if not bases:
                continue
            inherited = set().union(*(known[b.id] for b in bases if isinstance(b, ast.Name)))
            defined = {m.name for m in stmt.body if isinstance(m, ast.FunctionDef)}
            methods = {m.name for m in stmt.body if cls._is_test_method(m)}
            methods |= inherited - defined
            known[stmt.name] = frozenset(methods)
            if methods:
                cases.append((stmt.name, sorted(methods)))
        return cases

    @staticmethod
    def _is_main_call(stmt, names):
        modules, mains = names
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return False
        func = stmt.value.func
        if isinstance(func, ast.Name):
            return func.id in mains
        return (isinstance(func, ast.Attribute) and func.attr == "main"
                and isinstance(func.value, ast.Name) and func.value.id in modules)

    def _expand_main_calls(self, body, names, cases, index):
        """Replace each main() in *body* in place; returns the next case index."""
        expanded = []
        for stmt in body:
            if isinstance(stmt, ast.If):
                index = self._expand_main_calls(stmt.body, names, cases, index)
                index = self._expand_main_calls(stmt.orelse, names, cases, index)
            elif self._is_main_call(stmt, names):
                run, index = self._build_case_runs(cases, stmt, index)
                expanded.extend(run)
                continue
            expanded.append(stmt)
        body[:] = expanded
        return index

    def _build_case_runs(self, cases, source, index):
        run = []
        for class_name, methods in cases:
            for method in methods:
                var = f"{_CASE_VAR_PREFIX}{index}"
                index += 1
                run.append(
                    ast.Assign(targets=[ast.Name(id=var, ctx=ast.Store())],
                               value=self._call(ast.Name(id=class_name, ctx=ast.Load()))))
                run.extend(
                    ast.Expr(value=self._call(
                        ast.Attribute(
                            value=ast.Name(id=var, ctx=ast.Load()), attr=name, ctx=ast.Load())))
                    for name in ("setUp", method, "tearDown"))
        for stmt in run:
            self.ensure_all_locations(stmt, source)
            ast.fix_missing_locations(stmt)
        return run, index

    def _inline_assertions(self, module_node, modules, case_names):
        """Lower `self.assertX(...)` in a TestCase subclass to plain asserts.

        The model's assertX parameters are Any, which erases a container's type
        (`first == second` compares two lists by address), so each claim is
        typed at the call site instead. A call that may reach another assertX
        -- a name some class here binds or some attribute store sets, or a
        class with a base, decorator or metaclass outside this module -- stays.
        """
        if not modules and not case_names:
            return
        known = set(case_names)
        overridden = self._rebound_names(module_node)
        for stmt in module_node.body:
            if self._is_plain_test_case(stmt, modules, known):
                known.add(stmt.name)
                for method in stmt.body:
                    self._inline_method_assertions(method, overridden)

    def _rebound_names(self, module_node):
        """Names a class here binds, and attributes some store sets."""
        names = {
            name
            for c in ast.walk(module_node) if isinstance(c, ast.ClassDef) for n in ast.walk(c)
            for name in self._names_bound_by(n)
        }
        names.update(n.attr for n in ast.walk(module_node)
                     if isinstance(n, ast.Attribute) and not isinstance(n.ctx, ast.Load))
        return names

    def _is_plain_test_case(self, stmt, modules, known):
        return (isinstance(stmt, ast.ClassDef) and stmt.bases and not stmt.decorator_list
                and not stmt.keywords
                and all(self._is_test_case_base(b, modules, known) for b in stmt.bases))

    def _inline_method_assertions(self, method, overridden):
        if not isinstance(method, ast.FunctionDef) or method.decorator_list or not method.args.args:
            return
        receiver = method.args.args[0]
        if self._binds_name(method, receiver.arg, receiver):
            return
        self._inline_in_body(method.body, receiver.arg, overridden)

    def _inline_in_body(self, body, receiver, overridden):
        lowered = []
        for stmt in body:
            claims = self._assertion_claims(stmt, receiver, overridden)
            if claims is None and not isinstance(stmt, _SCOPES):
                for field in ("body", "orelse", "finalbody"):
                    self._inline_in_body(getattr(stmt, field, []), receiver, overridden)
                for handler in getattr(stmt, "handlers", []):
                    self._inline_in_body(handler.body, receiver, overridden)
            if claims is None:
                lowered.append(stmt)
                continue
            for claim in claims:
                self.ensure_all_locations(claim, stmt)
                ast.fix_missing_locations(claim)
            lowered.extend(claims)
        body[:] = lowered

    @staticmethod
    def _assertion_call(stmt, receiver, overridden):
        """The `receiver.assertX(...)` call *stmt* consists of, or None."""
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return None
        func = stmt.value.func
        if (isinstance(func, ast.Attribute) and func.attr in _ASSERTIONS
                and func.attr not in overridden and isinstance(func.value, ast.Name)
                and func.value.id == receiver):
            return stmt.value
        return None

    def _assertion_claims(self, stmt, receiver, overridden):
        """The statements that replace an assertX call statement, or None."""
        call = self._assertion_call(stmt, receiver, overridden)
        if call is None:
            return None
        params, build = _ASSERTIONS[call.func.attr]
        bound = _bind_arguments(call, params)
        if bound is None:
            return None
        staged, temps = self._stage_arguments(call)
        operands = (ast.Name(id=temps[id(bound[p])], ctx=ast.Load())
                    if id(bound[p]) in temps else bound[p] for p in params)
        return staged + [ast.Assert(test=t, msg=None) for t in build(*operands)]

    def _stage_arguments(self, call):
        """Evaluate each argument once, in source order, as the call would.

        A constant, and a last argument made of names (isinstance's classes),
        stay inline: nothing evaluated after them can change their value.
        """
        args = call.args + [k.value for k in call.keywords]
        temps = {}
        staged = []
        for i, arg in enumerate(args):
            if isinstance(arg, ast.Constant) or (i == len(args) - 1 and _is_names(arg)):
                continue
            name = f"{_ARG_VAR_PREFIX}{self._next_unpack_tmp_id()}"
            temps[id(arg)] = name
            staged.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=arg))
        return staged, temps

    @staticmethod
    def _call(func):
        return ast.Call(func=func, args=[], keywords=[])

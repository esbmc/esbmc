"""Expansion of unittest.main() into the calls its runner would make."""

import ast

_CASE_VAR_PREFIX = "__ESBMC_unittest_case_"


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

    @staticmethod
    def _call(func):
        return ast.Call(func=func, args=[], keywords=[])

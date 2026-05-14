import ast
import importlib.util
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY_FRONTEND_DIR = os.path.join(ROOT, "src", "python-frontend")

if PY_FRONTEND_DIR not in sys.path:
    sys.path.insert(0, PY_FRONTEND_DIR)


def _load_module(module_name: str, rel_path: str):
    module_path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocessor_mod = _load_module("esbmc_preprocessor_regressions",
                                "src/python-frontend/preprocessor.py")


def test_isolate_genexp_targets_renames_inner_iter_loads_before_shadowing():
    call = ast.parse("all(x > 0 for x in xs for x in range(x))", mode="eval").body
    genexp = call.args[0]

    pre = preprocessor_mod.Preprocessor("test_module")
    isolated = pre._isolate_genexp_targets(genexp)

    outer_target = isolated.generators[0].target
    inner_iter_arg = isolated.generators[1].iter.args[0]
    inner_target = isolated.generators[1].target

    assert isinstance(outer_target, ast.Name)
    assert isinstance(inner_iter_arg, ast.Name)
    assert isinstance(inner_target, ast.Name)
    assert inner_iter_arg.id == outer_target.id
    assert inner_target.id != outer_target.id


def test_lower_listcomp_in_boolop_preserves_list_result_type():
    module = ast.parse(
        """
xs: list[int] = [1]
vals = [x for x in xs] and []
if vals:
    pass
"""
    )

    pre = preprocessor_mod.Preprocessor("test_module")
    transformed = pre.visit(module)

    assert pre.known_variable_types["vals"] == "list"
    if_node = transformed.body[-1]
    assert isinstance(if_node, ast.If)
    assert isinstance(if_node.test, ast.Compare)
    assert isinstance(if_node.test.left, ast.Call)
    assert isinstance(if_node.test.left.func, ast.Name)
    assert if_node.test.left.func.id == "len"


def _find_call(tree, callee_id):
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == callee_id):
            return n
    return None


def test_range_alias_rewrite_canonicalises_call_site():
    module = ast.parse(
        """
my_range = range
for i in my_range(5):
    pass
"""
    )

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._rewrite_range_aliases(module)

    assert _find_call(module, "my_range") is None
    assert _find_call(module, "range") is not None
    assert not any(
        isinstance(s, ast.Assign) and s.targets[0].id == "my_range"
        for s in module.body
    )


def test_range_alias_rewrite_resolves_chains():
    module = ast.parse(
        """
a = range
b = a
for i in b(3):
    pass
"""
    )

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._rewrite_range_aliases(module)

    assert _find_call(module, "a") is None
    assert _find_call(module, "b") is None
    assert _find_call(module, "range") is not None
    assert module.body[0].__class__.__name__ == "For"


def test_range_alias_rewrite_ignores_unrelated_assignments():
    module = ast.parse(
        """
xs = [1, 2, 3]
n = 5
for i in range(n):
    pass
"""
    )

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._rewrite_range_aliases(module)

    assert len(module.body) == 3
    assert isinstance(module.body[0], ast.Assign)
    assert isinstance(module.body[1], ast.Assign)


def test_range_wrapper_inline_rewrites_call_site():
    module = ast.parse(
        """
def my_range(n):
    return range(n)

for i in my_range(5):
    pass
"""
    )

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._inline_range_wrappers(module)

    call = _find_call(module, "range")
    assert call is not None
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Constant)
    assert call.args[0].value == 5


def test_range_wrapper_inline_substitutes_param_in_template():
    module = ast.parse(
        """
def affine(start, stop):
    return range(start, stop, 2)

for i in affine(10, 20):
    pass
"""
    )

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._inline_range_wrappers(module)

    call = _find_call(module, "range")
    assert call is not None
    assert len(call.args) == 3
    assert isinstance(call.args[0], ast.Constant) and call.args[0].value == 10
    assert isinstance(call.args[1], ast.Constant) and call.args[1].value == 20
    assert isinstance(call.args[2], ast.Constant) and call.args[2].value == 2


def test_range_wrapper_inline_rejects_non_trivial_body():
    module = ast.parse(
        """
def not_a_wrapper(n):
    x = n + 1
    return range(x)

for i in not_a_wrapper(5):
    pass
"""
    )

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._inline_range_wrappers(module)

    # The wrapper is not inlined: a call to not_a_wrapper remains, and
    # no synthesised top-level `range(...)` call appears.
    assert _find_call(module, "not_a_wrapper") is not None


def test_range_wrapper_inline_rejects_computed_arg():
    module = ast.parse(
        """
def adds_one(n):
    return range(n + 1)

for i in adds_one(5):
    pass
"""
    )

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._inline_range_wrappers(module)

    assert _find_call(module, "adds_one") is not None
    # The only `range` call left is the one inside the wrapper body.
    range_call = _find_call(module, "range")
    assert isinstance(range_call.args[0], ast.BinOp)


def test_alias_pass_before_wrapper_pass_unlocks_wrapper_detection():
    module = ast.parse(
        """
alias = range

def my_range(n):
    return alias(n)

for i in my_range(5):
    pass
"""
    )

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._rewrite_range_aliases(module)
    pre._inline_range_wrappers(module)

    call = _find_call(module, "range")
    assert call is not None
    assert isinstance(call.args[0], ast.Constant)
    assert call.args[0].value == 5
    assert _find_call(module, "my_range") is None
    assert _find_call(module, "alias") is None


def test_alias_rewrite_skips_nested_scope_that_shadows_name():
    module = ast.parse(
        """
my_range = range

def inner():
    my_range = list
    for i in my_range(5):
        pass
""")

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._rewrite_range_aliases(module)

    inner = module.body[0]
    assert isinstance(inner, ast.FunctionDef) and inner.name == "inner"
    call = _find_call(inner, "my_range")
    assert call is not None, "shadowed inner call must not be rewritten"


def test_wrapper_inline_skips_nested_scope_that_shadows_name():
    module = ast.parse(
        """
def my_range(n):
    return range(n)

def caller():
    def my_range(n):
        return [99]
    for i in my_range(5):
        pass
""")

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._inline_range_wrappers(module)

    caller = module.body[1]
    assert isinstance(caller, ast.FunctionDef) and caller.name == "caller"
    call = _find_call(caller, "my_range")
    assert call is not None, "shadowed inner call must not be rewritten"


def test_alias_rewrite_descends_into_nested_scope_without_shadow():
    module = ast.parse(
        """
my_range = range

def inner():
    for i in my_range(5):
        pass
""")

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._rewrite_range_aliases(module)

    inner = module.body[0]
    assert _find_call(inner, "my_range") is None
    assert _find_call(inner, "range") is not None


def test_rebound_detection_catches_augassign():
    module = ast.parse(
        """
my_range = range
my_range += 1
""")

    rebound = preprocessor_mod.Preprocessor._rebound_module_names(module)
    assert "my_range" in rebound


def test_rebound_detection_catches_tuple_unpacking():
    module = ast.parse(
        """
my_range = range
my_range, other = list, 2
""")

    rebound = preprocessor_mod.Preprocessor._rebound_module_names(module)
    assert "my_range" in rebound


def test_rebound_detection_catches_conditional_rebind():
    module = ast.parse(
        """
my_range = range
if True:
    my_range = list
""")

    rebound = preprocessor_mod.Preprocessor._rebound_module_names(module)
    assert "my_range" in rebound


def test_rebound_detection_catches_global_declaration():
    module = ast.parse(
        """
my_range = range

def mutate():
    global my_range
    my_range = list
""")

    rebound = preprocessor_mod.Preprocessor._rebound_module_names(module)
    assert "my_range" in rebound


def test_alias_rewrite_rejects_alias_with_global_rebind():
    module = ast.parse(
        """
my_range = range

def mutate():
    global my_range
    my_range = list

for i in my_range(5):
    pass
""")

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._rewrite_range_aliases(module)

    assert _find_call(module, "my_range") is not None
    assert _find_call(module, "range") is None

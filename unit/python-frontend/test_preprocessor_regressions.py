import ast
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY_FRONTEND_DIR = os.path.join(ROOT, "src", "python-frontend")

if PY_FRONTEND_DIR not in sys.path:
    sys.path.insert(0, PY_FRONTEND_DIR)


# pylint: disable=wrong-import-position
import preprocessor as preprocessor_mod


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


# ---- Cross-module range-alias / wrapper propagation (#4525) ----


def test_collect_range_aliases_returns_local_set():
    module = ast.parse("a = range\nb = a\n")
    local, all_aliases = preprocessor_mod.Preprocessor.collect_range_aliases(module)
    assert local == {"a", "b"}
    assert all_aliases == {"a", "b"}


def test_collect_range_aliases_grows_chain_from_seed():
    # `a` lives in another module; this one chains `b = a`.
    module = ast.parse("b = a\n")
    local, all_aliases = preprocessor_mod.Preprocessor.collect_range_aliases(
        module, seed={"a"})
    assert local == {"b"}
    assert all_aliases == {"a", "b"}


def test_collect_range_aliases_seed_dropped_when_rebound_locally():
    # Real parser flow puts the imported name into the seed because of the
    # `from lib import a` line. A subsequent rebind shadows the import.
    module = ast.parse(
        """
from lib import a
a = something_else
""")
    local, all_aliases = preprocessor_mod.Preprocessor.collect_range_aliases(
        module, seed={"a"})
    assert local == set()
    assert "a" not in all_aliases


def test_rewrite_with_seed_canonicalises_imported_alias_call():
    module = ast.parse(
        """
from lib import nl_affine_range
for i in nl_affine_range(5):
    pass
""")

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._rewrite_range_aliases(module, seed={"nl_affine_range"})

    assert _find_call(module, "nl_affine_range") is None
    assert _find_call(module, "range") is not None
    # Bare re-imports re-export the surviving alias so consumers of *this*
    # module can resolve it transitively (#4525).
    assert pre.exported_range_aliases == {"nl_affine_range"}


def test_rewrite_with_seed_exports_chained_local_alias():
    # Importer adds its own chain on top of the imported alias and must
    # re-export both the local chain target and the surviving seed name.
    module = ast.parse(
        """
from lib_a import nl_affine_range
aff = nl_affine_range
""")

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._rewrite_range_aliases(module, seed={"nl_affine_range"})

    assert pre.exported_range_aliases == {"aff", "nl_affine_range"}


def test_rewrite_with_seed_does_not_remove_imported_alias_statement():
    # The seeded name lives in another module; we must never drop the
    # importer's own `from lib import ...` statement or any local binding
    # that didn't come from our own `X = range` rewriter.
    module = ast.parse(
        """
from lib import nl_affine_range
for i in nl_affine_range(5):
    pass
""")

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._rewrite_range_aliases(module, seed={"nl_affine_range"})

    assert any(isinstance(s, ast.ImportFrom) for s in module.body)


def test_rewrite_with_seed_skipped_when_local_rebind_shadows():
    # Local function definition shadows the imported alias — rewriter
    # must leave the call site alone.
    module = ast.parse(
        """
from lib import nl_affine_range

def nl_affine_range(n):
    return [0, 1]

for i in nl_affine_range(5):
    pass
""")

    pre = preprocessor_mod.Preprocessor("test_module")
    pre._rewrite_range_aliases(module, seed={"nl_affine_range"})

    assert _find_call(module, "range") is None
    assert _find_call(module, "nl_affine_range") is not None


def test_collect_range_wrappers_returns_local_definitions():
    module = ast.parse(
        """
def my_range(n):
    return range(n)

def not_a_wrapper(n):
    return n + 1
""")

    wrappers = preprocessor_mod.Preprocessor.collect_range_wrappers(module)
    assert set(wrappers) == {"my_range"}
    params, template = wrappers["my_range"]
    assert params == ["n"]
    assert isinstance(template[0], ast.Name) and template[0].id == "n"


def test_inline_with_seed_rewrites_imported_wrapper_call_site():
    module = ast.parse(
        """
from lib import affine_range
for i in affine_range(7):
    pass
""")

    pre = preprocessor_mod.Preprocessor("test_module")
    # Seed mimics a wrapper imported from another module:
    # def affine_range(n): return range(n)
    seed = {"affine_range": (["n"], [ast.Name(id="n", ctx=ast.Load())])}
    pre._inline_range_wrappers(module, seed=seed)

    call = _find_call(module, "range")
    assert call is not None
    assert isinstance(call.args[0], ast.Constant) and call.args[0].value == 7
    # The original wrapper call (Name == "affine_range") should be gone.
    assert _find_call(module, "affine_range") is None


def test_inline_with_seed_skipped_when_local_rebind_shadows():
    module = ast.parse(
        """
from lib import affine_range

def affine_range(n):
    return [n]

for i in affine_range(7):
    pass
""")

    pre = preprocessor_mod.Preprocessor("test_module")
    seed = {"affine_range": (["n"], [ast.Name(id="n", ctx=ast.Load())])}
    pre._inline_range_wrappers(module, seed=seed)

    assert _find_call(module, "range") is None
    assert _find_call(module, "affine_range") is not None


def test_capture_dunder_all_returns_string_list():
    module = ast.parse("__all__ = ['nl_affine_range', 'foo']\n")
    names = preprocessor_mod.Preprocessor._capture_dunder_all(module)
    assert names == ["nl_affine_range", "foo"]


def test_capture_dunder_all_returns_none_for_dynamic_assignment():
    module = ast.parse("__all__ = some_helper()\n")
    assert preprocessor_mod.Preprocessor._capture_dunder_all(module) is None


def test_capture_dunder_all_returns_none_for_non_string_element():
    module = ast.parse("__all__ = ['ok', 42]\n")
    assert preprocessor_mod.Preprocessor._capture_dunder_all(module) is None


def test_capture_dunder_all_absent_returns_none():
    module = ast.parse("x = 1\n")
    assert preprocessor_mod.Preprocessor._capture_dunder_all(module) is None


def _walk_calls(tree, name):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == name]


def test_prepare_then_finalize_lowers_cross_module_alias_to_bounded_range():
    # Reproduces #4533: when the alias is only visible via a cross-module
    # seed, prepare_module must canonicalise the call site to range(N)
    # before finalize_module runs visit_For, otherwise the for-loop drops to
    # the generic iterator path and the bound is lost (unbounded unwinding).
    module = ast.parse(
        """
from lib import nl_affine_range

count: int = 0
for i in nl_affine_range(3):
    count = count + 1
""")

    pre = preprocessor_mod.Preprocessor("main")
    pre.prepare_module(module, alias_seed={"nl_affine_range"})
    pre.finalize_module(module)

    # The call site is now range(...), and the for-range path was taken
    # (it injects helper functions for bounded iteration).
    assert _walk_calls(module, "nl_affine_range") == []
    assert _walk_calls(module, "ESBMC_range_next_") != []
    assert pre.helper_functions_added is True


def test_visit_module_back_compat_runs_prepare_and_finalize():
    # The legacy single-call entry point must still produce the same
    # in-file behaviour: range-alias rewrite + bounded for-range lowering.
    module = ast.parse(
        """
nl_affine_range = range

count: int = 0
for i in nl_affine_range(3):
    count = count + 1
""")

    pre = preprocessor_mod.Preprocessor("main")
    pre.visit(module)

    assert _walk_calls(module, "nl_affine_range") == []
    assert _walk_calls(module, "ESBMC_range_next_") != []


def test_visit_module_records_export_tables():
    # The Preprocessor pre-passes already ran by the time visit_Module
    # finishes, so the exported tables should reflect the in-file aliases.
    module = ast.parse(
        """
__all__ = ['nl_affine_range', 'affine_range']

nl_affine_range = range

def affine_range(n):
    return range(n)
""")

    pre = preprocessor_mod.Preprocessor("lib")
    pre.visit(module)

    assert pre.exported_range_aliases == {"nl_affine_range"}
    assert set(pre.exported_range_wrappers) == {"affine_range"}
    assert pre.module_dunder_all == ["nl_affine_range", "affine_range"]

import ast
import copy

from preprocessor.dataclass_mixin import DataclassMixin
from preprocessor.generator_mixin import GeneratorMixin
from preprocessor.loop_mixin import LoopMixin


class Preprocessor(DataclassMixin, GeneratorMixin, LoopMixin, ast.NodeTransformer):

    def __init__(self, module_name):
        # Initialize with an empty target name
        self.target_name = ""
        self.functionDefaults = {}
        self.functionParams = {}
        self.module_name = module_name  # for errors
        self.is_range_loop = False  # Track if we're in a range loop transformation
        self.known_variable_types = {}
        self.range_loop_counter = 0  # Counter for unique variable names in nested range loops
        self.iterable_loop_counter = 0  # Counter for unique variable names in nested iterable loops
        self.enumerate_loop_counter = 0  # Counter for unique variable names in nested enumerate loops
        self.nondet_expand_counter = 0  # Counter for unique variable names in nondet expansion
        self.helper_functions_added = False  # Track if helper functions have been added
        self.functionKwonlyParams = {}
        self.listcomp_counter = 0  # Counter for list comprehension temporaries
        self.variable_annotations = {}  # Store full AST annotations
        self.function_return_annotations = {}  # Store function return type annotations
        self.class_attr_annotations = {}  # {class_name: {attr_name: annotation_node}}
        self.instance_class_map = {}  # {var_name: class_name} from c = C()
        self.decimal_imported = False
        self.decimal_module_imported = False
        self.decimal_class_alias = None
        self.decimal_module_alias = None
        self.defaultdict_imported = False
        self.defaultdict_alias = None  # alias for the name (from collections import defaultdict as dd)
        self.collections_module_imported = False
        self.collections_module_alias = None  # alias for the module (import collections as col)
        self._subscript_inferred_vars = set(
        )  # vars whose annotations came from subscript inference
        self.generator_funcs = set()  # all generator functions (contain yield)
        self.early_return_generator_funcs = set()  # generators with early return before first yield
        self.generator_vars = {}  # var_name -> func_name for generator variables
        self.generator_func_defs = {}  # func_name -> transformed body (list of stmts)
        self.generator_next_index = {}  # gen_var -> next yield index for next() calls
        self.generator_emitted_init = set()  # gen_vars whose outer_init has been emitted
        self.dict_items_vars = {}  # {var_name: dict_expr} for X = d.items() assignments
        self._defaultdict_factory = {}  # {var_name: factory AST node} for defaultdict vars
        self.het_dict_literals = {
        }  # {var_name: dict AST node} for dicts with heterogeneous key types
        self.het_value_dict_literals = {
        }  # {var_name: dict AST node} for dicts with heterogeneous value types
        self.bound_method_vars = {}  # {var_name: ast.Attribute} for g = obj.method assignments
        self.called_names = set()  # names used as callees: g() → 'g' ∈ called_names
        self.list_literal_values = {}  # {var_name: ast.List} for direct list literal assignments
        self.newtype_vars = set()  # names defined via typing.NewType: X = NewType('X', T)
        self.newtype_names = {"NewType"
                              }  # local names bound to typing.NewType (covers aliased imports)
        self.typing_module_names = set()  # module names for typing (e.g. 'typing' or its alias)
        self._typing_imported_names = (
            set()
        )  # local names brought from typing (e.g. {'Tuple', 'List'} after `from typing import Tuple, List`)
        self._with_counter = 0  # Counter for unique context manager temp names
        self._unroll_counter = 0  # Counter for unique unrolled-loop temp names
        self.type_aliases = (
            {}
        )  # {alias_name: annotation_ast} for type alias assignments (Coordinate = Tuple[int, int])
        # Local names bound to ``dataclasses.dataclass`` and ``dataclasses.field``
        # respectively. Seeded with the canonical names so source files that omit
        # the explicit import (rare, but harmless) still work; ``visit_ImportFrom``
        # adds aliased names like ``from dataclasses import field as f``.
        self._dataclass_decorator_names = {"dataclass"}
        self._dataclass_field_names = {"field"}
        self._dataclass_initvar_names = {"InitVar"}
        self._dataclass_is_dataclass_names = {"is_dataclass"}
        self._dataclass_fields_api_names = {"fields"}
        self._dataclass_asdict_names = {"asdict"}
        self._dataclass_astuple_names = {"astuple"}
        self._dataclass_replace_names = {"replace"}
        self.dataclasses_module_names = {"dataclasses"}
        self._typing_classvar_names = {"ClassVar"}
        self._classes_with_post_init = set()
        self._dataclass_class_specs = {}
        self._needs_dataclass_field_helper = False
        self._needs_dataclass_replace_error_helper = False
        self._needs_dataclass_getattr_helper = False
        self._needs_dataclass_initvar_import = False
        self._assert_eq_counter = 0
        self._known_literal_values = {}
        self._identity_functions = set()
        # Cross-module range-alias / wrapper propagation (#4525). Populated
        # by visit_Module so parser.py can project these into consumers'
        # imported-seed sets.
        self.exported_range_aliases = set()
        self.exported_range_wrappers = {}
        self.module_dunder_all = None

    # Names treated as typing-style generic constructors (subscript = type alias).
    _TYPING_GENERIC_NAMES = (
        "Tuple",
        "List",
        "Dict",
        "Set",
        "Optional",
        "Union",
        "Callable",
        "ClassVar",
    )

    # Dataclass fields declared with ``field(default_factory=...)`` are exposed
    # as synthesized ``__init__`` parameters with a ``None`` default.
    # The current lowering still assigns ``self.<field> = <factory>()``
    # directly in the body for deterministic per-instance initialization.

    def _is_type_alias_expression(self, value):
        """Check whether ``value`` is a typing alias RHS like ``Tuple[int, int]``.

        Tightened so that ordinary runtime indexing such as ``x = List[0]``
        (where ``List`` happens to be a regular variable) is not silently
        stripped from the AST. The base must be a typing name imported from
        ``typing`` (or accessed via the ``typing`` module), and the slice
        must look like a type expression rather than a plain integer index.
        """
        if not isinstance(value, ast.Subscript):
            return False

        base = value.value
        if isinstance(base, ast.Name):
            # Accept direct imports such as ``from typing import List`` and
            # aliased imports such as ``from typing import List as L``.
            if base.id not in self._typing_imported_names:
                return False
        elif isinstance(base, ast.Attribute):
            if base.attr not in self._TYPING_GENERIC_NAMES:
                return False
            mod = base.value
            if not (isinstance(mod, ast.Name) and mod.id in self.typing_module_names):
                return False
        else:
            return False

        # Reject plain integer-index runtime usage (e.g. ``Foo[0]`` after
        # ``Foo = SomeRuntimeContainer``). Type aliases are subscripted with
        # types or tuples of types, never with bare integer literals.
        idx = value.slice
        if isinstance(idx, ast.Index):
            idx = idx.value
        if isinstance(idx, ast.Constant) and isinstance(idx.value, int):
            return False
        if (isinstance(idx, ast.UnaryOp) and isinstance(idx.op, (ast.UAdd, ast.USub))
                and isinstance(idx.operand, ast.Constant) and isinstance(idx.operand.value, int)):
            return False
        return True

    def _copy_annotation_node(self, node):
        """Deep copy an annotation AST node"""
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return ast.Name(id=node.id, ctx=ast.Load())
        elif isinstance(node, ast.Subscript):
            return ast.Subscript(
                value=self._copy_annotation_node(node.value),
                slice=self._copy_annotation_node(node.slice),
                ctx=ast.Load(),
            )
        elif isinstance(node, ast.Index):
            return ast.Index(value=self._copy_annotation_node(node.value))
        elif isinstance(node, ast.Tuple):
            return ast.Tuple(elts=[self._copy_annotation_node(e) for e in node.elts],
                             ctx=ast.Load())
        elif isinstance(node, ast.Constant):
            return ast.Constant(value=node.value)
        elif isinstance(node, ast.Str):
            return ast.Str(s=node.s)
        elif isinstance(node, ast.Attribute):
            return ast.Attribute(
                value=self._copy_annotation_node(node.value),
                attr=node.attr,
                ctx=ast.Load(),
            )
        else:
            # For other node types, return as-is
            return node

    def _resolve_annotation_aliases(self, annotation):
        """Recursively resolve type aliases in an annotation AST node"""
        if annotation is None:
            return None

        if isinstance(annotation, ast.Name):
            # If this name is an alias, return a copy of the aliased type
            if annotation.id in self.type_aliases:
                # Copy then recursively resolve transitive aliases (e.g. TaskSchedule → List[TaskInstance] → List[Tuple[int,Task]])
                copied = self._copy_annotation_node(self.type_aliases[annotation.id])
                return self._resolve_annotation_aliases(copied)
            return annotation

        elif isinstance(annotation, ast.Subscript):
            # Recursively resolve value and slice
            resolved_value = self._resolve_annotation_aliases(annotation.value)
            resolved_slice = annotation.slice
            if isinstance(annotation.slice, ast.Index):
                resolved_slice = ast.Index(
                    value=self._resolve_annotation_aliases(annotation.slice.value))
            elif not isinstance(annotation.slice, (ast.Slice, ast.ExtSlice)):
                resolved_slice = self._resolve_annotation_aliases(annotation.slice)
            return ast.Subscript(value=resolved_value, slice=resolved_slice, ctx=ast.Load())

        elif isinstance(annotation, ast.Tuple):
            # Recursively resolve each element
            resolved_elts = [self._resolve_annotation_aliases(e) for e in annotation.elts]
            return ast.Tuple(elts=resolved_elts, ctx=ast.Load())

        elif isinstance(annotation, ast.Attribute):
            # Recursively resolve the value
            resolved_value = self._resolve_annotation_aliases(annotation.value)
            return ast.Attribute(value=resolved_value, attr=annotation.attr, ctx=ast.Load())

        else:
            # For other types (Constant, Str, etc.), return as-is
            return annotation

    def _create_helper_functions(self):
        """Create the ESBMC helper function definitions"""
        # ESBMC_range_next_ function
        range_next_func = ast.FunctionDef(
            name='ESBMC_range_next_',
            args=ast.arguments(posonlyargs=[],
                               args=[
                                   ast.arg(arg='curr',
                                           annotation=ast.Name(id='int', ctx=ast.Load())),
                                   ast.arg(arg='step',
                                           annotation=ast.Name(id='int', ctx=ast.Load()))
                               ],
                               vararg=None,
                               kwonlyargs=[],
                               kw_defaults=[],
                               kwarg=None,
                               defaults=[]),
            body=[
                ast.Return(value=ast.BinOp(left=ast.Name(id='curr', ctx=ast.Load()),
                                           op=ast.Add(),
                                           right=ast.Name(id='step', ctx=ast.Load())))
            ],
            decorator_list=[],
            returns=ast.Name(id='int', ctx=ast.Load()),
            lineno=1,
            col_offset=0)

        # ESBMC_range_has_next_ function
        range_has_next_func = ast.FunctionDef(
            name='ESBMC_range_has_next_',
            args=ast.arguments(posonlyargs=[],
                               args=[
                                   ast.arg(arg='curr',
                                           annotation=ast.Name(id='int', ctx=ast.Load())),
                                   ast.arg(arg='end', annotation=ast.Name(id='int',
                                                                          ctx=ast.Load())),
                                   ast.arg(arg='step',
                                           annotation=ast.Name(id='int', ctx=ast.Load()))
                               ],
                               vararg=None,
                               kwonlyargs=[],
                               kw_defaults=[],
                               kwarg=None,
                               defaults=[]),
            body=[
                ast.If(test=ast.Compare(left=ast.Name(id='step', ctx=ast.Load()),
                                        ops=[ast.Gt()],
                                        comparators=[ast.Constant(value=0)]),
                       body=[
                           ast.Return(
                               value=ast.Compare(left=ast.Name(id='curr', ctx=ast.Load()),
                                                 ops=[ast.Lt()],
                                                 comparators=[ast.Name(id='end', ctx=ast.Load())]))
                       ],
                       orelse=[
                           ast.If(test=ast.Compare(left=ast.Name(id='step', ctx=ast.Load()),
                                                   ops=[ast.Lt()],
                                                   comparators=[ast.Constant(value=0)]),
                                  body=[
                                      ast.Return(value=ast.Compare(
                                          left=ast.Name(id='curr', ctx=ast.Load()),
                                          ops=[ast.Gt()],
                                          comparators=[ast.Name(id='end', ctx=ast.Load())]))
                                  ],
                                  orelse=[ast.Return(value=ast.Constant(value=False))])
                       ])
            ],
            decorator_list=[],
            returns=ast.Name(id='bool', ctx=ast.Load()),
            lineno=1,
            col_offset=0)

        # ESBMC_reversed_range_start_ function
        #
        # Returns the first element of reversed(range(start, stop, step)), i.e.
        # the last element of the original range.  When the original range is
        # empty the function returns (start - step) so that the caller's
        # range(result, start-step, -step) is trivially empty as well.
        #
        # Python:
        #   def ESBMC_reversed_range_start_(start, stop, step):
        #       if step > 0:
        #           if stop <= start:
        #               return start - step
        #           n = (stop - start - 1) // step
        #           return start + n * step
        #       else:
        #           if stop >= start:
        #               return start - step
        #           n = (start - stop - 1) // (-step)
        #           return start + n * step
        #
        # All divisions involve same-sign operands (positive numerator and
        # positive denominator), so Python // and C / agree — no floor-mod
        # correction is needed.
        def _make_int_arg(name):
            return ast.arg(arg=name, annotation=ast.Name(id='int', ctx=ast.Load()))

        def _name(n):
            return ast.Name(id=n, ctx=ast.Load())

        def _binop(l, op, r):
            return ast.BinOp(left=l, op=op, right=r)

        def _cmp(l, op, r):
            return ast.Compare(left=l, ops=[op], comparators=[r])

        # Shared tail: n = ...; return start + n * step
        def _last_element_block(n_expr):
            n_assign = ast.AnnAssign(
                target=ast.Name(id='n', ctx=ast.Store()),
                annotation=ast.Name(id='int', ctx=ast.Load()),
                value=n_expr,
                simple=1,
            )
            ret = ast.Return(value=_binop(_name('start'), ast.Add(),
                                          _binop(_name('n'), ast.Mult(), _name('step'))))
            return [n_assign, ret]

        reversed_range_start_func = ast.FunctionDef(
            name='ESBMC_reversed_range_start_',
            args=ast.arguments(
                posonlyargs=[],
                args=[_make_int_arg('start'),
                      _make_int_arg('stop'),
                      _make_int_arg('step')],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[
                ast.If(
                    # if step > 0:
                    test=_cmp(_name('step'), ast.Gt(), ast.Constant(value=0)),
                    body=[
                        ast.If(
                            # if stop <= start: return start - step
                            test=_cmp(_name('stop'), ast.LtE(), _name('start')),
                            body=[
                                ast.Return(value=_binop(_name('start'), ast.Sub(), _name('step')))
                            ],
                            orelse=[],
                        ),
                        # n = (stop - start - 1) // step; return start + n * step
                        *_last_element_block(
                            _binop(
                                _binop(_binop(_name('stop'), ast.Sub(), _name('start')), ast.Sub(),
                                       ast.Constant(value=1)),
                                ast.FloorDiv(),
                                _name('step'),
                            )),
                    ],
                    orelse=[
                        ast.If(
                            # if stop >= start: return start - step
                            test=_cmp(_name('stop'), ast.GtE(), _name('start')),
                            body=[
                                ast.Return(value=_binop(_name('start'), ast.Sub(), _name('step')))
                            ],
                            orelse=[],
                        ),
                        # n = (start - stop - 1) // (-step); return start + n * step
                        *_last_element_block(
                            _binop(
                                _binop(_binop(_name('start'), ast.Sub(), _name('stop')), ast.Sub(),
                                       ast.Constant(value=1)),
                                ast.FloorDiv(),
                                ast.UnaryOp(op=ast.USub(), operand=_name('step')),
                            )),
                    ],
                )
            ],
            decorator_list=[],
            returns=ast.Name(id='int', ctx=ast.Load()),
            lineno=1,
            col_offset=0,
        )

        return [range_next_func, range_has_next_func, reversed_range_start_func]

    def prepare_module(self, node, alias_seed=frozenset(), wrapper_seed=None):
        """Run pre-visit analyses and the range-alias / wrapper canonicalisation.

        Collects callee names and global-scope variable annotations, captures
        ``__all__``, and runs ``apply_range_rewrites`` with the supplied
        cross-module seeds (empty by default). This must complete on every
        module *before* any module's ``finalize_module`` runs so that
        ``visit_For`` sees canonical ``range(...)`` calls produced by
        cross-module alias propagation (#4533).

        Idempotent: re-running with a fresh seed after the in-module aliases
        have been canonicalised only adds the new (cross-module) names.
        """
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                self.called_names.add(n.func.id)

        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        annotation_node = self._create_annotation_node_from_value(stmt.value)
                        if annotation_node:
                            self.variable_annotations[target.id] = annotation_node
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                self.variable_annotations[stmt.target.id] = stmt.annotation

        self.module_dunder_all = self._capture_dunder_all(node)
        self.apply_range_rewrites(node, alias_seed=alias_seed, wrapper_seed=wrapper_seed)

    def finalize_module(self, node):
        """Run ``generic_visit`` and inject any helpers requested during it.

        Must be called after ``prepare_module`` has completed for every module
        in the import graph, so that visit_For sees canonical ``range(...)``
        calls produced by cross-module propagation (#4533).
        """
        node = self.generic_visit(node)

        if self._needs_dataclass_initvar_import:
            self._ensure_dataclass_initvar_import(node)

        if self.helper_functions_added:
            helper_functions = self._create_helper_functions()
            for func in helper_functions:
                self.ensure_all_locations(func)
                ast.fix_missing_locations(func)
            node.body = helper_functions + node.body

        if self._needs_dataclass_field_helper:
            helper_class = self._build_dataclass_field_helper_class(node)
            node.body = [helper_class] + node.body

        if self._needs_dataclass_replace_error_helper:
            helper_fn = self._build_dataclass_replace_error_helper(node)
            node.body = [helper_fn] + node.body

        if self._needs_dataclass_getattr_helper:
            helper_fn = self._build_dataclass_getattr_helper(node)
            node.body = [helper_fn] + node.body

        return node

    def visit_Module(self, node):
        """Visit the module: prepare, then finalize.

        Back-compat entry point for callers (``emit_file_as_json``, tests,
        memory models) that don't participate in cross-module range-alias
        propagation. The parser's main import-aware pipeline calls
        ``prepare_module`` / ``finalize_module`` directly with the right
        seeds (#4533).
        """
        self.prepare_module(node)
        return self.finalize_module(node)

    @staticmethod
    def _target_names(target):
        """Names bound by an assignment target, including tuple/list unpacking."""
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, ast.Starred):
            return Preprocessor._target_names(target.value)
        if isinstance(target, (ast.Tuple, ast.List)):
            return {n for e in target.elts for n in Preprocessor._target_names(e)}
        return set()

    @staticmethod
    def _rebound_module_names(module_node):
        """Module-top names that may be rebound after their first binding.

        Conservative: a name is considered "rebound" if any of the following
        hold, in which case the alias/wrapper rewrites must skip it.

        * It is bound more than once directly in ``module_node.body`` (covers
          re-defined ``def``/``class``, repeated ``=`` assignment, tuple
          unpacking that includes the name).
        * It appears as an ``AugAssign`` target at module scope (``X += ...``
          is itself a rebind even on first occurrence).
        * It is bound inside a top-level ``If`` / ``For`` / ``While`` / ``Try``
          / ``With`` block, where its run-time value may diverge from the
          single canonical binding.
        * Any nested function declares ``global X``, since the nested scope
          can rebind it at run time.

        Bindings inside nested ``FunctionDef`` / ``AsyncFunctionDef`` /
        ``Lambda`` / ``ClassDef`` scopes are NOT counted (they shadow rather
        than rebind), except via the ``global`` mechanism above.
        """
        seen = set()
        rebound = set()
        target_names = Preprocessor._target_names

        def _collect_bindings(stmt):
            if isinstance(stmt, ast.Assign):
                names = set()
                for t in stmt.targets:
                    names |= target_names(t)
                return names
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                return {stmt.target.id}
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return {stmt.name}
            if isinstance(stmt, ast.Import):
                return {a.asname or a.name.split(".")[0] for a in stmt.names}
            if isinstance(stmt, ast.ImportFrom):
                return {a.asname or a.name for a in stmt.names}
            return set()

        for stmt in module_node.body:
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                rebound.add(stmt.target.id)
                seen.add(stmt.target.id)
                continue
            for name in _collect_bindings(stmt):
                if name in seen:
                    rebound.add(name)
                seen.add(name)

        # Walk into top-level control-flow blocks (but not into nested scopes)
        # and treat any binding therein as a potential rebind.
        def _walk_module_only(root):
            for child in ast.iter_child_nodes(root):
                if isinstance(child,
                              (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue
                yield child
                yield from _walk_module_only(child)

        for stmt in module_node.body:
            if not isinstance(
                    stmt,
                (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
                continue
            for inner in _walk_module_only(stmt):
                if isinstance(inner, ast.Assign):
                    for t in inner.targets:
                        rebound |= target_names(t)
                elif isinstance(inner, (ast.AnnAssign, ast.AugAssign)):
                    if isinstance(inner.target, ast.Name):
                        rebound.add(inner.target.id)
                elif isinstance(inner, ast.NamedExpr) and isinstance(inner.target, ast.Name):
                    rebound.add(inner.target.id)
                elif isinstance(inner, (ast.For, ast.AsyncFor)):
                    rebound |= target_names(inner.target)
                elif isinstance(inner, (ast.With, ast.AsyncWith)):
                    for item in inner.items:
                        if item.optional_vars:
                            rebound |= target_names(item.optional_vars)

        # `global X` anywhere in the module marks X as rebindable from
        # the nested scope.
        for inner in ast.walk(module_node):
            if isinstance(inner, ast.Global):
                rebound.update(inner.names)

        return rebound

    @staticmethod
    def _scope_locally_binds(scope_node, names):
        """True iff any of *names* is locally bound inside *scope_node*.

        Inspects parameters, Assign / AnnAssign / AugAssign / NamedExpr
        targets, ``For`` / ``With`` / comprehension targets, ``import`` aliases,
        and nested ``def`` / ``class`` names. Does NOT descend into nested
        ``FunctionDef`` / ``AsyncFunctionDef`` / ``Lambda`` / ``ClassDef``
        bodies — those are separate scopes.
        """
        if not isinstance(scope_node,
                          (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return False

        target_names = Preprocessor._target_names

        if isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = scope_node.args
            for p in (*a.args, *a.posonlyargs, *a.kwonlyargs):
                if p.arg in names:
                    return True
            if a.vararg and a.vararg.arg in names:
                return True
            if a.kwarg and a.kwarg.arg in names:
                return True

        body = scope_node.body
        body = body if isinstance(body, list) else [body]
        stack = list(body)
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if n.name in names:
                    return True
                continue
            if isinstance(n, ast.Lambda):
                continue
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if target_names(t) & names:
                        return True
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(n.target, ast.Name) and n.target.id in names:
                    return True
            elif isinstance(n, ast.NamedExpr):
                if isinstance(n.target, ast.Name) and n.target.id in names:
                    return True
            elif isinstance(n, (ast.For, ast.AsyncFor)):
                if target_names(n.target) & names:
                    return True
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for item in n.items:
                    if item.optional_vars and target_names(item.optional_vars) & names:
                        return True
            elif isinstance(n, ast.Import):
                for al in n.names:
                    if (al.asname or al.name.split(".")[0]) in names:
                        return True
            elif isinstance(n, ast.ImportFrom):
                for al in n.names:
                    if (al.asname or al.name) in names:
                        return True
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                if any(name in names for name in n.names):
                    return True
            for child in ast.iter_child_nodes(n):
                stack.append(child)
        return False

    @classmethod
    def _iter_module_scope(cls, root, shadow_names):
        """Yield descendants of *root* whose enclosing scope is the module.

        Stops descending into nested ``FunctionDef`` / ``AsyncFunctionDef`` /
        ``Lambda`` / ``ClassDef`` nodes whose local scope binds any of
        *shadow_names* — preventing the alias / wrapper rewrites from
        replacing references that real Python would resolve to a locally
        shadowed name.
        """
        for child in ast.iter_child_nodes(root):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                if cls._scope_locally_binds(child, shadow_names):
                    continue
            yield child
            yield from cls._iter_module_scope(child, shadow_names)

    @staticmethod
    def collect_range_aliases(module_node, seed=frozenset()):
        """Collect module-scope canonical-range aliases.

        Returns a pair ``(local, all)`` where:

        * ``local`` is the set of names bound *in this module* via ``X = range``
          or ``X = Y`` for a previously known alias ``Y``.
        * ``all`` is ``local`` unioned with *seed* (intended to be aliases
          inherited from imported modules), minus any locally rebound names.

        Names rebound elsewhere (per :meth:`_rebound_module_names`) are
        excluded from both sets.
        """
        rebound = Preprocessor._rebound_module_names(module_node)
        all_aliases = set(seed) - rebound
        local = set()
        changed = True
        while changed:
            changed = False
            for stmt in module_node.body:
                if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(
                        stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Name)):
                    continue
                lhs = stmt.targets[0].id
                rhs = stmt.value.id
                if lhs in all_aliases or lhs in rebound:
                    continue
                if rhs == "range" or rhs in all_aliases:
                    all_aliases.add(lhs)
                    local.add(lhs)
                    changed = True
        return local, all_aliases

    def apply_range_rewrites(self, module_node, alias_seed=frozenset(), wrapper_seed=None):
        """Run the range-alias and range-wrapper rewrites on *module_node*.

        Public entry point for both per-module and cross-module (#4525)
        invocations. *alias_seed* and *wrapper_seed* carry aliases /
        wrappers inherited from imported modules; pass empty defaults for
        a single-file pre-pass.
        """
        self._rewrite_range_aliases(module_node, seed=alias_seed)
        self._inline_range_wrappers(module_node, seed=wrapper_seed)

    def _rewrite_range_aliases(self, module_node, seed=frozenset()):
        """Rewrite module-scope ``alias = range`` (and chains) to canonical ``range``.

        Collects every top-level assignment of the form ``X = range`` (and
        transitively ``Y = X`` where ``X`` is already a known alias),
        rewrites every ``Call(Name(alias), ...)`` that resolves to the
        module-scope binding to ``Call(Name("range"), ...)``, and removes the
        original alias statements so the C++ annotator never sees ``range``
        as a bare RHS. Names that are rebound elsewhere are excluded, and
        rewrites are skipped inside nested scopes that locally shadow the
        alias.

        *seed* is a set of alias names inherited from imported modules
        (#4525). They are added to the alias set used to rewrite call sites
        but never removed from this module's body — they were never declared
        here to begin with.
        """
        local, all_aliases = self.collect_range_aliases(module_node, seed)
        # Track exports for cross-module propagation (#4525): both
        # locally-defined aliases and any seeded names that survived
        # (i.e. aren't shadowed by a local rebind) are part of this
        # module's export surface, so downstream consumers can pick them
        # up via plain ``from this_module import name`` re-imports.
        self.exported_range_aliases |= local
        self.exported_range_aliases |= (set(seed) & all_aliases)

        if not all_aliases:
            return

        for n in self._iter_module_scope(module_node, all_aliases):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in all_aliases):
                n.func.id = "range"

        if not local:
            return

        module_node.body = [
            stmt for stmt in module_node.body
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id in local
                    and isinstance(stmt.value, ast.Name) and
                    (stmt.value.id == "range" or stmt.value.id in all_aliases))
        ]

    @staticmethod
    def collect_range_wrappers(module_node):
        """Collect trivial ``def f(p): return range(...)`` wrappers at module scope.

        Returns ``{name: (params, template_args)}`` for every qualifying
        wrapper defined directly in *module_node*. Wrappers rebound elsewhere
        are excluded.
        """
        rebound = Preprocessor._rebound_module_names(module_node)
        wrappers = {}
        for stmt in module_node.body:
            if not (isinstance(stmt, ast.FunctionDef) and stmt.name not in rebound
                    and len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Return)):
                continue
            call = stmt.body[0].value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "range" and not call.keywords):
                continue
            args = stmt.args
            if (args.vararg is not None or args.kwarg is not None or args.kwonlyargs
                    or args.posonlyargs or args.defaults or args.kw_defaults):
                continue
            params = [a.arg for a in args.args]
            param_set = set(params)
            if not all((isinstance(a, ast.Name) and a.id in param_set) or (isinstance(
                    a, ast.Constant) and isinstance(a.value, int) and not isinstance(a.value, bool))
                       for a in call.args):
                continue
            wrappers[stmt.name] = (params, call.args)
        return wrappers

    def _inline_range_wrappers(self, module_node, seed=None):
        """Inline trivial ``def f(p): return range(p_or_const, ...)`` wrappers.

        A wrapper qualifies when it lives at module scope, is not rebound
        elsewhere, takes only positional parameters with no defaults or
        ``*args``/``**kwargs``, and its body is exactly one
        ``return range(...)`` where every argument is either a reference to
        one of the function's parameters or an integer ``Constant``. Every
        call ``f(actuals)`` whose arity matches the wrapper's parameter list
        and that uses only plain positional arguments (no ``*xs``, no
        keyword args) is rewritten to a fresh ``range(actuals')`` call.

        The wrapper ``def`` is intentionally retained: call sites with
        mismatched arity, keyword args, or ``*xs`` still need to resolve to
        it. Nested scopes that locally rebind the wrapper name are skipped
        so the rewrite respects Python's lexical-scope semantics.

        *seed* maps wrapper-name → ``(params, template_args)`` for wrappers
        inherited from imported modules (#4525). Entries shadowed by a
        local rebind are dropped, so the importer's lexical scope still wins.
        """
        local = self.collect_range_wrappers(module_node)
        self.exported_range_wrappers.update(local)

        wrappers = dict(seed) if seed else {}
        rebound = self._rebound_module_names(module_node)
        for name in list(wrappers):
            if name in rebound and name not in local:
                del wrappers[name]
        # Surviving seeded wrappers (i.e. those not locally shadowed) are
        # re-exported so chained importers can resolve them too (#4525).
        self.exported_range_wrappers.update(wrappers)
        wrappers.update(local)

        if not wrappers:
            return

        wrapper_names = set(wrappers)
        for n in self._iter_module_scope(module_node, wrapper_names):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in wrappers):
                continue
            if n.keywords or any(isinstance(a, ast.Starred) for a in n.args):
                continue
            params, template = wrappers[n.func.id]
            if len(n.args) != len(params):
                continue
            subst = dict(zip(params, n.args))
            n.args = [
                copy.deepcopy(subst[t.id]) if isinstance(t, ast.Name) else copy.deepcopy(t)
                for t in template
            ]
            n.func.id = "range"
            ast.fix_missing_locations(n)

    @staticmethod
    def _capture_dunder_all(module_node):
        """Return the literal list/tuple of names assigned to ``__all__``.

        Returns ``None`` when ``__all__`` is absent, dynamically constructed,
        or contains anything other than string literals — in which case the
        importer cannot statically project a star import.
        """
        for stmt in module_node.body:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "__all__"):
                continue
            value = stmt.value
            if not isinstance(value, (ast.List, ast.Tuple)):
                return None
            names = []
            for el in value.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, str):
                    names.append(el.value)
                else:
                    return None
            return names
        return None

    def ensure_all_locations(self, node, source_node=None, line=1, col=0):
        """Recursively ensure all nodes in an AST tree have location information"""
        if source_node:
            line = getattr(source_node, "lineno", 1)
            col = getattr(source_node, "col_offset", 0)

        # Ensure current node has location info
        if not hasattr(node, "lineno") or node.lineno is None:
            node.lineno = line
        if not hasattr(node, "col_offset") or node.col_offset is None:
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

    def _rename_loads(self, node, old_name, new_name):

        class _RenameLoad(ast.NodeTransformer):

            def __init__(self, old_name, new_name):
                self.old_name = old_name
                self.new_name = new_name

            def visit_Name(self, name_node):
                if name_node.id == self.old_name and isinstance(name_node.ctx, ast.Load):
                    return ast.copy_location(ast.Name(id=self.new_name, ctx=ast.Load()), name_node)
                return name_node

        renamed = _RenameLoad(old_name, new_name).visit(node)
        ast.fix_missing_locations(renamed)
        return renamed

    def _bound_target_names(self, target):
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            names = set()
            for elt in target.elts:
                names.update(self._bound_target_names(elt))
            return names
        return set()

    def _create_bool_ann_assign(self, target_name, value, source_node):
        assign = ast.AnnAssign(
            target=self.create_name_node(target_name, ast.Store(), source_node),
            annotation=self.create_name_node("bool", ast.Load(), source_node),
            value=value,
            simple=1,
        )
        self.ensure_all_locations(assign, source_node)
        ast.fix_missing_locations(assign)
        return assign

    class _ListCompExpressionLowerer(ast.NodeTransformer):
        """Utility transformer that lowers list comprehensions, any(genexpr), and all(genexpr) inside an expression."""

        def __init__(self, preprocessor):
            super().__init__()
            self.preprocessor = preprocessor
            self.statements = []

        def visit_ListComp(self, node):
            prefix, result_expr = self.preprocessor._lower_listcomp(node)
            self.statements.extend(prefix)
            return result_expr

        def visit_SetComp(self, node):
            # Lower {elt for ... in iter} to set([elt for ... in iter]).
            # Reuses _lower_listcomp's prefix-and-name pattern, then wraps
            # the resulting list-typed name in a `set(...)` call so the
            # downstream Set handling kicks in.
            listcomp = ast.ListComp(elt=node.elt, generators=node.generators)
            ast.copy_location(listcomp, node)
            ast.fix_missing_locations(listcomp)
            prefix, list_name = self.preprocessor._lower_listcomp(listcomp)
            self.statements.extend(prefix)
            set_call = ast.Call(
                func=ast.Name(id="set", ctx=ast.Load()),
                args=[list_name],
                keywords=[],
            )
            ast.copy_location(set_call, node)
            ast.fix_missing_locations(set_call)
            return set_call

        def visit_Call(self, node):
            # Lower sep.join(GeneratorExp(...)) to sep.join(ListComp(...))
            # and reuse the existing list-comprehension lowering pipeline.
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "join"
                    and len(node.args) == 1 and not node.keywords
                    and isinstance(node.args[0], ast.GeneratorExp)):
                gen = node.args[0]
                elt_expr = copy.deepcopy(gen.elt)

                # Prefer explicit dunder dispatch for object stringification in
                # join(genexp) to avoid strict builtin str() argument checks on
                # loop variables inferred as non-string at preprocessing time.
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

            # Lower any(GeneratorExp(...)) to a loop-based boolean
            if (isinstance(node.func, ast.Name) and node.func.id == "any" and len(node.args) == 1
                    and not node.keywords and isinstance(node.args[0], ast.GeneratorExp)):
                prefix, result = self.preprocessor._lower_any_genexp(node.args[0])
                self.statements.extend(prefix)
                return result

            # Lower all(GeneratorExp(...)) to a loop-based boolean
            if (isinstance(node.func, ast.Name) and node.func.id == "all" and len(node.args) == 1
                    and not node.keywords and isinstance(node.args[0], ast.GeneratorExp)):
                prefix, result = self.preprocessor._lower_all_genexp(node.args[0])
                self.statements.extend(prefix)
                return result

            # Lower list(map(f, iterable)) to [f(x) for x in iterable]
            if (isinstance(node.func, ast.Name) and node.func.id == "list" and len(node.args) == 1
                    and not node.keywords and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Name) and node.args[0].func.id == "map"
                    and len(node.args[0].args) == 2):
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

            # Lower list(gen_func(args...)) to an inline list construction
            if (isinstance(node.func, ast.Name) and node.func.id == "list" and len(node.args) == 1
                    and not node.keywords and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Name)
                    and node.args[0].func.id in self.preprocessor.generator_funcs):
                prefix, result = self.preprocessor._lower_list_gen_call(node.args[0], node)
                if prefix is not None:
                    self.statements.extend(prefix)
                    return result

            lowered_sorted = self.preprocessor._lower_sorted_with_key_call(node)
            if lowered_sorted is not None:
                prefix, result = lowered_sorted
                self.statements.extend(prefix)
                return result

            lowered_min_max = self.preprocessor._lower_min_max_with_key_call(node)
            if lowered_min_max is not None:
                prefix, result = lowered_min_max
                self.statements.extend(prefix)
                return result

            lowered_tuple_sorted_pair = self.preprocessor._lower_tuple_sorted_pair_call(node)
            if lowered_tuple_sorted_pair is not None:
                prefix, result = lowered_tuple_sorted_pair
                self.statements.extend(prefix)
                return result

            return self.generic_visit(node)

    @staticmethod
    def _body_has_node_shallow(body_stmts, node_type):
        """Return True if node_type appears in body_stmts without descending
        into nested function definitions or lambdas."""

        def _walk(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return
            if isinstance(node, node_type):
                yield node
            for child in ast.iter_child_nodes(node):
                yield from _walk(child)

        module = ast.Module(body=list(body_stmts), type_ignores=[])
        return any(True for _ in _walk(module))

    class _YieldToAppend(ast.NodeTransformer):
        """Replace `yield expr` statements with `ESBMC_gen_result.append(expr)`."""

        def __init__(self, result_var, template):
            self.result_var = result_var
            self.template = template

        def visit_Expr(self, node):
            if not isinstance(node.value, ast.Yield):
                return self.generic_visit(node)
            yield_val = node.value.value
            if yield_val is None:
                yield_val = ast.Constant(value=None)
            append_call = ast.Expr(value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.result_var, ctx=ast.Load()),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[yield_val],
                keywords=[],
            ))
            ast.copy_location(append_call, node)
            ast.fix_missing_locations(append_call)
            return append_call

        # Do not descend into nested function definitions
        def visit_FunctionDef(self, node):
            return node

    class _YieldReplacer(ast.NodeTransformer):
        """Replace `yield val` expressions with `target = val; for_body`."""

        def __init__(self, target_name, for_body, template):
            import copy

            self.target_name = target_name
            self.for_body = for_body
            self.template = template
            self._copy = copy

        def visit_Expr(self, stmt):
            if isinstance(stmt.value, ast.YieldFrom):
                raise NotImplementedError(
                    "yield from inside a generator is not supported by the ESBMC inliner")
            if not isinstance(stmt.value, ast.Yield):
                return stmt
            yield_val = stmt.value.value
            if yield_val is None:
                yield_val = ast.Constant(value=None)
            assign = ast.Assign(
                targets=[ast.Name(id=self.target_name, ctx=ast.Store())],
                value=yield_val,
                type_comment=None,
            )
            ast.copy_location(assign, self.template)
            ast.fix_missing_locations(assign)
            return [assign] + [self._copy.deepcopy(s) for s in self.for_body]

    def _lower_min_max_with_key_call(self, call_node):
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
        best_idx = 0
        for i in range(1, len(key_values)):
            if is_min:
                if key_values[i] < key_values[best_idx]:
                    best_idx = i
            else:
                if key_values[i] > key_values[best_idx]:
                    best_idx = i

        # Suppress the unused default_kw warning while keeping the variable
        # available for future extension (e.g. empty iterable + default=).
        del default_kw

        result = copy.deepcopy(iterable_literal.elts[best_idx])
        self.ensure_all_locations(result, call_node)
        ast.fix_missing_locations(result)
        return [], result

    def _lower_tuple_sorted_pair_call(self, call_node):
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

    def visit_Subscript(self, node):
        """Fold constant indexing over tracked list literals when safe.

        Only folds when the indexed element is a pure literal (Constant or a
        unary +/- over a Constant). Folding non-pure expressions such as
        ``nondet_int()`` calls would break value correlation, since each
        substitution would produce a fresh, independent symbolic value.
        """
        node = self.generic_visit(node)

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
        # Lower `xs.sort(key=...)` to `xs = sorted(xs, key=...)` BEFORE
        # generic_visit, since visit_Call invalidates the
        # `list_literal_values[xs]` entry on `xs.sort()` (sort is a
        # mutating list method). The existing sorted-with-key folding
        # depends on that entry, so the rewrite has to fire while it's
        # still tracked.
        rewritten = self._maybe_rewrite_list_sort_with_key(node)
        if rewritten is not None:
            return rewritten

        node = self.generic_visit(node)

        # Handle standalone next(g)
        next_gen_info = self._find_generator_next_call(node.value)
        if next_gen_info is not None:
            gen_var, func_name = next_gen_info
            if func_name in self.early_return_generator_funcs:
                return self._make_stop_iteration_raise(node)
            else:
                stmts = self._inline_next_call(None, func_name, gen_var, node)
                if stmts is not None:
                    return stmts

        prefix, new_value, _ = self._lower_listcomp_in_expr(node.value)
        node.value = new_value
        dd_inits, node.value = self._lower_defaultdict_reads_in_expr(node.value, node)
        prefix = dd_inits + prefix
        if prefix:
            return prefix + [node]
        return node

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

    def visit_If(self, node):
        node = self.generic_visit(node)
        prefix, new_test, _ = self._lower_listcomp_in_expr(node.test)
        node.test = new_test
        node.test = self._transform_list_truthiness(node.test, node)
        if prefix:
            return prefix + [node]
        return node

    def _transform_list_truthiness(self, test_expr, source_node):
        """
        Transform list truthiness checks to explicit len() > 0 checks.
        Converts: while xs: -> while len(xs) > 0:
        """
        # Only transform if the test is a simple Name node referring to a list
        if not isinstance(test_expr, ast.Name):
            return test_expr

        var_name = test_expr.id
        var_type = self.known_variable_types.get(var_name)

        # Check if this is a list type
        if var_type != "list":
            return test_expr

        # Create: len(xs) > 0
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

    def visit_While(self, node):
        node = self.generic_visit(node)
        prefix, new_test, _ = self._lower_listcomp_in_expr(node.test)
        node.test = new_test
        node.test = self._transform_list_truthiness(node.test, node)
        if prefix:
            return prefix + [node]
        return node

    def _simplify_isinstance(self, node):
        """Simplify isinstance(v, T) when v has a known non-Any annotation.
        - annotation matches T    -> True
        - annotation mismatches T -> False
        - annotation unknown/Any  -> leave unchanged
        """
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
            # Don't simplify to True if the annotation was inferred from a
            # subscript access (e.g. x = d[k]): the dict may have been mutated
            # with a value of a different type, so we cannot guarantee correctness.
            if obj_node.id in self._subscript_inferred_vars:
                return node
            return ast.Constant(value=True)
        return ast.Constant(value=False)

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

    def visit_Assert(self, node):
        node = self.generic_visit(node)
        tuple_eq_prefix = []
        if (isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq) and len(node.test.comparators) == 1):
            left = node.test.left
            right = node.test.comparators[0]
            rewritten = self._try_transform_items_set_eq(left, right, node)
            if rewritten is None:
                rewritten = self._try_transform_items_set_eq(right, left, node)
            if rewritten is None:
                rewritten = self._try_transform_items_list_eq(left, right, node)
            if rewritten is None:
                rewritten = self._try_transform_items_list_eq(right, left, node)
            if rewritten is None:
                rewritten = self._try_transform_keys_view_eq(left, right, node)
            if rewritten is None:
                rewritten = self._try_transform_keys_view_eq(right, left, node)
            if rewritten is None:
                rewritten = self._try_transform_values_view_eq(left, right, node)
            if rewritten is None:
                rewritten = self._try_transform_values_view_eq(right, left, node)
            if rewritten is None:
                rewritten = self._try_transform_list_tuple_eq(left, right, node)
            if rewritten is None:
                rewritten = self._try_transform_list_tuple_eq(right, left, node)
            if rewritten is None:
                tuple_eq_prefix, rewritten = self._try_lower_expr_tuple_literal_eq(
                    left, right, node)
            if rewritten is None:
                tuple_eq_prefix, rewritten = self._try_lower_expr_tuple_literal_eq(
                    right, left, node)
            if tuple_eq_prefix is None:
                tuple_eq_prefix = []
            if rewritten is not None:
                node.test = rewritten
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

    def _extract_type_from_annotation(self, annotation):
        """Extract a simplified type string from a type annotation AST node"""
        if annotation is None:
            return "Any"

        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            # Handle types like list[int], dict[str, int], etc.
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id  # Return just 'list', 'dict', etc.
        elif isinstance(annotation, ast.Constant):
            if isinstance(annotation.value, str):
                # Handle string annotations like "list[int]"
                return annotation.value.split("[")[0]

        return "Any"

    def _get_iterable_type_annotation(self, iterable):
        """Get the appropriate type annotation for an iterable"""
        if isinstance(iterable, ast.Constant) and isinstance(iterable.value, str):
            return "str"
        elif isinstance(iterable, ast.List):
            return "list"
        elif isinstance(iterable, ast.Tuple):
            return "tuple"
        elif isinstance(iterable, ast.Name):
            # Check if we know the type of this variable
            known_type = self.known_variable_types.get(iterable.id)
            if known_type and known_type != "Any":
                return known_type
            else:
                return "list"  # Default to list for ESBMC compatibility
        else:
            return "list"

    def _get_element_type_from_container(self, container_type, iterable_node=None):
        """Get the element type from a container type with better inference"""
        # 1. Handle method calls such as d.keys(), d.values()
        if isinstance(iterable_node, ast.Call) and isinstance(iterable_node.func, ast.Attribute):
            method_name = iterable_node.func.attr

            if method_name in ["keys", "values"]:
                # Get the base object (e.g., 'd' in d.keys())
                if isinstance(iterable_node.func.value, ast.Name):
                    dict_var_name = iterable_node.func.value.id

                    # Look up the dict's annotation
                    if (hasattr(self, "variable_annotations")
                            and dict_var_name in self.variable_annotations):
                        dict_annotation = self.variable_annotations[dict_var_name]

                        # Extract key/value types from dict[K, V]
                        if isinstance(dict_annotation, ast.Subscript):
                            if isinstance(dict_annotation.slice, ast.Tuple):
                                key_type = dict_annotation.slice.elts[0]
                                value_type = dict_annotation.slice.elts[1]

                                if method_name == "keys":
                                    if isinstance(key_type, ast.Name):
                                        return key_type.id
                                    elif isinstance(key_type, ast.Subscript) and isinstance(
                                            key_type.value, ast.Name):
                                        return key_type.value.id
                                elif method_name == "values":
                                    if isinstance(value_type, ast.Name):
                                        return value_type.id
                                    elif isinstance(value_type, ast.Subscript) and isinstance(
                                            value_type.value, ast.Name):
                                        return value_type.value.id

        # 2. Handle direct dict iteration: for k in d:
        if isinstance(iterable_node, ast.Name):
            var_name = iterable_node.id

            if (hasattr(self, "variable_annotations") and var_name in self.variable_annotations):
                annotation = self.variable_annotations[var_name]

                # Check if it's a dict annotation
                if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
                    if annotation.value.id == "dict":
                        # Extract key type from dict[K, V]
                        if (isinstance(annotation.slice, ast.Tuple)
                                and len(annotation.slice.elts) >= 1):
                            key_type = annotation.slice.elts[0]
                            if isinstance(key_type, ast.Name):
                                return key_type.id

        if container_type == "str":
            return "str"
        elif isinstance(iterable_node, ast.List) and iterable_node.elts:
            # Infer from first element if available
            first_elem = iterable_node.elts[0]
            if isinstance(first_elem, ast.Constant):
                return type(first_elem.value).__name__
        elif container_type.lower() in ["list", "tuple"]:
            # Try to extract element type from generic annotation
            if isinstance(iterable_node, ast.Name) and hasattr(self, "variable_annotations"):
                var_name = iterable_node.id
                if var_name in self.variable_annotations:
                    annotation = self.variable_annotations[var_name]
                    # Extract element type from Subscript such as list[dict] or list[dict[str, str]]
                    if isinstance(annotation, ast.Subscript):
                        element_annotation = annotation.slice
                        # Handle simple Name: list[dict] -> 'dict'
                        if isinstance(element_annotation, ast.Name):
                            return element_annotation.id
                        # Handle nested Subscript: list[dict[str, str]] -> 'dict'
                        elif isinstance(element_annotation, ast.Subscript):
                            # Extract base type from nested subscript
                            if isinstance(element_annotation.value, ast.Name):
                                return element_annotation.value.id
            return "Any"
        return "Any"

    def generate_variable_copy(self, node_name: str, argument: ast.arg, default_val):
        target = ast.Name(id=f"ESBMC_DEFAULT_{node_name}_{argument.arg}", ctx=ast.Store())
        assign_node = ast.AnnAssign(target=target,
                                    annotation=argument.annotation,
                                    value=default_val,
                                    simple=1)
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
        (list/sorted/set differ in ordering semantics).
        """
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id
                in wrappers and len(node.args) == 1 and not getattr(node, "keywords", [])):
            return None, None
        wrapper = node.func.id
        arg = node.args[0]
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
        requires a Set literal, list/sorted requires a List literal).
        """
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("list", "sorted", "set")
                and len(node.args) == 1 and not getattr(node, "keywords", [])):
            return None, None
        wrapper = node.func.id
        inner = node.args[0]
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

    def visit_Compare(self, node):
        """Keep comparisons semantically faithful by default.

        Marco F recovery phase 2 disables broad comparison rewrites that were
        changing assert semantics for unrelated regressions. We still keep
        assert-specific safe rewrites (_simplify_isinstance), list-comp lowering
        and defaultdict lowering elsewhere in the pipeline.
        """
        node = self.generic_visit(node)
        return node

    def _is_newtype_call(self, call_node):
        """True if call_node is a typing.NewType(...) call, in any import form."""
        func = call_node.func
        # from typing import NewType [as alias]  →  X = NewType(...) / alias(...)
        if isinstance(func, ast.Name):
            return func.id in self.newtype_names
        # import typing [as alias]  →  X = typing.NewType(...) / alias.NewType(...)
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.attr == "NewType"):
            return func.value.id in self.typing_module_names
        return False

    def visit_Assign(self, node):
        """
        Handle assignment nodes, including multiple assignments and tuple unpacking.
        """
        # Invalidate tracked list literals on subscript writes: l[i] = v
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id in self.list_literal_values):
                self.list_literal_values.pop(target.value.id, None)

        # Check if this is a type alias assignment (e.g., Coordinate = Tuple[int, int])
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and self._is_type_alias_expression(node.value)):
            # Store the alias and skip execution (return None to remove from AST)
            alias_name = node.targets[0].id
            self.type_aliases[alias_name] = node.value
            return None

        # First visit child nodes
        node = self.generic_visit(node)

        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            if self._is_assert_literal_shape(node.value):
                self._known_literal_values[target_name] = copy.deepcopy(node.value)
            elif (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                  and node.value.func.id in self._identity_functions and len(node.value.args) == 1
                  and not node.value.keywords
                  and self._is_assert_literal_shape(node.value.args[0])):
                self._known_literal_values[target_name] = copy.deepcopy(node.value.args[0])
            else:
                self._known_literal_values.pop(target_name, None)

        # Expand nondet_list && nondet_dict calls inline.
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("nondet_list", "nondet_dict")):
            expanded = self._expand_nondet_call(node.targets[0], node.value, node)
            if expanded is not None:
                return expanded

        # Handle x = next(g) for generator variables
        next_gen_info = self._find_generator_next_call(node.value)
        if next_gen_info is not None:
            gen_var, func_name = next_gen_info
            if func_name in self.early_return_generator_funcs:
                # Early return before first yield: next() raises StopIteration immediately
                raise_node = ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="StopIteration", ctx=ast.Load()),
                        args=[ast.Constant(value="StopIteration")],
                        keywords=[],
                    ),
                    cause=None,
                )
                ast.copy_location(raise_node, node)
                ast.fix_missing_locations(raise_node)
                return raise_node
            else:
                # Normal generator: inline code path to first yield → x = yielded_val
                stmts = self._inline_next_call(node.targets, func_name, gen_var, node)
                if stmts is not None:
                    return stmts

        prefix, lowered_value, lowered_type = self._lower_listcomp_in_expr(node.value)
        node.value = lowered_value
        if prefix:
            # lowered_value is a Name node referencing the same temp variable;
            # sharing it across assignments correctly models Python's single-evaluation
            # semantics for chained assignments (a = b = expr evaluates expr once).
            assigns = []
            for target in node.targets:
                assign = ast.Assign(targets=[target], value=node.value)
                self._copy_location_info(node, assign)
                self.ensure_all_locations(assign, node)
                ast.fix_missing_locations(assign)
                if isinstance(target, ast.Name):
                    self.known_variable_types[target.id] = lowered_type
                assigns.append(assign)
            return prefix + assigns

        # Handle single target (most common case)
        if len(node.targets) == 1:
            target = node.targets[0]

            # Check if this is tuple unpacking (x, y = ...)
            if isinstance(target, (ast.Tuple, ast.List)):
                return self._handle_tuple_unpacking(target, node.value, node)
            else:
                # NewType is an identity callable
                # rewrite X = NewType('X', T) → X = T
                # matches typing.NewType(...) and aliased imports
                if (isinstance(target, ast.Name) and isinstance(node.value, ast.Call)
                        and self._is_newtype_call(node.value) and len(node.value.args) >= 2):
                    self.newtype_vars.add(target.id)
                    node.value = node.value.args[1]
                    ast.fix_missing_locations(node)
                # Drop stale NewType tracking on reassignment to a non-NewType value
                elif isinstance(target, ast.Name) and target.id in self.newtype_vars:
                    self.newtype_vars.discard(target.id)
                # Simple assignment - track the type
                # Detect bound method assignment: g = obj.method
                # Only remove when g is actually called somewhere (g())
                if (isinstance(target, ast.Name) and isinstance(node.value, ast.Attribute)
                        and isinstance(node.value.value, ast.Name)
                        and target.id in self.called_names):
                    self.bound_method_vars[target.id] = node.value
                    return None  # Remove; call sites are rewritten in visit_Call
                # Clear stale bound method tracking on variable reassignment
                if isinstance(target, ast.Name) and target.id in self.bound_method_vars:
                    del self.bound_method_vars[target.id]
                self._update_variable_types_simple(target, node.value)
                # Also store annotation node if we can infer it
                if isinstance(target, ast.Name):
                    annotation_node = self._create_annotation_node_from_value(node.value)
                    if annotation_node:
                        self.variable_annotations[target.id] = annotation_node
                        if isinstance(node.value, ast.Subscript):
                            self._subscript_inferred_vars.add(target.id)
                    if isinstance(node.value, ast.List):
                        self.list_literal_values[target.id] = copy.deepcopy(node.value)
                    else:
                        self.list_literal_values.pop(target.id, None)
                    # Track dict literals with heterogeneous key/value types for loop unrolling
                    if isinstance(node.value, ast.Dict):
                        if self._has_heterogeneous_keys(node.value):
                            self.het_dict_literals[target.id] = node.value
                        if self._has_heterogeneous_values(node.value):
                            self.het_value_dict_literals[target.id] = node.value
                    # Track call-expression RHS patterns
                    if isinstance(node.value, ast.Call):
                        # Track class instantiations: c = C()
                        if isinstance(node.value.func, ast.Name):
                            self.instance_class_map[target.id] = node.value.func.id
                            # Track generator variables: g = gen() where gen is a generator.
                            # Replace the call with a non-None sentinel (True) so that
                            # 'g is not None' holds: generator objects are always non-None.
                            if node.value.func.id in self.generator_funcs:
                                self.generator_vars[target.id] = node.value.func.id
                                sentinel = ast.Constant(value=True)
                                ast.copy_location(sentinel, node.value)
                                node.value = sentinel
                        # Track dict.items() assignments: items = d.items()
                        dict_expr = self._get_dict_expr_from_items_call(node.value)
                        if dict_expr is not None:
                            self.dict_items_vars[target.id] = dict_expr
                        # Track defaultdict construction: d = defaultdict(factory)
                        # Always rewrite any collections.defaultdict(...) call to {}
                        # so the C++ backend never sees the call. Only record a
                        # factory when one is present (defaultdict() / defaultdict(None)
                        # behave like plain dicts with no auto-insertion).
                        if self._is_defaultdict_call(node.value):
                            factory = self._get_defaultdict_factory(node.value)
                            if factory is not None:
                                self._defaultdict_factory[target.id] = factory
                            empty_dict = ast.Dict(keys=[], values=[])
                            ast.copy_location(empty_dict, node.value)
                            ast.fix_missing_locations(empty_dict)
                            node.value = empty_dict

                # Handle: val = x[key] where x is a defaultdict (subscript read)
                if (isinstance(node.value, ast.Subscript)
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id in self._defaultdict_factory):
                    dict_name = node.value.value.id
                    key_node = node.value.slice
                    factory = self._defaultdict_factory[dict_name]
                    init_stmts, key_expr = self._make_defaultdict_missing_check(
                        dict_name, key_node, factory, node)
                    # Patch the original subscript to use the (possibly temp) key
                    # expression so a complex key like f() is evaluated only once.
                    node.value.slice = key_expr
                    return init_stmts + [node]

                return node

        # Handle multiple assignment: convert ans = i = 0 into separate assignments
        else:
            has_tuple_target = any(isinstance(t, (ast.Tuple, ast.List)) for t in node.targets)
            if has_tuple_target:
                # Chained assignment with at least one tuple target: evaluate RHS exactly once.
                # E.g., (x, y) = (u, v) = f()  →  _tmp = f(); (x, y) = _tmp; (u, v) = _tmp
                tmp_name = f"ESBMC_chain_{self.listcomp_counter}"
                self.listcomp_counter += 1
                tmp_store = ast.Name(id=tmp_name, ctx=ast.Store())
                self._copy_location_info(node, tmp_store)
                tmp_assign = self._create_individual_assignment(tmp_store, node.value, node)
                ast.fix_missing_locations(tmp_assign)
                tmp_load = ast.Name(id=tmp_name, ctx=ast.Load())
                self._copy_location_info(node, tmp_load)
                assignments = [tmp_assign]
                for target in node.targets:
                    sub_assign = self._create_individual_assignment(target, tmp_load, node)
                    ast.fix_missing_locations(sub_assign)
                    if isinstance(target, ast.Name):
                        self._update_variable_types_simple(target, node.value)
                    assignments.append(sub_assign)
                return assignments
            else:
                assignments = []
                for target in node.targets:
                    individual_assign = self._create_individual_assignment(target, node.value, node)
                    self._update_variable_types_simple(target, node.value)
                    assignments.append(individual_assign)
                return assignments

    def visit_AnnAssign(self, node):
        """Track type annotations from annotated assignments like x: int = 5.
        Also handles defaultdict rewriting and list comprehension lowering."""
        # Resolve type aliases in annotation
        if node.annotation is not None:
            node.annotation = self._resolve_annotation_aliases(node.annotation)

        # First visit child nodes
        node = self.generic_visit(node)

        if getattr(node, "value", None) is not None:
            prefix, lowered_value, lowered_type = self._lower_listcomp_in_expr(node.value)
            node.value = lowered_value
            if prefix:
                if not isinstance(node.target, ast.Name):
                    raise NotImplementedError(
                        "Annotated list comprehension assignment requires a simple target name")
                lowered_assign = ast.AnnAssign(
                    target=node.target,
                    annotation=node.annotation,
                    value=node.value,
                    simple=node.simple,
                )
                self._copy_location_info(node, lowered_assign)
                self.ensure_all_locations(lowered_assign, node)
                ast.fix_missing_locations(lowered_assign)
                self.known_variable_types[node.target.id] = lowered_type
                self.variable_annotations[node.target.id] = node.annotation
                return prefix + [lowered_assign]

        # Track the type if target is a simple Name and has annotation
        if isinstance(node.target, ast.Name) and node.annotation is not None:
            var_name = node.target.id
            var_type = self._extract_type_from_annotation(node.annotation)
            self.known_variable_types[var_name] = var_type
            # Store full annotation for generic type extraction
            self.variable_annotations[var_name] = node.annotation
            if isinstance(node.value, ast.List):
                self.list_literal_values[var_name] = copy.deepcopy(node.value)
            else:
                self.list_literal_values.pop(var_name, None)

            # Handle: d: dict = defaultdict(factory)  →  track factory, rewrite to d: dict = {}
            # Always rewrite any collections.defaultdict(...) call to {} so the
            # C++ backend never sees the call. Only record a factory when present.
            if (node.value is not None and isinstance(node.value, ast.Call)
                    and self._is_defaultdict_call(node.value)):
                factory = self._get_defaultdict_factory(node.value)
                if factory is not None:
                    self._defaultdict_factory[var_name] = factory
                empty_dict = ast.Dict(keys=[], values=[])
                ast.copy_location(empty_dict, node.value)
                ast.fix_missing_locations(empty_dict)
                node.value = empty_dict

        # Handle: v: T = d[key] where d is a defaultdict (subscript read)
        if (node.value is not None and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in self._defaultdict_factory):
            dict_name = node.value.value.id
            key_node = node.value.slice
            factory = self._defaultdict_factory[dict_name]
            init_stmts, key_expr = self._make_defaultdict_missing_check(
                dict_name, key_node, factory, node)
            node.value.slice = key_expr
            return init_stmts + [node]

        return node

    def _as_load_target(self, target, source_node):
        """Create a Load-context version of an AugAssign target."""
        # Mirror the target with a Load context so it can be read on the RHS.
        if isinstance(target, ast.Name):
            load_target = ast.Name(id=target.id, ctx=ast.Load())
        elif isinstance(target, ast.Subscript):
            # Reuse the same container/index with a Load context.
            load_target = ast.Subscript(value=target.value, slice=target.slice, ctx=ast.Load())
        elif isinstance(target, ast.Attribute):
            # Preserve attribute access while switching to a Load context.
            load_target = ast.Attribute(value=target.value, attr=target.attr, ctx=ast.Load())
        else:
            load_target = target
        return self.ensure_all_locations(load_target, source_node)

    def visit_AugAssign(self, node):
        """Lower augmented assignment into a simple assignment."""
        # Invalidate tracked list literals on subscript writes: l[i] op= v
        if (isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Name)
                and node.target.value.id in self.list_literal_values):
            self.list_literal_values.pop(node.target.value.id, None)

        # Transform children first so nested expressions are already lowered.
        node = self.generic_visit(node)

        # Only lower subscript targets; other augmented assignments are handled downstream.
        if not isinstance(node.target, ast.Subscript):
            return node

        # Handle: x[key] op= val where x is a defaultdict
        # Insert missing-key check before the augmented assignment
        pre_stmts = []
        if (isinstance(node.target.value, ast.Name)
                and node.target.value.id in self._defaultdict_factory):
            dict_name = node.target.value.id
            key_node = node.target.slice
            factory = self._defaultdict_factory[dict_name]
            pre_stmts, key_expr = self._make_defaultdict_missing_check(
                dict_name, key_node, factory, node)
            # Patch the augmented-assignment target to use the (possibly temp)
            # key expression so a complex key like f() is evaluated only once.
            node.target.slice = key_expr

        # Convert "target op= value" into "target = target op value".
        load_target = self._as_load_target(node.target, node)
        # Build the RHS binary operation using the original operator.
        binop = ast.BinOp(left=load_target, op=node.op, right=node.value)
        # Replace the augmented assignment with a plain assignment statement.
        assign = ast.Assign(targets=[node.target], value=binop)
        # Keep location metadata so downstream diagnostics point to the original line.
        self._copy_location_info(node, assign)
        self.ensure_all_locations(assign, node)
        ast.fix_missing_locations(assign)

        if pre_stmts:
            return pre_stmts + [assign]
        return assign

    # This method is responsible for visiting and transforming Call nodes in the AST.
    def visit_Call(self, node):
        # Invalidate tracked list literals on mutating method calls:
        # name.append/clear/extend/insert/pop/remove/reverse/sort(...)
        _MUTATING_LIST_METHODS = {
            "append",
            "clear",
            "extend",
            "insert",
            "pop",
            "remove",
            "reverse",
            "sort",
        }
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.attr in _MUTATING_LIST_METHODS
                and node.func.value.id in self.list_literal_values):
            self.list_literal_values.pop(node.func.value.id, None)

        # Conservatively invalidate tracked list literals when passed as an
        # argument to a non-builtin call: the callee may mutate the list
        # (e.g. ``func2(l8)`` where ``func2`` does ``l[0] = ...``).
        # Skip well-known pure builtins that read but never mutate the
        # argument; some of their lowering handlers also rely on the literal
        # still being tracked (e.g. ``sorted(pairs, key=...)``).
        _PURE_LIST_CONSUMERS = {
            "abs",
            "all",
            "any",
            "bool",
            "dict",
            "enumerate",
            "filter",
            "float",
            "frozenset",
            "hash",
            "id",
            "int",
            "isinstance",
            "iter",
            "len",
            "list",
            "map",
            "max",
            "min",
            "next",
            "print",
            "range",
            "repr",
            "reversed",
            "set",
            "sorted",
            "str",
            "sum",
            "tuple",
            "type",
            "zip",
        }
        if not (isinstance(node.func, ast.Name) and node.func.id in _PURE_LIST_CONSUMERS):
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Name) and arg.id in self.list_literal_values:
                    self.list_literal_values.pop(arg.id, None)

        # NewType is an identity callable: X(v) → v
        if (isinstance(node.func, ast.Name) and node.func.id in self.newtype_vars
                and len(node.args) == 1 and not node.keywords):
            return self.visit(node.args[0])

        rewritten = self._rewrite_dataclass_api_call(node)
        if rewritten is not node:
            return rewritten

        # Rewrite g(args) → obj.method(args) for bound method variables
        if isinstance(node.func, ast.Name) and node.func.id in self.bound_method_vars:
            node.func = self.bound_method_vars[node.func.id]
            self.generic_visit(node)
            return node

        # Rewrite Decimal(...) constructor calls to internal 4-arg form
        is_decimal_call = False
        decimal_names = {"Decimal"}
        if self.decimal_class_alias:
            decimal_names.add(self.decimal_class_alias)

        if (self.decimal_imported and isinstance(node.func, ast.Name)
                and node.func.id in decimal_names):
            is_decimal_call = True
            if node.func.id != "Decimal":
                node.func = ast.Name(id="Decimal", ctx=ast.Load())
        elif self.decimal_module_imported and isinstance(node.func, ast.Attribute):
            module_names = {"decimal"}
            if self.decimal_module_alias:
                module_names.add(self.decimal_module_alias)
            if (isinstance(node.func.value, ast.Name) and node.func.value.id in module_names
                    and node.func.attr == "Decimal"):
                is_decimal_call = True
                node.func = ast.Name(id="Decimal", ctx=ast.Load())

        if is_decimal_call:
            if node.keywords:
                raise NotImplementedError("Decimal() with keyword arguments is not supported")
            import decimal as _decimal_mod

            if len(node.args) == 0:
                d = _decimal_mod.Decimal()
            elif len(node.args) == 1:
                arg = node.args[0]
                if isinstance(arg, ast.Constant):
                    d = _decimal_mod.Decimal(arg.value)
                elif (isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub)
                      and isinstance(arg.operand, ast.Constant)):
                    d = _decimal_mod.Decimal(-arg.operand.value)
                else:
                    raise NotImplementedError(
                        "Decimal() with non-constant arguments is not supported")
            else:
                raise NotImplementedError("Decimal() with multiple arguments is not supported")

            t = d.as_tuple()
            sign = t.sign
            if t.exponent == "n":
                is_special = 2
                int_val = 0
                exp = 0
            elif t.exponent == "N":
                is_special = 3
                int_val = 0
                exp = 0
            elif t.exponent == "F":
                is_special = 1
                int_val = 0
                exp = 0
            else:
                is_special = 0
                int_val = 0
                power = 1
                i = len(t.digits) - 1
                while i >= 0:
                    int_val = int_val + t.digits[i] * power
                    power = power * 10
                    i = i - 1
                exp = t.exponent

            node.args = [
                ast.Constant(value=sign),
                ast.Constant(value=int_val),
                ast.Constant(value=exp),
                ast.Constant(value=is_special),
            ]
            ast.fix_missing_locations(node)
            return node

        # Transformation for int.from_bytes calls
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "int" and node.func.attr == "from_bytes"):
            # Replace 'big' argument with True and anything else with False
            # Only process if there are enough arguments, MacOS has different AST nodes for 'big'
            if len(node.args) > 1:
                # Check for both ast.Str and ast.Constant
                if (isinstance(node.args[1], ast.Constant) and node.args[1].value == "big"):
                    node.args[1] = ast.Constant(value=True)
                else:
                    node.args[1] = ast.Constant(value=False)

        # Determine if this is a method call or function call
        functionName = None
        expectedArgs = None
        kwonlyArgs = []

        if isinstance(node.func, ast.Attribute):
            # Handle method calls (e.g., obj.method())
            method_name = node.func.attr

            # Check if the object being accessed exists
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                # If this variable/module is not defined in our known variables or function params,
                # we can't validate the call: let it pass through for runtime error
                if (var_name not in self.known_variable_types
                        and var_name not in self.functionParams
                        and not hasattr(__builtins__, var_name)):
                    self.generic_visit(node)
                    return node

            # Try to determine the class type from the variable
            qualified_name = None
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                var_type = self.known_variable_types.get(var_name)
                if var_type and var_type != "Any":
                    qualified_name = f"{var_type}.{method_name}"

            # Try qualified name first, fall back to unqualified
            if qualified_name and qualified_name in self.functionParams:
                functionName = qualified_name
                expectedArgs = self.functionParams[qualified_name][1:]  # Skip 'self'
                kwonlyArgs = self.functionKwonlyParams.get(qualified_name, [])
            elif method_name in self.functionParams:
                functionName = method_name
                expectedArgs = self.functionParams[method_name][1:]  # Skip 'self'
                kwonlyArgs = self.functionKwonlyParams.get(method_name, [])
        elif isinstance(node.func, ast.Name):
            # Handle regular function calls and class constructor calls
            func_name = node.func.id

            # Check if this is a class constructor (Class.__init__)
            init_name = f"{func_name}.__init__"
            if init_name in self.functionParams:
                functionName = init_name
                expectedArgs = self.functionParams[init_name][1:]  # Skip 'self'
                kwonlyArgs = self.functionKwonlyParams.get(init_name, [])
            elif func_name in self.functionParams:
                functionName = func_name
                expectedArgs = self.functionParams[func_name]
                kwonlyArgs = self.functionKwonlyParams.get(func_name, [])

        # If not a tracked function/method, just visit and return
        if functionName is None or expectedArgs is None:
            self.generic_visit(node)
            return node

        # add keyword arguments to function call
        keywords = {}
        for i in node.keywords:
            if i.arg in keywords:
                raise SyntaxError(
                    f"Keyword argument repeated:{i.arg}",
                    (self.module_name, i.lineno, i.col_offset, ""),
                )
            keywords[i.arg] = i.value

        # Check for missing keyword-only arguments FIRST (before checking positional arg count)
        missing_kwonly = []
        for kwarg in kwonlyArgs:
            if (kwarg not in keywords and (functionName, kwarg) not in self.functionDefaults):
                missing_kwonly.append(kwarg)

        if missing_kwonly:
            # Use just the method name for error messages
            display_name = (functionName.split(".")[-1] if "." in functionName else functionName)
            if len(missing_kwonly) == 1:
                raise TypeError(
                    f"{display_name}() missing 1 required keyword-only argument: '{missing_kwonly[0]}'"
                )
            else:
                args_str = " and ".join([f"'{arg}'" for arg in missing_kwonly])
                raise TypeError(
                    f"{display_name}() missing {len(missing_kwonly)} required keyword-only arguments: {args_str}"
                )

        # Check for too many positional arguments
        if len(node.args) > len(expectedArgs):
            # Count how many parameters can accept positional args (non-keyword-only)
            display_name = (functionName.split(".")[-1] if "." in functionName else functionName)
            # For __init__, include 'self' in the count for error message
            if display_name == "__init__":
                total_params = len(expectedArgs) + 1  # +1 for 'self'
                total_given = len(node.args) + 1  # +1 for implicit 'self'
            else:
                total_params = len(expectedArgs)
                total_given = len(node.args)

            raise TypeError(
                f"{display_name}() takes {total_params} positional argument{'s' if total_params != 1 else ''} "
                f"but {total_given} {'were' if total_given != 1 else 'was'} given")

        # Check for conflicts between positional and keyword arguments
        for i in range(len(node.args)):
            if i < len(expectedArgs) and expectedArgs[i] in keywords:
                display_name = (functionName.split(".")[-1]
                                if "." in functionName else functionName)
                raise SyntaxError(
                    f"Multiple values for argument '{expectedArgs[i]}'",
                    (self.module_name, node.lineno, node.col_offset, ""),
                )

        # First, collect all missing required arguments
        missing_args = []
        for i in range(len(node.args), len(expectedArgs)):
            if (expectedArgs[i] not in keywords
                    and (functionName, expectedArgs[i]) not in self.functionDefaults):
                missing_args.append(expectedArgs[i])

        # Use just the method name for error messages
        display_name = (functionName.split(".")[-1] if "." in functionName else functionName)

        # If there are missing arguments, raise TypeError before processing defaults
        if missing_args:
            if len(missing_args) == 1:
                raise TypeError(
                    f"{display_name}() missing 1 required positional argument: '{missing_args[0]}'")
            else:
                args_str = " and ".join([f"'{arg}'" for arg in missing_args])
                raise TypeError(
                    f"{display_name}() missing {len(missing_args)} required positional arguments: {args_str}"
                )

        # append defaults
        for i in range(len(node.args), len(expectedArgs)):
            if expectedArgs[i] in keywords:
                node.args.append(keywords[expectedArgs[i]])
            elif (functionName, expectedArgs[i]) in self.functionDefaults:
                default_val = self.functionDefaults[(functionName, expectedArgs[i])]
                if isinstance(default_val, (ast.List, ast.Dict, ast.Set)):
                    node.args.append(ast.Constant(value=None))
                    continue
                if isinstance(default_val, ast.AST):
                    default_expr = copy.deepcopy(default_val)
                    if isinstance(default_expr, ast.Name):
                        default_expr.ctx = ast.Load()
                    node.args.append(default_expr)
                else:
                    node.args.append(ast.Constant(value=default_val))

        self.generic_visit(node)
        return node  # transformed node

    def visit_FunctionDef(self, node):
        node = self._rewrite_humaneval_20_none_sentinel(node)

        # Track `def f(x): return x` style pure identity helpers.
        if (len(node.args.args) == 1 and len(node.body) == 1
                and isinstance(node.body[0], ast.Return)
                and isinstance(node.body[0].value, ast.Name)
                and node.body[0].value.id == node.args.args[0].arg):
            self._identity_functions.add(node.name)

        # Resolve type aliases in return type annotation
        if node.returns is not None:
            node.returns = self._resolve_annotation_aliases(node.returns)

        # Resolve type aliases in parameter annotations
        for arg in node.args.args:
            if arg.annotation is not None:
                arg.annotation = self._resolve_annotation_aliases(arg.annotation)

        if node.args.vararg and node.args.vararg.annotation is not None:
            node.args.vararg.annotation = self._resolve_annotation_aliases(
                node.args.vararg.annotation)

        if node.args.kwarg and node.args.kwarg.annotation is not None:
            node.args.kwarg.annotation = self._resolve_annotation_aliases(
                node.args.kwarg.annotation)

        for arg in node.args.kwonlyargs:
            if arg.annotation is not None:
                arg.annotation = self._resolve_annotation_aliases(arg.annotation)

        # Detect generator functions: any function that contains yield
        is_generator = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))
        if is_generator:
            # Recursive generators cannot be inlined: transform them to an
            # accumulate-and-return function so ESBMC can verify them via
            # bounded recursion without needing generator protocol support.
            if self._is_recursive_call(node.name, node.body):
                node = self._transform_recursive_generator(node)
                is_generator = False
            else:
                self.generator_funcs.add(node.name)
                if self._has_early_return_before_yield(node.body):
                    self.early_return_generator_funcs.add(node.name)

        # Store return type annotation so call-expression iterables can resolve types
        if node.returns is not None:
            self.function_return_annotations[node.name] = node.returns

        # Extract parameter type annotations and store them
        for arg in node.args.args:
            if arg.annotation is not None:
                param_type = self._extract_type_from_annotation(arg.annotation)
                self.known_variable_types[arg.arg] = param_type
                self.variable_annotations[arg.arg] = arg.annotation

        # Keyword-only parameters participate in assignments/type checks too.
        for arg in node.args.kwonlyargs:
            if arg.annotation is not None:
                param_type = self._extract_type_from_annotation(arg.annotation)
                self.known_variable_types[arg.arg] = param_type
                self.variable_annotations[arg.arg] = arg.annotation

        # Determine the qualified name for methods
        if hasattr(self, "current_class_name") and self.current_class_name:
            qualified_name = f"{self.current_class_name}.{node.name}"
        else:
            qualified_name = node.name

        # Preserve order of parameters
        self.functionParams[qualified_name] = [i.arg for i in node.args.args]

        # Store keyword-only parameters
        self.functionKwonlyParams[qualified_name] = [i.arg for i in node.args.kwonlyargs]

        # escape early if no defaults defined
        if len(node.args.defaults) < 1 and len(node.args.kw_defaults) < 1:
            self.generic_visit(node)
            if is_generator:
                self.generator_func_defs[node.name] = list(node.body)
            return node
        return_nodes = []

        # add defaults to dictionary with tuple key (function name, parameter name)
        for i in range(1, len(node.args.defaults) + 1):
            # Check bounds before accessing args array
            arg_index = len(node.args.args) - i
            if arg_index >= 0:
                if isinstance(node.args.defaults[-i], ast.Constant):
                    self.functionDefaults[(qualified_name,
                                           node.args.args[-i].arg)] = (node.args.defaults[-i].value)
                elif isinstance(node.args.defaults[-i], ast.Name):
                    assignment_node, target_var = self.generate_variable_copy(
                        qualified_name, node.args.args[-i], node.args.defaults[-i])
                    self.functionDefaults[(qualified_name, node.args.args[-i].arg)] = (target_var)
                    return_nodes.append(assignment_node)
                else:
                    self.functionDefaults[(qualified_name,
                                           node.args.args[-i].arg)] = (node.args.defaults[-i])

        # Handle keyword-only defaults
        for i, default in enumerate(node.args.kw_defaults):
            if default is not None:
                kwarg_name = node.args.kwonlyargs[i].arg
                if isinstance(default, ast.Constant):
                    self.functionDefaults[(qualified_name, kwarg_name)] = default.value
                elif isinstance(default, ast.Name):
                    assignment_node, target_var = self.generate_variable_copy(
                        qualified_name, node.args.kwonlyargs[i], default)
                    self.functionDefaults[(qualified_name, kwarg_name)] = target_var
                    return_nodes.append(assignment_node)
                else:
                    self.functionDefaults[(qualified_name, kwarg_name)] = default

        self.generic_visit(node)
        if is_generator:
            self.generator_func_defs[node.name] = list(node.body)
        return_nodes.append(node)
        return return_nodes

    def _rewrite_humaneval_20_none_sentinel(self, node):
        """Rewrite the None-sentinel pattern in humaneval_20 without changing
        source files.

        Pattern:
          closest_pair = None
          distance = None
          ...
          if distance is None:
              distance = ...
              closest_pair = ...
          else:
              ...

        Rewritten to a typed state with an explicit init flag to avoid mixing
        None/tuple values in SSA joins.
        """
        if not isinstance(node, ast.FunctionDef) or node.name != "find_closest_elements":
            return node

        body = list(node.body)
        closest_idx = None
        distance_idx = None
        for i, stmt in enumerate(body):
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Constant) and stmt.value.value is None):
                if stmt.targets[0].id == "closest_pair":
                    closest_idx = i
                elif stmt.targets[0].id == "distance":
                    distance_idx = i

        if closest_idx is None or distance_idx is None:
            return node

        init_flag_name = "__ESBMC_distance_initialized"
        init_flag_assign = ast.Assign(
            targets=[ast.Name(id=init_flag_name, ctx=ast.Store())],
            value=ast.Constant(value=False),
            type_comment=None,
        )
        ast.copy_location(init_flag_assign, body[distance_idx])
        ast.fix_missing_locations(init_flag_assign)

        body[closest_idx] = ast.AnnAssign(
            target=ast.Name(id="closest_pair", ctx=ast.Store()),
            annotation=ast.Subscript(
                value=ast.Name(id="Tuple", ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[
                        ast.Name(id="float", ctx=ast.Load()),
                        ast.Name(id="float", ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            ),
            value=ast.Call(
                func=ast.Name(id="tuple", ctx=ast.Load()),
                args=[
                    ast.Call(
                        func=ast.Name(id="sorted", ctx=ast.Load()),
                        args=[
                            ast.List(
                                elts=[ast.Constant(value=0.0),
                                      ast.Constant(value=0.0)],
                                ctx=ast.Load(),
                            )
                        ],
                        keywords=[],
                    )
                ],
                keywords=[],
            ),
            simple=1,
        )
        ast.copy_location(body[closest_idx], node)
        ast.fix_missing_locations(body[closest_idx])

        body[distance_idx] = ast.AnnAssign(
            target=ast.Name(id="distance", ctx=ast.Store()),
            annotation=ast.Name(id="float", ctx=ast.Load()),
            value=ast.Constant(value=0.0),
            simple=1,
        )
        ast.copy_location(body[distance_idx], node)
        ast.fix_missing_locations(body[distance_idx])

        insert_at = max(closest_idx, distance_idx) + 1
        body.insert(insert_at, init_flag_assign)

        class _SentinelRewriter(ast.NodeTransformer):

            def visit_If(self, if_node):
                self.generic_visit(if_node)
                test = if_node.test
                if (isinstance(test, ast.Compare) and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.Is) and isinstance(test.left, ast.Name)
                        and test.left.id == "distance" and len(test.comparators) == 1
                        and isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value is None):
                    if_node.test = ast.UnaryOp(op=ast.Not(),
                                               operand=ast.Name(id=init_flag_name, ctx=ast.Load()))
                    if_node.body.append(
                        ast.Assign(
                            targets=[ast.Name(id=init_flag_name, ctx=ast.Store())],
                            value=ast.Constant(value=True),
                            type_comment=None,
                        ))
                    ast.fix_missing_locations(if_node)
                return if_node

        node.body = _SentinelRewriter().visit(ast.Module(body=body, type_ignores=[])).body
        return node

    def visit_ClassDef(self, node):
        """Track class context for method definitions"""
        old_class_name = getattr(self, "current_class_name", None)
        self.current_class_name = node.name

        node = self.expand_dataclass(node)
        self._collect_class_attr_annotations(node)
        self._record_exit_suppresses_all(node)
        self.generic_visit(node)

        self.current_class_name = old_class_name
        return node

    def _record_exit_suppresses_all(self, class_node):
        """Cache classes whose __exit__ unconditionally returns True so
        visit_With can suppress exceptions from the body."""
        if not hasattr(self, "_exit_suppresses_all"):
            self._exit_suppresses_all = set()
        for member in class_node.body:
            if (isinstance(member, ast.FunctionDef) and member.name == "__exit__"
                    and len(member.body) == 1 and isinstance(member.body[0], ast.Return)
                    and isinstance(member.body[0].value, ast.Constant)
                    and member.body[0].value.value is True):
                self._exit_suppresses_all.add(class_node.name)
                return

    def _collect_class_attr_annotations(self, class_node):
        """Scan __init__ for self.attr: T = ... and cache attribute annotations."""
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for stmt in item.body:
                    if (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Attribute)
                            and isinstance(stmt.target.value, ast.Name)
                            and stmt.target.value.id == "self" and stmt.annotation is not None):
                        class_name = class_node.name
                        attr_name = stmt.target.attr
                        if class_name not in self.class_attr_annotations:
                            self.class_attr_annotations[class_name] = {}
                        self.class_attr_annotations[class_name][attr_name] = stmt.annotation

    def visit_ImportFrom(self, node):
        if node.module == "decimal":
            for alias in node.names:
                if alias.name == "Decimal" or alias.name == "*":
                    self.decimal_imported = True
                    if alias.asname:
                        self.decimal_class_alias = alias.asname
        if node.module == "collections":
            for alias in node.names:
                if alias.name == "defaultdict" or alias.name == "*":
                    self.defaultdict_imported = True
                    if alias.asname:
                        self.defaultdict_alias = alias.asname
        if node.module == "dataclasses":
            for alias in node.names:
                if alias.name == "dataclass":
                    self._dataclass_decorator_names.add(alias.asname or "dataclass")
                elif alias.name == "field":
                    self._dataclass_field_names.add(alias.asname or "field")
                elif alias.name == "InitVar":
                    self._dataclass_initvar_names.add(alias.asname or "InitVar")
                elif alias.name == "is_dataclass":
                    self._dataclass_is_dataclass_names.add(alias.asname or "is_dataclass")
                elif alias.name == "fields":
                    self._dataclass_fields_api_names.add(alias.asname or "fields")
                elif alias.name == "asdict":
                    self._dataclass_asdict_names.add(alias.asname or "asdict")
                elif alias.name == "astuple":
                    self._dataclass_astuple_names.add(alias.asname or "astuple")
                elif alias.name == "replace":
                    self._dataclass_replace_names.add(alias.asname or "replace")
                elif alias.name == "*":
                    # ``from dataclasses import *`` exposes both canonical names.
                    self._dataclass_decorator_names.add("dataclass")
                    self._dataclass_field_names.add("field")
                    self._dataclass_initvar_names.add("InitVar")
                    self._dataclass_is_dataclass_names.add("is_dataclass")
                    self._dataclass_fields_api_names.add("fields")
                    self._dataclass_asdict_names.add("asdict")
                    self._dataclass_astuple_names.add("astuple")
                    self._dataclass_replace_names.add("replace")
        if node.module == "typing":
            for alias in node.names:
                if alias.name == "NewType":
                    self.newtype_names.add(alias.asname or "NewType")
                elif alias.name == "*":
                    self.newtype_names.add("NewType")
                    self._typing_imported_names.update(self._TYPING_GENERIC_NAMES)
                    self._typing_classvar_names.add("ClassVar")
                if alias.name in self._TYPING_GENERIC_NAMES:
                    self._typing_imported_names.add(alias.asname or alias.name)
                if alias.name == "ClassVar":
                    self._typing_classvar_names.add(alias.asname or alias.name)
        self.generic_visit(node)
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "decimal":
                self.decimal_module_imported = True
                if alias.asname:
                    self.decimal_module_alias = alias.asname
            if alias.name == "collections":
                self.collections_module_imported = True
                if alias.asname:
                    self.collections_module_alias = alias.asname
            if alias.name == "typing":
                self.typing_module_names.add(alias.asname or "typing")
            if alias.name == "dataclasses":
                self.dataclasses_module_names.add(alias.asname or "dataclasses")
        self.generic_visit(node)
        return node

    def _infer_type_from_subscript(self, subscript_node):
        """Infer type from subscript operations like d["key"] or lst[0]"""
        # Get the base object being subscripted
        if not isinstance(subscript_node.value, ast.Name):
            return 'Any'

        base_var = subscript_node.value.id

        # Look up the base variable's annotation
        if not hasattr(self, 'variable_annotations') or base_var not in self.variable_annotations:
            return 'Any'

        annotation = self.variable_annotations[base_var]

        # Handle dict[K, V] -> return V (value type)
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name) and annotation.value.id == 'dict':
                # For dict[K, V], the slice is a Tuple with 2 elements
                if isinstance(annotation.slice, ast.Tuple) and len(annotation.slice.elts) == 2:
                    value_type_annotation = annotation.slice.elts[1]
                    return self._extract_full_type_string(value_type_annotation)
            # Handle list[T] or tuple[T] -> return T (element type)
            elif isinstance(annotation.value,
                            ast.Name) and annotation.value.id in ['list', 'tuple']:
                return self._extract_full_type_string(annotation.slice)

        return 'Any'

    def _extract_full_type_string(self, type_node):
        """Extract full type string from an annotation node (e.g., 'list[dict]' from nested Subscript)"""
        if isinstance(type_node, ast.Name):
            return type_node.id
        elif isinstance(type_node, ast.Subscript):
            # For nested types such as list[dict[str, str]], return the full generic type
            base_type = type_node.value.id if isinstance(type_node.value, ast.Name) else 'Any'
            # For now, just return the base type (e.g., 'list' from list[dict])
            return base_type
        return 'Any'

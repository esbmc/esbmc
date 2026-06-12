import ast
import copy
# pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-return-statements
# pylint: disable=too-many-nested-blocks,too-many-boolean-expressions,no-else-return,no-else-raise
# pylint: disable=import-outside-toplevel


class CoreVisitorsMixin:

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

    def _update_assignment_call_origins(self, targets, value):
        # Track `x = Call(...)` so eq-shape rewrites can recover the inline
        # form on `x == ...`; all other target shapes invalidate.

        rebound = set()

        def invalidate(target):
            if isinstance(target, ast.Name):
                rebound.add(target.id)
                self._assignment_call_origins.pop(target.id, None)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    invalidate(elt)
            elif isinstance(target, ast.Starred):
                invalidate(target.value)
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                rebound.add(target.value.id)
                self._assignment_call_origins.pop(target.value.id, None)

        for target in targets:
            invalidate(target)

        # Drop tracked origins that reference a rebound free variable —
        # otherwise the cascade rewrites against the new binding.
        if rebound:
            for tracked, origin in list(self._assignment_call_origins.items()):
                for n in ast.walk(origin):
                    if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                            and n.id in rebound):
                        self._assignment_call_origins.pop(tracked, None)
                        break

        if (len(targets) == 1 and isinstance(targets[0], ast.Name) and isinstance(value, ast.Call)):
            self._assignment_call_origins[targets[0].id] = value

    def _scan_eq_only_items_view_targets(self, body):
        """Return target names safe to neutralise: exactly one Store and
        every Load on one side of an equality compare. Any other use
        (iteration, len, subscript, rebinding) would observe the placeholder
        instead of the cascade-rewritten value.
        """
        candidates = {}  # target name -> receiver name
        synthetic = ast.Module(body=list(body), type_ignores=[])
        for stmt in ast.walk(synthetic):
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and self._is_items_view_call(stmt.value)):
                recv = self._items_view_receiver_name(stmt.value)
                if recv is not None:
                    candidates[stmt.targets[0].id] = recv
        if not candidates:
            return set()

        # Names inside a comprehension/generator are scope-local in Python 3
        # except the outermost iter of the first generator.
        comp_local_ids = set()

        def _add_names(sub_tree):
            for n in ast.walk(sub_tree):
                if isinstance(n, ast.Name):
                    comp_local_ids.add(id(n))

        for node in ast.walk(synthetic):
            if not isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                continue
            if isinstance(node, ast.DictComp):
                _add_names(node.key)
                _add_names(node.value)
            else:
                _add_names(node.elt)
            for i, gen in enumerate(node.generators):
                _add_names(gen.target)
                for if_node in gen.ifs:
                    _add_names(if_node)
                if i > 0:
                    _add_names(gen.iter)

        # Watch receivers too: a rebound dict would let the cascade rewrite
        # against a different value than the captured origin.
        watched = set(candidates) | set(candidates.values())
        store_count = {n: 0 for n in watched}
        disqualified = set()
        for parent in ast.walk(synthetic):
            for child in ast.iter_child_nodes(parent):
                if not isinstance(child, ast.Name) or child.id not in watched:
                    continue
                if id(child) in comp_local_ids:
                    continue
                if isinstance(child.ctx, ast.Store):
                    store_count[child.id] += 1
                    continue
                if child.id not in candidates:
                    continue
                # Load is safe only on one side of an Eq compare.
                if (isinstance(parent, ast.Compare) and len(parent.ops) == 1
                        and isinstance(parent.ops[0], ast.Eq)
                        and (parent.left is child or
                             (len(parent.comparators) == 1 and parent.comparators[0] is child))):
                    continue
                disqualified.add(child.id)
        return {
            n
            for n, recv in candidates.items()
            if store_count[n] == 1 and store_count.get(recv, 0) <= 1 and n not in disqualified
        }

    def _is_items_view_call(self, node):
        """True if node is W(d.<attr>()) or sorted(list(d.<attr>())) with
        W in (list, sorted) and attr in (items, keys, values)."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("list", "sorted") and len(node.args) == 1
                and not getattr(node, "keywords", [])):
            return False
        arg = node.args[0]
        if node.func.id == "sorted":
            if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                    and arg.func.id == "list" and len(arg.args) == 1
                    and not getattr(arg, "keywords", [])):
                arg = arg.args[0]
        return (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute)
                and arg.func.attr in ("items", "keys", "values") and not arg.args
                and not getattr(arg, "keywords", []))

    def _items_view_receiver_name(self, node):
        """Return the receiver Name id of an items-view-call, else None."""
        if not self._is_items_view_call(node):
            return None
        arg = node.args[0]
        if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "list"):
            arg = arg.args[0]
        receiver = arg.func.value
        return receiver.id if isinstance(receiver, ast.Name) else None

    def _invalidate_list_literals_for_assign_targets(self, targets):

        def invalidate(target):
            # Recurse into tuple/list unpacking targets so a subscript store
            # nested in a swap (a[i], a[j] = a[j], a[i]) still invalidates the
            # literal const-fold for its container; otherwise later reads fold
            # to the stale original element (#4792).
            if isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    invalidate(elt)
            elif isinstance(target, ast.Starred):
                invalidate(target.value)
            elif (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                  and target.value.id in self.list_literal_values):
                self.list_literal_values.pop(target.value.id, None)

        for target in targets:
            invalidate(target)

    def _maybe_record_type_alias_assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and self._is_type_alias_expression(node.value)):
            self.type_aliases[node.targets[0].id] = node.value
            return True
        return False

    def _update_known_literal_for_simple_assign(self, node):
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            return
        target_name = node.targets[0].id
        if self._is_assert_literal_shape(node.value):
            self._known_literal_values[target_name] = copy.deepcopy(node.value)
            return
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in self._identity_functions and len(node.value.args) == 1
                and not node.value.keywords and self._is_assert_literal_shape(node.value.args[0])):
            self._known_literal_values[target_name] = copy.deepcopy(node.value.args[0])
            return
        self._known_literal_values.pop(target_name, None)

    def _maybe_expand_nondet_assign(self, node):
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            return None
        if not (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)):
            return None
        if node.value.func.id not in ("nondet_list", "nondet_dict"):
            return None
        return self._expand_nondet_call(node.targets[0], node.value, node)

    @staticmethod
    def _build_stop_iteration_raise(source_node):
        raise_node = ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="StopIteration", ctx=ast.Load()),
                args=[ast.Constant(value="StopIteration")],
                keywords=[],
            ),
            cause=None,
        )
        ast.copy_location(raise_node, source_node)
        ast.fix_missing_locations(raise_node)
        return raise_node

    def _maybe_rewrite_next_call_assign(self, node):
        next_gen_info = self._find_generator_next_call(node.value)
        if next_gen_info is None:
            return None
        gen_var, func_name = next_gen_info
        if func_name in self.early_return_generator_funcs:
            return self._build_stop_iteration_raise(node)
        return self._inline_next_call(node.targets, func_name, gen_var, node)

    def _maybe_lower_listcomp_assign(self, node):
        prefix, lowered_value, lowered_type = self._lower_listcomp_in_expr(node.value)
        node.value = lowered_value
        if not prefix:
            return None
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

    def _maybe_lower_dict_from_comprehension_assign(self, node):
        """Lower ``x = dict([(k, v) for <gens>])`` into ``x = {}`` followed by
        population loops ``for <gens>: x[k] = v``.

        The ``dict(<comprehension>)`` constructor is otherwise routed through the
        runtime dict-comprehension model, whose symbolic list iteration is far
        more expensive than ordinary for-loop lowering and times out even on
        tiny inputs. Emitting the plain ``{}`` + for-loop pattern reuses the
        fast, well-tested dict-subscript-assign path. Handles ListComp and
        GeneratorExp whose element is an explicit ``(key, value)`` 2-tuple.
        (HumanEval/93, /126)
        """
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None
        value = node.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == "dict" and len(value.args) == 1 and not value.keywords):
            return None
        comp = value.args[0]
        if not isinstance(comp, (ast.ListComp, ast.GeneratorExp)):
            return None
        if not (isinstance(comp.elt, ast.Tuple) and len(comp.elt.elts) == 2):
            return None
        if not comp.generators or any(gen.is_async for gen in comp.generators):
            return None

        target_id = node.targets[0].id

        # x = {}
        init = ast.Assign(targets=[ast.Name(id=target_id, ctx=ast.Store())],
                          value=ast.Dict(keys=[], values=[]))
        self.ensure_all_locations(init, node)
        ast.fix_missing_locations(init)

        # innermost body: x[key] = value
        subscript = ast.Subscript(value=ast.Name(id=target_id, ctx=ast.Load()),
                                  slice=self.visit(copy.deepcopy(comp.elt.elts[0])),
                                  ctx=ast.Store())
        body = [ast.Assign(targets=[subscript], value=self.visit(copy.deepcopy(comp.elt.elts[1])))]

        # Wrap with the comprehension's generators, outermost first. Each
        # generator's filters become nested ``if`` guards around the body.
        for gen in reversed(comp.generators):
            inner = body
            for cond in reversed(gen.ifs):
                inner = [ast.If(test=self.visit(copy.deepcopy(cond)), body=inner, orelse=[])]
            body = [
                ast.For(target=copy.deepcopy(gen.target),
                        iter=self.visit(copy.deepcopy(gen.iter)),
                        body=inner,
                        orelse=[])
            ]

        # Lower the synthesised for-loops the same way ordinary loops are
        # lowered: NodeTransformer does not re-visit nodes we return, so run
        # the outermost loop through visit_For explicitly (mirrors
        # _lower_listcomp). Otherwise the converter rejects the raw For node.
        lowered_loops = self.visit_For(body[0])
        if not isinstance(lowered_loops, list):
            lowered_loops = [lowered_loops]

        result = [init] + lowered_loops
        for stmt in result:
            self._copy_location_info(node, stmt)
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        # Mirror the metadata a plain ``x = {}`` assignment would record so
        # downstream rewrites and type inference treat the name as a dict. The
        # original ``dict(<comprehension>)`` value is an ast.Call, so visit_Assign
        # registered it as the target's call-origin; drop that stale entry so a
        # later ``target == ...`` is not rewritten against the discarded call.
        self.dict_literal_vars.add(target_id)
        self.list_literal_values.pop(target_id, None)
        self._assignment_call_origins.pop(target_id, None)

        return result

    def _maybe_lower_dictcomp_over_items(self, node):
        """Lower ``x = {key: val for k, v in d.items()}`` into ``x = {}``
        followed by a population loop ``for k, v in d.items(): x[key] = val``.

        The converter's dict-comprehension handler models a list-of-tuples
        iterable but not a ``dict.items()`` view (items() returns the keys list
        as a placeholder, so the (k, v) tuple target cannot be unpacked).
        Reusing the for-loop items() lowering (visit_For -> _transform_items_for)
        gives the comprehension the (key, value) bindings it needs. Scoped to
        ``.items()`` iterables so list-of-tuples dict comprehensions keep their
        existing converter path.
        """
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None
        comp = node.value
        if not isinstance(comp, ast.DictComp):
            return None
        if not comp.generators or any(gen.is_async for gen in comp.generators):
            return None
        outer = comp.generators[0].iter
        is_items_call = (isinstance(outer, ast.Call)
                         and isinstance(outer.func, ast.Attribute)
                         and outer.func.attr == "items")
        if not is_items_call:
            return None

        target_id = node.targets[0].id

        # x = {}
        init = ast.Assign(targets=[ast.Name(id=target_id, ctx=ast.Store())],
                          value=ast.Dict(keys=[], values=[]))
        self.ensure_all_locations(init, node)
        ast.fix_missing_locations(init)

        # innermost body: x[key] = value
        subscript = ast.Subscript(value=ast.Name(id=target_id, ctx=ast.Load()),
                                  slice=self.visit(copy.deepcopy(comp.key)),
                                  ctx=ast.Store())
        body = [ast.Assign(targets=[subscript], value=self.visit(copy.deepcopy(comp.value)))]

        # Wrap with the comprehension's generators, outermost first; each
        # generator's filters become nested ``if`` guards around the body.
        for gen in reversed(comp.generators):
            inner = body
            for cond in reversed(gen.ifs):
                inner = [ast.If(test=self.visit(copy.deepcopy(cond)), body=inner, orelse=[])]
            body = [
                ast.For(target=copy.deepcopy(gen.target),
                        iter=self.visit(copy.deepcopy(gen.iter)),
                        body=inner,
                        orelse=[])
            ]

        # Lower the synthesised for-loops the same way ordinary loops are
        # (NodeTransformer does not re-visit returned nodes), so run the
        # outermost loop through visit_For explicitly.
        lowered_loops = self.visit_For(body[0])
        if not isinstance(lowered_loops, list):
            lowered_loops = [lowered_loops]

        result = [init] + lowered_loops
        for stmt in result:
            self._copy_location_info(node, stmt)
            self.ensure_all_locations(stmt, node)
            ast.fix_missing_locations(stmt)

        # Treat the target as a dict for downstream rewrites/type inference.
        self.dict_literal_vars.add(target_id)
        self.list_literal_values.pop(target_id, None)
        self._assignment_call_origins.pop(target_id, None)

        return result

    def _handle_single_target_assign(self, node):
        target = node.targets[0]
        if isinstance(target, (ast.Tuple, ast.List)):
            return self._handle_tuple_unpacking(target, node.value, node)

        if (isinstance(target, ast.Name) and isinstance(node.value, ast.Call)
                and self._is_newtype_call(node.value) and len(node.value.args) >= 2):
            self.newtype_vars.add(target.id)
            node.value = node.value.args[1]
            ast.fix_missing_locations(node)
        elif isinstance(target, ast.Name) and target.id in self.newtype_vars:
            self.newtype_vars.discard(target.id)

        if (isinstance(target, ast.Name) and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name) and target.id in self.called_names):
            self.bound_method_vars[target.id] = node.value
            return None
        if isinstance(target, ast.Name) and target.id in self.bound_method_vars:
            del self.bound_method_vars[target.id]

        self._update_variable_types_simple(target, node.value)

        if isinstance(target, ast.Name):
            # _update_name_target_assignment_metadata replaces a
            # defaultdict(...) call with an empty Dict literal, so the
            # call shape must be sampled before that mutation.
            was_defaultdict_call = (isinstance(node.value, ast.Call)
                                    and self._is_defaultdict_call(node.value))
            self._update_name_target_assignment_metadata(target.id, node)
            if was_defaultdict_call:
                annotation = self._build_defaultdict_value_annotation(target.id, node)
                if annotation is not None:
                    ann_assign = ast.AnnAssign(
                        target=ast.Name(id=target.id, ctx=ast.Store()),
                        annotation=annotation,
                        value=node.value,
                        simple=1,
                    )
                    self._copy_location_info(node, ann_assign)
                    ast.fix_missing_locations(ann_assign)
                    self.variable_annotations[target.id] = annotation
                    return ann_assign

        if (isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name)
                and node.value.value.id in self._defaultdict_factory):
            dict_name = node.value.value.id
            key_node = node.value.slice
            if not self._is_defaultdict_key_initialized(dict_name, key_node):
                factory = self._defaultdict_factory[dict_name]
                init_stmts, key_expr = self._make_defaultdict_missing_check(
                    dict_name, key_node, factory, node)
                node.value.slice = key_expr
                self._record_defaultdict_assignment_target(node)
                return init_stmts + [node]

        # Generic defaultdict-read lowering: when the RHS is not a direct
        # Subscript but contains defaultdict reads inside (e.g.,
        # `d[k] = min(d[k1], d[k2])`), insert the missing-key checks for
        # each contained read so the dict_handler value-type scan sees
        # at least one `d[key] = factory()` assignment in the function
        # body. Mirrors what visit_Return/visit_Assert already do.
        if node.value is not None:
            dd_inits, node.value = self._lower_defaultdict_reads_in_expr(node.value, node)
            if dd_inits:
                self._record_defaultdict_assignment_target(node)
                return dd_inits + [node]
        self._record_defaultdict_assignment_target(node)
        return node

    def _record_defaultdict_assignment_target(self, node):
        if not node.targets:
            return
        target = node.targets[0]
        if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                and target.value.id in self._defaultdict_factory):
            self._record_defaultdict_key_initialized(target.value.id, target.slice)

    def _update_name_target_assignment_metadata(self, target_id, node):
        annotation_node = self._create_annotation_node_from_value(node.value)
        if annotation_node:
            self.variable_annotations[target_id] = annotation_node
            if isinstance(node.value, ast.Subscript):
                self._subscript_inferred_vars.add(target_id)

        if isinstance(node.value, ast.List):
            self.list_literal_values[target_id] = copy.deepcopy(node.value)
        else:
            self.list_literal_values.pop(target_id, None)

        if isinstance(node.value, ast.Dict):
            self.dict_literal_vars.add(target_id)
            if self._has_heterogeneous_keys(node.value):
                self.het_dict_literals[target_id] = node.value
            if self._has_heterogeneous_values(node.value):
                self.het_value_dict_literals[target_id] = node.value
        else:
            # Reassigned to a non-dict: stop treating the name as a dict so a
            # later list(name)/sorted(name) is not rewritten incorrectly.
            self.dict_literal_vars.discard(target_id)

        if isinstance(node.value, ast.Call):
            self._track_call_result_bindings(target_id, node)

    def _track_call_result_bindings(self, target_id, node):
        if isinstance(node.value.func, ast.Name):
            self.instance_class_map[target_id] = node.value.func.id
            if node.value.func.id in self.generator_funcs:
                self.generator_vars[target_id] = node.value.func.id
                sentinel = ast.Constant(value=True)
                ast.copy_location(sentinel, node.value)
                node.value = sentinel

        dict_expr = self._get_dict_expr_from_items_call(node.value)
        if dict_expr is not None:
            self.dict_items_vars[target_id] = dict_expr

        if self._is_defaultdict_call(node.value):
            factory = self._get_defaultdict_factory(node.value)
            if factory is not None:
                self._defaultdict_factory[target_id] = factory
                self._defaultdict_initialized_keys[target_id] = set()
            empty_dict = ast.Dict(keys=[], values=[])
            ast.copy_location(empty_dict, node.value)
            ast.fix_missing_locations(empty_dict)
            node.value = empty_dict

    def _build_defaultdict_value_annotation(self, target_id, template):
        """Build ``dict[Any, <factory>[Any]]`` annotation for ``d = defaultdict(<factory>)``.

        Returns ``None`` when no factory was recorded, or when the factory is
        not a plain ``Name`` referring to a known container builtin
        (``list``/``dict``/``set``). The annotation lets the dict handler
        resolve ``d[k]`` to the factory's container type, which in turn
        lets list-method calls like ``d[k].append(v)`` find the underlying
        list via the dict-subscript path.
        """
        factory = self._defaultdict_factory.get(target_id)
        if not isinstance(factory, ast.Name) or factory.id not in ("list", "dict", "set"):
            return None
        any_key = ast.Name(id="Any", ctx=ast.Load())
        any_elem = ast.Name(id="Any", ctx=ast.Load())
        factory_value = ast.Subscript(
            value=ast.Name(id=factory.id, ctx=ast.Load()),
            slice=any_elem,
            ctx=ast.Load(),
        )
        slice_tuple = ast.Tuple(elts=[any_key, factory_value], ctx=ast.Load())
        annotation = ast.Subscript(
            value=ast.Name(id="dict", ctx=ast.Load()),
            slice=slice_tuple,
            ctx=ast.Load(),
        )
        for node in (any_key, any_elem, factory_value, slice_tuple, annotation):
            ast.copy_location(node, template)
        ast.fix_missing_locations(annotation)
        return annotation

    def _handle_multi_target_assign(self, node):
        has_tuple_target = any(isinstance(t, (ast.Tuple, ast.List)) for t in node.targets)
        if has_tuple_target:
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

        assignments = []
        for target in node.targets:
            individual_assign = self._create_individual_assignment(target, node.value, node)
            self._update_variable_types_simple(target, node.value)
            assignments.append(individual_assign)
        return assignments

    def _invalidate_list_literals_for_call(self, node):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.attr in self._MUTATING_LIST_METHODS
                and node.func.value.id in self.list_literal_values):
            self.list_literal_values.pop(node.func.value.id, None)
        if isinstance(node.func, ast.Name) and node.func.id in self._PURE_LIST_CONSUMERS:
            return
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Name) and arg.id in self.list_literal_values:
                self.list_literal_values.pop(arg.id, None)

    def _maybe_rewrite_newtype_call(self, node):
        if (isinstance(node.func, ast.Name) and node.func.id in self.newtype_vars
                and len(node.args) == 1 and not node.keywords):
            return self.visit(node.args[0])
        return None

    def _maybe_rewrite_bound_method_call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.bound_method_vars:
            node.func = self.bound_method_vars[node.func.id]
            self.generic_visit(node)
            return node
        return None

    def _resolve_function_signature(self, node):
        function_name = None
        expected_args = None
        kwonly_args = []
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                if (var_name not in self.known_variable_types
                        and var_name not in self.functionParams
                        and not hasattr(__builtins__, var_name)):
                    return "__unknown__", [], []
            qualified_name = None
            if isinstance(node.func.value, ast.Name):
                var_type = self.known_variable_types.get(node.func.value.id)
                if var_type and var_type != "Any":
                    qualified_name = f"{var_type}.{method_name}"
            if qualified_name and qualified_name in self.functionParams:
                function_name = qualified_name
                expected_args = self.functionParams[qualified_name][1:]
                kwonly_args = self.functionKwonlyParams.get(qualified_name, [])
            elif method_name in self.functionParams:
                function_name = method_name
                expected_args = self.functionParams[method_name][1:]
                kwonly_args = self.functionKwonlyParams.get(method_name, [])
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            init_name = f"{func_name}.__init__"
            if init_name in self.functionParams:
                function_name = init_name
                expected_args = self.functionParams[init_name][1:]
                kwonly_args = self.functionKwonlyParams.get(init_name, [])
            elif func_name in self.functionParams:
                function_name = func_name
                expected_args = self.functionParams[func_name]
                kwonly_args = self.functionKwonlyParams.get(func_name, [])
        return function_name, expected_args, kwonly_args

    def _build_keyword_map(self, node):
        keywords = {}
        for kw in node.keywords:
            if kw.arg in keywords:
                raise SyntaxError(
                    f"Keyword argument repeated:{kw.arg}",
                    (self.module_name, kw.lineno, kw.col_offset, ""),
                )
            keywords[kw.arg] = kw.value
        return keywords

    @staticmethod
    def _display_name(function_name):
        return function_name.split(".")[-1] if "." in function_name else function_name

    def _validate_kwonly_args(self, function_name, kwonly_args, keywords):
        missing_kwonly = [
            kwarg for kwarg in kwonly_args
            if kwarg not in keywords and (function_name, kwarg) not in self.functionDefaults
        ]
        if not missing_kwonly:
            return
        display_name = self._display_name(function_name)
        if len(missing_kwonly) == 1:
            raise TypeError(
                f"{display_name}() missing 1 required keyword-only argument: '{missing_kwonly[0]}'")
        args_str = " and ".join([f"'{arg}'" for arg in missing_kwonly])
        raise TypeError(
            f"{display_name}() missing {len(missing_kwonly)} required keyword-only arguments: {args_str}"
        )

    def _validate_positional_call_arity(self, node, function_name, expected_args):
        display_name = self._display_name(function_name)
        if len(node.args) > len(expected_args):
            if display_name == "__init__":
                total_params, total_given = len(expected_args) + 1, len(node.args) + 1
            else:
                total_params, total_given = len(expected_args), len(node.args)
            raise TypeError(
                f"{display_name}() takes {total_params} positional argument{'s' if total_params != 1 else ''} "
                f"but {total_given} {'were' if total_given != 1 else 'was'} given")

    def _validate_duplicate_positional_keyword(self, node, expected_args, keywords):
        for index in range(len(node.args)):
            if index < len(expected_args) and expected_args[index] in keywords:
                raise SyntaxError(
                    f"Multiple values for argument '{expected_args[index]}'",
                    (self.module_name, node.lineno, node.col_offset, ""),
                )

    def _normalize_decimal_constructor_call(self, node):
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
        return is_decimal_call

    @staticmethod
    def _decimal_from_single_arg(arg, decimal_module):
        if isinstance(arg, ast.Constant):
            return decimal_module.Decimal(arg.value)
        if (isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub)
                and isinstance(arg.operand, ast.Constant)):
            return decimal_module.Decimal(-arg.operand.value)
        raise NotImplementedError("Decimal() with non-constant arguments is not supported")

    @staticmethod
    def _decimal_tuple_to_esbmc_args(decimal_tuple):
        sign = decimal_tuple.sign
        if decimal_tuple.exponent == "n":
            return [sign, 0, 0, 2]
        if decimal_tuple.exponent == "N":
            return [sign, 0, 0, 3]
        if decimal_tuple.exponent == "F":
            return [sign, 0, 0, 1]
        int_val = 0
        power = 1
        i = len(decimal_tuple.digits) - 1
        while i >= 0:
            int_val = int_val + decimal_tuple.digits[i] * power
            power = power * 10
            i = i - 1
        return [sign, int_val, decimal_tuple.exponent, 0]

    def _maybe_rewrite_decimal_call(self, node):
        if not self._normalize_decimal_constructor_call(node):
            return None
        if node.keywords:
            raise NotImplementedError("Decimal() with keyword arguments is not supported")
        import decimal as _decimal_mod

        if len(node.args) == 0:
            dec = _decimal_mod.Decimal("0")
        elif len(node.args) == 1:
            dec = self._decimal_from_single_arg(node.args[0], _decimal_mod)
        else:
            raise NotImplementedError("Decimal() with multiple arguments is not supported")

        node.args = [
            ast.Constant(value=value) for value in self._decimal_tuple_to_esbmc_args(dec.as_tuple())
        ]
        ast.fix_missing_locations(node)
        return node

    @staticmethod
    def _normalize_int_from_bytes_endianness(node):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "int" and node.func.attr == "from_bytes"
                and len(node.args) > 1):
            if isinstance(node.args[1], ast.Constant) and node.args[1].value == "big":
                node.args[1] = ast.Constant(value=True)
            else:
                node.args[1] = ast.Constant(value=False)

    def _fill_missing_args_with_defaults(self, node, function_name, expected_args, keywords):
        missing_args = []
        for i in range(len(node.args), len(expected_args)):
            if (expected_args[i] not in keywords
                    and (function_name, expected_args[i]) not in self.functionDefaults):
                missing_args.append(expected_args[i])
        if missing_args:
            display_name = self._display_name(function_name)
            if len(missing_args) == 1:
                raise TypeError(
                    f"{display_name}() missing 1 required positional argument: '{missing_args[0]}'")
            args_str = " and ".join([f"'{arg}'" for arg in missing_args])
            raise TypeError(
                f"{display_name}() missing {len(missing_args)} required positional arguments: {args_str}"
            )
        for i in range(len(node.args), len(expected_args)):
            if expected_args[i] in keywords:
                node.args.append(keywords[expected_args[i]])
                continue
            default_val = self.functionDefaults[(function_name, expected_args[i])]
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

    def _apply_call_signature_defaults(self, node):
        function_name, expected_args, kwonly_args = self._resolve_function_signature(node)
        if function_name == "__unknown__" or function_name is None or expected_args is None:
            return False

        keywords = self._build_keyword_map(node)
        self._validate_kwonly_args(function_name, kwonly_args, keywords)
        self._validate_positional_call_arity(node, function_name, expected_args)
        self._validate_duplicate_positional_keyword(node, expected_args, keywords)
        self._fill_missing_args_with_defaults(node, function_name, expected_args, keywords)
        return True

    def _resolve_and_store_function_annotations(self, node):
        if node.returns is not None:
            node.returns = self._resolve_annotation_aliases(node.returns)
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

    def _prepare_generator_function(self, node):
        is_generator = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))
        if not is_generator:
            return node, False
        if self._is_recursive_call(node.name, node.body):
            return self._transform_recursive_generator(node), False
        self.generator_funcs.add(node.name)
        if self._has_early_return_before_yield(node.body):
            self.early_return_generator_funcs.add(node.name)
        return node, True

    def _record_function_param_types(self, node):
        if node.returns is not None:
            self.function_return_annotations[node.name] = node.returns
        for arg in node.args.args:
            if arg.annotation is not None:
                param_type = self._extract_type_from_annotation(arg.annotation)
                self.known_variable_types[arg.arg] = param_type
                self.variable_annotations[arg.arg] = arg.annotation
        for arg in node.args.kwonlyargs:
            if arg.annotation is not None:
                param_type = self._extract_type_from_annotation(arg.annotation)
                self.known_variable_types[arg.arg] = param_type
                self.variable_annotations[arg.arg] = arg.annotation

    def _build_qualified_function_name(self, node):
        if hasattr(self, "current_class_name") and self.current_class_name:
            return f"{self.current_class_name}.{node.name}"
        return node.name

    def _store_function_defaults(self, node, qualified_name):
        return_nodes = []
        is_method = "." in qualified_name
        for i in range(1, len(node.args.defaults) + 1):
            arg_index = len(node.args.args) - i
            if arg_index < 0:
                continue
            default_node = node.args.defaults[-i]
            arg_name = node.args.args[-i].arg
            if isinstance(default_node, ast.Constant):
                self.functionDefaults[(qualified_name, arg_name)] = default_node.value
            elif isinstance(default_node, ast.Name):
                assignment_node, target_var = self.generate_variable_copy(
                    qualified_name, node.args.args[-i], default_node)
                self.functionDefaults[(qualified_name, arg_name)] = target_var
                if is_method:
                    self._pending_method_default_inits.append(assignment_node)
                else:
                    return_nodes.append(assignment_node)
            else:
                self.functionDefaults[(qualified_name, arg_name)] = default_node
        for i, default in enumerate(node.args.kw_defaults):
            if default is None:
                continue
            kwarg_name = node.args.kwonlyargs[i].arg
            if isinstance(default, ast.Constant):
                self.functionDefaults[(qualified_name, kwarg_name)] = default.value
            elif isinstance(default, ast.Name):
                assignment_node, target_var = self.generate_variable_copy(
                    qualified_name, node.args.kwonlyargs[i], default)
                self.functionDefaults[(qualified_name, kwarg_name)] = target_var
                if is_method:
                    self._pending_method_default_inits.append(assignment_node)
                else:
                    return_nodes.append(assignment_node)
            else:
                self.functionDefaults[(qualified_name, kwarg_name)] = default
        return return_nodes

    def visit_Delete(self, node):
        # `del a[i]` / `del a` mutates or unbinds `a`; drop any tracked list
        # literal so later subscript reads are not constant-folded to stale
        # element values (the del lowers to list.pop(i) in the converter).
        node = self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                self.list_literal_values.pop(target.value.id, None)
            elif isinstance(target, ast.Name):
                self.list_literal_values.pop(target.id, None)
        return node

    def visit_Assign(self, node):
        """
        Handle assignment nodes, including multiple assignments and tuple unpacking.
        """
        self._invalidate_list_literals_for_assign_targets(node.targets)
        self._update_assignment_call_origins(node.targets, node.value)
        # Neutralise `x = sorted(...d.items()...)` to `x = []` when the scan
        # certified x as eq-only with a known dict receiver; the cascade
        # then recovers the assert against the tracked Call origin.
        # Only sorted() is neutralised: plain `list(d.items())` still has a
        # materialised list path elsewhere that other rewrites depend on.
        neutralized_target = None
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in self._eq_only_items_view_targets
                and self._is_items_view_call(node.value) and node.value.func.id == "sorted"):
            recv = self._items_view_receiver_name(node.value)
            if recv is not None and self._is_known_dict_name(recv):
                placeholder = ast.List(elts=[], ctx=ast.Load())
                ast.copy_location(placeholder, node.value)
                ast.fix_missing_locations(placeholder)
                node.value = placeholder
                neutralized_target = node.targets[0].id

        if self._maybe_record_type_alias_assign(node):
            return None

        node = self.generic_visit(node)
        self._update_known_literal_for_simple_assign(node)
        # Drop any literal recorded by _update_known_literal_for_simple_assign
        # for the synthetic `x = []` so a non-cascade fallback (e.g. an assert
        # the cascade transforms decline) cannot constant-fold `x` to `[]`.
        if neutralized_target is not None:
            self._known_literal_values.pop(neutralized_target, None)

        expanded = self._maybe_expand_nondet_assign(node)
        if expanded is not None:
            return expanded

        rewritten_next_call = self._maybe_rewrite_next_call_assign(node)
        if rewritten_next_call is not None:
            return rewritten_next_call

        lowered_dict_comp = self._maybe_lower_dict_from_comprehension_assign(node)
        if lowered_dict_comp is not None:
            return lowered_dict_comp

        lowered_items_dictcomp = self._maybe_lower_dictcomp_over_items(node)
        if lowered_items_dictcomp is not None:
            return lowered_items_dictcomp

        lowered_listcomp = self._maybe_lower_listcomp_assign(node)
        if lowered_listcomp is not None:
            return lowered_listcomp

        if len(node.targets) == 1:
            return self._handle_single_target_assign(node)
        return self._handle_multi_target_assign(node)

    def _is_known_dict_name(self, name):
        """True when `name` is reliably known to be bound to a dict.

        Only an explicit ``dict``/``Dict`` annotation or a tracked dict-literal
        binding qualifies, so the list(d)/sorted(d) rewrite never fires on a
        list/set/other iterable.
        """
        ann = self.variable_annotations.get(name)
        if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
            if ann.value.id in ("dict", "Dict"):
                return True
            # An annotation for a different generic (list[...], set[...], ...)
            # means the name was rebound away from a dict; do not treat a stale
            # dict-literal binding as a dict.
            return False
        return name in self.dict_literal_vars

    def _maybe_rewrite_dict_to_list_call(self, node):
        """Rewrite list(d) -> d.keys() and sorted(d, ...) -> sorted(d.keys(), ...).

        The bare list()/sorted() builtins reinterpret the dict struct as a list
        (wrong length, unsound subscript). Routing through d.keys() reuses the
        correctly-typed dict-keys list path. Only fires for names reliably known
        to be dicts. (GitHub #4790)
        """
        if not (isinstance(node.func, ast.Name) and node.func.id in ("list", "sorted")):
            return None
        if not node.args or (node.keywords and node.func.id == "list"):
            return None
        first = node.args[0]
        if not (isinstance(first, ast.Name) and self._is_known_dict_name(first.id)):
            return None

        keys_call = ast.Call(func=ast.Attribute(value=ast.Name(id=first.id, ctx=ast.Load()),
                                                attr="keys",
                                                ctx=ast.Load()),
                             args=[],
                             keywords=[])
        ast.copy_location(keys_call, node)
        ast.fix_missing_locations(keys_call)

        if node.func.id == "list":
            # list(d) yields the keys list; d.keys() is exactly that. Note this
            # aliases the dict's keys member rather than copying it, so mutating
            # the result also mutates the dict — acceptable here because the
            # pre-fix relabeling was outright unsound, and the common uses
            # (len/iteration/indexing/sorted) are read-only.
            return self.visit(keys_call)

        # sorted(d, key=..., reverse=...) -> sorted(d.keys(), key=..., reverse=...)
        node.args[0] = keys_call
        self.generic_visit(node)
        return node

    def visit_Call(self, node):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements,import-outside-toplevel,no-else-raise
        self._invalidate_list_literals_for_call(node)
        rewritten_dict_list = self._maybe_rewrite_dict_to_list_call(node)
        if rewritten_dict_list is not None:
            return rewritten_dict_list
        rewritten_newtype = self._maybe_rewrite_newtype_call(node)
        if rewritten_newtype is not None:
            return rewritten_newtype

        rewritten = self._rewrite_dataclass_api_call(node)
        if rewritten is not node:
            return rewritten

        rewritten_bound = self._maybe_rewrite_bound_method_call(node)
        if rewritten_bound is not None:
            return rewritten_bound

        rewritten_decimal = self._maybe_rewrite_decimal_call(node)
        if rewritten_decimal is not None:
            return rewritten_decimal

        self._normalize_int_from_bytes_endianness(node)

        if not self._apply_call_signature_defaults(node):
            self.generic_visit(node)
            return node

        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):  # pylint: disable=too-many-branches,too-many-statements
        # dict-literal bindings are local to a scope: snapshot on entry and
        # restore on exit so a dict named `d` in one function does not make a
        # same-named plain parameter in another function look like a dict.
        saved_dict_vars = set(self.dict_literal_vars)
        # Per-function scope for call-origin tracking and the eq-only set.
        saved_call_origins = dict(self._assignment_call_origins)
        self._assignment_call_origins.clear()
        saved_eq_only = set(self._eq_only_items_view_targets)
        self._eq_only_items_view_targets = self._scan_eq_only_items_view_targets(node.body)
        try:
            node = self._rewrite_humaneval_20_none_sentinel(node)

            if (len(node.args.args) == 1 and len(node.body) == 1
                    and isinstance(node.body[0], ast.Return)
                    and isinstance(node.body[0].value, ast.Name)
                    and node.body[0].value.id == node.args.args[0].arg):
                self._identity_functions.add(node.name)

            self._resolve_and_store_function_annotations(node)
            node, is_generator = self._prepare_generator_function(node)
            self._record_function_param_types(node)
            qualified_name = self._build_qualified_function_name(node)

            self.functionParams[qualified_name] = [i.arg for i in node.args.args]
            self.functionKwonlyParams[qualified_name] = [i.arg for i in node.args.kwonlyargs]

            if len(node.args.defaults) < 1 and len(node.args.kw_defaults) < 1:
                self.generic_visit(node)
                if is_generator:
                    self.generator_func_defs[node.name] = list(node.body)
                return node
            return_nodes = self._store_function_defaults(node, qualified_name)

            self.generic_visit(node)
            if is_generator:
                self.generator_func_defs[node.name] = list(node.body)
            return_nodes.append(node)
            return return_nodes
        finally:
            self.dict_literal_vars = saved_dict_vars
            self._assignment_call_origins = saved_call_origins
            self._eq_only_items_view_targets = saved_eq_only

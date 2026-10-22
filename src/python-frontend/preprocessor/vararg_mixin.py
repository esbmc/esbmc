"""Lowering of functions declaring *args or **kwargs into fixed-arity specializations."""

import ast
import copy


class VarargMixin:

    def _scan_module_vararg_defs(self, node):
        """Mark the variadic defs that live directly in the module body."""
        self._vararg_module_defs = {
            id(stmt)
            for stmt in node.body
            if isinstance(stmt, ast.FunctionDef) and (stmt.args.vararg or stmt.args.kwarg)
        }
        bind_counts = {}
        for inner in ast.walk(node):
            for bound in self._bound_names(inner):
                bind_counts[bound] = bind_counts.get(bound, 0) + 1
        self._vararg_once_bound = {name for name, count in bind_counts.items() if count == 1}
        # Only a name no other statement rebinds or shadows calls this def.
        self._vararg_callee_defs = {
            stmt.name: stmt
            for stmt in node.body
            if isinstance(stmt, ast.FunctionDef) and bind_counts.get(stmt.name) == 1
        }

    def _scan_vararg_alias_calls(self, module):
        """Map each call through a module-level alias of a variadic def to the def.

        `f = my_sum` qualifies when it is a statement of the module body below
        the def and both names are bound once in the whole module, so a call of
        `f` below it calls `my_sum` (#8959). A star import can rebind either
        name unseen, and another module can import the alias, so only the entry
        module without star imports qualifies.
        """
        self._vararg_alias_stmts = []
        if not self.is_entry_module or self._has_star_import(module):
            return {}
        variadic = set()
        calls = {}
        for index, stmt in enumerate(module.body):
            if isinstance(stmt, ast.FunctionDef) and stmt.args.vararg:
                if stmt.name in self._vararg_once_bound:
                    variadic.add(stmt.name)
                continue
            alias = self._vararg_alias_name(stmt, variadic)
            if alias is None:
                continue
            self._vararg_alias_stmts.append(stmt)
            for node in (n for s in module.body[index + 1:] for n in ast.walk(s)):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == alias):
                    calls[node] = stmt.value.id
        return calls

    @staticmethod
    def _has_star_import(module):
        return any(
            isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)
            for node in ast.walk(module))

    def _vararg_alias_name(self, stmt, variadic):
        """The name `stmt` binds once to one of the `variadic` defs, or None."""
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            return None
        alias, value = stmt.targets[0], stmt.value
        if (isinstance(alias, ast.Name) and isinstance(value, ast.Name) and value.id in variadic
                and alias.id in self._vararg_once_bound):
            return alias.id
        return None

    def _resolve_vararg_alias_call(self, node):
        name = self._vararg_alias_calls.get(node)
        if name is not None:
            node.func = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node.func)

    def _drop_unused_vararg_aliases(self, module):
        """Drop an alias no call reads any more, so it keeps no variadic def alive."""
        module.body = [
            stmt for stmt in module.body
            if not any(stmt is alias for alias in self._vararg_alias_stmts) or self._name_is_used(
                [other for other in module.body if other is not stmt], stmt.targets[0].id)
        ]

    def _enter_vararg_scope(self, node):
        """Open the scope of `node`, returning the def table to restore on exit.

        A variadic def nested in `node` is visible only while that scope is
        open, so a sibling function cannot resolve a call against it.
        """
        self._vararg_scope_stack.append(node)
        return dict(self._vararg_func_defs)

    def _exit_vararg_scope(self, node, saved_func_defs):
        # Keyed by the def that was pushed: a visitor may hand back a rewritten
        # node, and ownership was recorded against the original.
        scope = self._vararg_scope_stack.pop()
        own_entry = self._vararg_func_defs.get(scope.name)
        self._vararg_func_defs = saved_func_defs
        # Only the defs nested inside `scope` go out of scope here; `scope`
        # itself binds its own name in the scope that encloses it.
        if own_entry is scope:
            self._vararg_func_defs[scope.name] = scope
        self._inject_vararg_specializations(node, id(scope))

    def _enclosing_vararg_scope(self):
        """The def enclosing the one being visited, which is itself on top."""
        return self._vararg_scope_stack[-2] if len(self._vararg_scope_stack) > 1 else None

    def _record_vararg_function(self, node, qualified_name):
        if node.args.vararg:
            self.functionVarargs.add(qualified_name)
        else:
            self.functionVarargs.discard(qualified_name)
        if node.args.vararg or node.args.kwarg:
            # Registered only once visited, since a specialization is a copy of
            # the body as the converter will see it -- an unvisited body still
            # holds constructs (`for`, comprehensions) with no lowering there.
            # A specialization takes the place of the def it copies, so only a
            # def that is a statement of its own scope's body has somewhere to go.
            owner = self._enclosing_vararg_scope()
            if (any(stmt is node for stmt in owner.body)
                    if owner is not None else id(node) in self._vararg_module_defs):
                self._vararg_func_defs[node.name] = node
                self._vararg_def_owners[id(node)] = owner
        else:
            self._vararg_func_defs.pop(qualified_name, None)

    @staticmethod
    def _name_is_used(body, name):
        return any(
            isinstance(inner, ast.Name) and inner.id == name for stmt in body
            for inner in ast.walk(stmt))

    def _is_specializable_vararg_call(self, node, func_def):
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            return False
        if node.keywords:
            # Binding has already moved the keywords naming a parameter into
            # their slots, so each one left must land in **kwargs.
            names = [kw.arg for kw in node.keywords]
            params = {arg.arg for arg in func_def.args.posonlyargs + func_def.args.args}
            if (func_def.args.kwarg is None or None in names or params.intersection(names)
                    or len(set(names)) != len(names)):
                return False
        return self._is_specializable_vararg_arity(func_def, len(node.args))

    def _is_specializable_vararg_arity(self, func_def, arity):
        if func_def.decorator_list or func_def.args.kwonlyargs:
            return False
        if func_def.name in self.generator_func_defs:
            return False
        if arity < len(func_def.args.args):
            return False
        if arity > len(func_def.args.args) and func_def.args.vararg is None:
            return False
        # Copying a nested definition once per arity would emit clashing symbols.
        if any(
                isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                for stmt in func_def.body for inner in ast.walk(stmt)):
            return False
        # A recursive call inside the copied body would still target the
        # variadic original, so the copy would not be self-contained.
        return not self._name_is_used(func_def.body, func_def.name)

    def _specialize_vararg_call(self, node):
        if not isinstance(node.func, ast.Name):
            return
        self._specialize_vararg_arguments(node)
        func_def = self._specialized_vararg_def(node.func.id)
        if func_def is None:
            self._call_imported_vararg_spec(node)
            return
        if not self._is_specializable_vararg_call(node, func_def):
            return
        extra = len(node.args) - len(func_def.args.args)
        kwnames = tuple(kw.arg for kw in node.keywords)
        node.func = ast.Name(id=self._vararg_specialization(func_def, extra, kwnames),
                             ctx=ast.Load())
        node.args = node.args + [kw.value for kw in node.keywords]
        node.keywords = []
        ast.fix_missing_locations(node)

    def _call_imported_vararg_spec(self, node):
        """Call the fixed-arity copy the module of an imported variadic def built.

        Each module is preprocessed on its own, so the copy is requested by the
        importer's call arities before that module is finalized (#8480).
        """
        name = node.func.id
        if (name not in self._vararg_once_bound or node.keywords
                or any(isinstance(arg, ast.Starred) for arg in node.args)):
            return
        spec = self._imported_vararg_specs.get((name, len(node.args)))
        if spec is None:
            return
        self._used_imported_vararg_specs.setdefault(name, set()).add(spec)
        node.func = ast.copy_location(ast.Name(id=spec, ctx=ast.Load()), node.func)

    def _specialize_external_varargs(self):
        """Build the copies of this module's variadic defs that an importer calls."""
        for name, arities in self.external_vararg_arities.items():
            func_def = self._specialized_vararg_def(name)
            if func_def is None or self._vararg_def_owners.get(id(func_def)) is not None:
                continue
            for arity in sorted(arities):
                if self._is_specializable_vararg_arity(func_def, arity):
                    extra = arity - len(func_def.args.args)
                    self.exported_vararg_specs[(name, arity)] = self._vararg_specialization(
                        func_def, extra)

    def _import_used_vararg_specs(self, module):
        """Import each called copy alongside the variadic def it stands for."""
        for node in ast.walk(module):
            if not isinstance(node, ast.ImportFrom):
                continue
            node.names += [
                ast.copy_location(ast.alias(name=spec, asname=None), alias)
                for alias in list(node.names) if alias.asname is None
                for spec in sorted(self._used_imported_vararg_specs.get(alias.name, ()))
            ]

    def _specialized_vararg_def(self, name):
        func_def = self._vararg_func_defs.get(name)
        if func_def is None or not self._packed_params(func_def):
            return None
        return func_def

    def _packed_params(self, func_def):
        """The *args and **kwargs names `func_def`'s body reads."""
        return {
            arg.arg
            for arg in (func_def.args.vararg, func_def.args.kwarg)
            if arg is not None and self._name_is_used(func_def.body, arg.arg)
        }

    def _specialize_vararg_arguments(self, node):
        """Pass a fixed-arity copy of a variadic def handed to a module function.

        `call(s)` qualifies when `call` reads its parameter only to call it,
        always with the same number of positional arguments, so the copy for
        that arity behaves as `s` does at every call (#8498).
        """
        callee = self._vararg_callee_defs.get(node.func.id)
        if (callee is None or callee.decorator_list
                or any(isinstance(arg, ast.Starred) for arg in node.args)):
            return
        params = [arg.arg for arg in callee.args.args]
        for index, arg in enumerate(node.args[:len(params)]):
            func_def = self._specialized_vararg_def(arg.id) if isinstance(arg, ast.Name) else None
            arity = self._param_call_arity(callee, params[index])
            if func_def is None or arity is None:
                continue
            if self._is_specializable_vararg_arity(func_def, arity):
                extra = arity - len(func_def.args.args)
                node.args[index] = ast.copy_location(
                    ast.Name(id=self._vararg_specialization(func_def, extra), ctx=ast.Load()), arg)

    def _param_call_arity(self, func_def, param):
        """The positional arity of every call of `param` in `func_def`'s body.

        None unless each read of `param` is the callee of such a call, all of
        the same arity, and nothing in the body rebinds or shadows `param`.
        """
        arities = set()
        callees = set()
        for node in (n for stmt in func_def.body for n in ast.walk(stmt)):
            if param in self._bound_names(node):
                return None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == param:
                if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
                    return None
                arities.add(len(node.args))
                callees.add(id(node.func))
            elif isinstance(node, ast.Name) and node.id == param and id(node) not in callees:
                return None
        return arities.pop() if len(arities) == 1 else None

    def _vararg_specialization(self, func_def, extra, kwnames=()):
        """The name of `func_def`'s copy taking `extra` packed arguments.

        `kwnames` are the keywords packed into **kwargs, in call order, which
        is the order the dict iterates in.
        """
        owner = self._vararg_def_owners.get(id(func_def))
        # The owner prefix keeps same-named defs nested in different functions
        # from sharing one specialization, which lives in only one of them.
        prefix = f"{owner.name}__" if owner is not None else ""
        # Each name is prefixed by its length, so no two keyword lists collide.
        kw_suffix = "".join(f"_{len(name)}{name}" for name in kwnames)
        if kwnames:
            kw_suffix = "_kw" + kw_suffix
        spec_name = f"{prefix}{func_def.name}__esbmc_va{extra}{kw_suffix}"
        if spec_name not in self._vararg_specializations:
            self._vararg_specializations[spec_name] = self._build_vararg_specialization(
                func_def, spec_name, extra, kwnames)
            owner_key = id(owner) if owner is not None else None
            owned = self._vararg_owner_specs.setdefault(owner_key, {})
            owned.setdefault(id(func_def), []).append(spec_name)
        self._vararg_dropped_defs.add(id(func_def))
        return spec_name

    def _build_vararg_specialization(self, func_def, spec_name, extra, kwnames):
        spec = copy.deepcopy(func_def)
        spec.name = spec_name
        pad = [f"__esbmc_va_{i}" for i in range(extra)]
        kw_pad = [f"__esbmc_kw_{i}" for i in range(len(kwnames))]
        spec.args.args = spec.args.args + [ast.arg(arg=name) for name in pad + kw_pad]
        packed_params = self._packed_params(func_def)
        packs = []
        vararg_name = spec.args.vararg.arg if spec.args.vararg else None
        if vararg_name in packed_params:
            elts = [ast.Name(id=name, ctx=ast.Load()) for name in pad]
            packs.append(
                ast.Assign(targets=[ast.Name(id=vararg_name, ctx=ast.Store())],
                           value=ast.Tuple(elts=elts, ctx=ast.Load())))
        kwarg_name = spec.args.kwarg.arg if spec.args.kwarg else None
        if kwarg_name in packed_params:
            packed = ast.Dict(keys=[ast.Constant(value=name) for name in kwnames],
                              values=[ast.Name(id=name, ctx=ast.Load()) for name in kw_pad])
            packs.append(
                ast.Assign(targets=[ast.Name(id=kwarg_name, ctx=ast.Store())], value=packed))
        spec.args.vararg = None
        spec.args.kwarg = None
        # Every call of a copy passes all its arguments, and a default would
        # otherwise attach to the trailing packed slots.
        spec.args.defaults = []
        spec.body = packs + spec.body
        self.functionParams[spec_name] = [arg.arg for arg in spec.args.args]
        self.functionKwonlyParams[spec_name] = []
        if vararg_name in packed_params and vararg_name not in self._names_bound_in(func_def.body):
            self._spell_out_forwarded_varargs(spec, vararg_name, pad)
        ast.fix_missing_locations(spec)
        return spec

    def _forwards_own_varargs(self, node):
        """Whether each `*x` in call `node` forwards the visited def's own *args.

        Each fixed-arity copy of the def spells such a call out, since the
        copy knows how many arguments `x` packs (#8480).
        """
        func_def = self._vararg_scope_stack[-1] if self._vararg_scope_stack else None
        if func_def is None or func_def.args.vararg is None or not isinstance(node.func, ast.Name):
            return False
        vararg = func_def.args.vararg.arg
        params = func_def.args.posonlyargs + func_def.args.args + func_def.args.kwonlyargs
        bound = self._names_bound_in(func_def.body) | {arg.arg for arg in params}
        if vararg in bound or node.func.id in bound:
            return False
        return all(
            isinstance(arg.value, ast.Name) and arg.value.id == vararg for arg in node.args
            if isinstance(arg, ast.Starred))

    def _names_bound_in(self, body):
        return {
            name
            for stmt in body
            for inner in ast.walk(stmt)
            for name in self._bound_names(inner)
        }

    def _spell_out_forwarded_varargs(self, spec, vararg_name, pad):
        """Pass each `*vararg_name` in `spec` as the names it packs, then bind the call."""
        mixin = self

        def forwarded(arg):
            return (isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name)
                    and arg.value.id == vararg_name)

        class Forward(ast.NodeTransformer):
            """Splice the packed names into each call that forwards `*vararg_name`."""

            def visit_Call(self, node):  # pylint: disable=invalid-name,missing-function-docstring
                self.generic_visit(node)
                if not any(forwarded(arg) for arg in node.args):
                    return node
                args = []
                for arg in node.args:
                    if forwarded(arg):
                        args.extend(
                            ast.copy_location(ast.Name(id=name, ctx=ast.Load()), arg)
                            for name in pad)
                    else:
                        args.append(arg)
                node.args = args
                raising = mixin._bind_call_arguments(node)  # pylint: disable=protected-access
                if raising is not None:
                    return raising
                mixin._specialize_vararg_call(node)  # pylint: disable=protected-access
                return node

        Forward().visit(spec)

    def _inject_vararg_specializations(self, node, owner_key=None):
        """Put the specializations owned by `node` where their originals sat.

        A nested def is copied into the function that encloses it rather than
        into the module, so the copy keeps the scope it closes over, and it
        takes the original's position so the names it reads are already bound
        there (#6800).
        """
        specs_by_source = self._vararg_owner_specs.pop(owner_key, None)
        if not specs_by_source:
            return
        specializations = [
            self._vararg_specializations[name] for names in specs_by_source.values()
            for name in names
        ]
        for spec in specializations:
            self.ensure_all_locations(spec)
            ast.fix_missing_locations(spec)
        survivors = [stmt for stmt in node.body if id(stmt) not in self._vararg_dropped_defs]
        body = []
        for stmt in node.body:
            body.extend(self._vararg_specializations[name]
                        for name in specs_by_source.get(id(stmt), ()))
            # A variadic def is dead only once every reference to it has been
            # rewritten; a call this pass cannot lower (`*` unpacking, an alias,
            # a caller in another module) must keep the original, so the
            # frontend reports the unsupported construct rather than a call to a
            # function that no longer exists.
            if (id(stmt) not in self._vararg_dropped_defs
                    or self._name_is_used(survivors + specializations, stmt.name)):
                body.append(stmt)
        node.body = body

    @staticmethod
    def _sequence_display_length(value):
        if not isinstance(value, (ast.List, ast.Tuple)):
            return None
        if any(isinstance(e, ast.Starred) for e in value.elts):
            return None
        return len(value.elts)

    @staticmethod
    def _bound_names(node):
        """Names `node` binds or rebinds without being a plain Name store."""
        if isinstance(node, ast.Name):
            return [node.id] if not isinstance(node.ctx, ast.Load) else []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return [node.name]
        if isinstance(node, ast.arg):
            return [node.arg]
        if isinstance(node, ast.alias):
            return [node.asname or node.name.split(".")[0]]
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return list(node.names)
        name = getattr(node, "name", None) or getattr(node, "rest", None)
        return [name] if isinstance(name, str) else []

    @staticmethod
    def _is_length_preserving_use(name, parent, grandparent, len_bound):
        """A load of `name` under `parent` that can neither resize nor alias it."""
        if isinstance(parent, ast.Starred):
            return isinstance(grandparent, ast.Call) and parent in grandparent.args
        if isinstance(parent, ast.Subscript):
            return parent.value is name and isinstance(parent.ctx, ast.Load)
        if isinstance(parent, (ast.For, ast.comprehension)):
            return parent.iter is name
        if isinstance(parent, ast.Call):
            return (not len_bound and isinstance(parent.func, ast.Name) and parent.func.id == "len"
                    and parent.args == [name] and not parent.keywords)
        return isinstance(parent, ast.Compare)

    def _scan_fixed_length_starred(self, module):
        """Map each `f(*xs)` argument whose `xs` has a statically known length.

        `xs` qualifies when it is bound once in the whole module, by a direct
        statement of its scope, to a list or tuple display, and every read of
        the name only indexes, iterates, measures, compares or unpacks it. An
        unpacking is mapped only below that statement, where `xs` is bound.
        """
        parents = {}
        bind_counts = {}
        unsafe = set()
        calls_to = {}
        for node in ast.walk(module):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
            for bound in self._bound_names(node):
                bind_counts[bound] = bind_counts.get(bound, 0) + 1
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls_to.setdefault(node.func.id, []).append(node)
        len_bound = "len" in bind_counts
        for node, parent in parents.items():
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            if not self._is_length_preserving_use(node, parent, parents.get(parent), len_bound):
                unsafe.add(node.id)
        fixed = {}
        defs = [
            n for n in ast.walk(module) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for scope in [module] + defs:
            for index, stmt in enumerate(scope.body):
                self._record_fixed_length_starred(stmt, scope.body[index + 1:], bind_counts, unsafe,
                                                  fixed)
        for scope in defs:
            fixed.update(
                self._fixed_length_starred_params(scope, calls_to.get(scope.name, []), parents,
                                                  bind_counts, unsafe))
        return fixed

    def _fixed_length_starred_params(self, scope, calls, parents, bind_counts, unsafe):
        """Map each `g(..., *p)` on a parameter p of `scope` that every one of
        its `calls` passes the same kind and length of display for (#8959).

        Every read of the def's name must be one of those calls, so it is never
        called through another name. A list at one call and a tuple at another
        give p no single type, and `g` must be bound once, since the frontend
        binds a call by its callee's bare name.
        """
        fixed = {}
        if not self._called_only_by(scope, calls, parents, bind_counts):
            return fixed
        for index, param in enumerate(scope.args.posonlyargs + scope.args.args):
            displays = {self._passed_display(call, index) for call in calls}
            if len(displays) != 1 or None in displays:
                continue
            (_, length), = displays
            if length is None or param.arg in unsafe or bind_counts.get(param.arg) != 1:
                continue
            fixed.update(
                dict.fromkeys(self._starred_reads(scope, param.arg, parents, bind_counts), length))
        return fixed

    @staticmethod
    def _called_only_by(scope, calls, parents, bind_counts):
        """Whether `scope` is bound once and read only as the callee of `calls`."""
        loads = sum(
            isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == scope.name
            for n in parents)
        return (not scope.decorator_list and calls and loads == len(calls)
                and bind_counts.get(scope.name) == 1)

    @staticmethod
    def _starred_reads(scope, name, parents, bind_counts):
        """Each `*name` in `scope` passed to a callee bound once."""
        return [
            node for node in (n for stmt in scope.body for n in ast.walk(stmt))
            if isinstance(node, ast.Starred) and isinstance(node.value, ast.Name)
            and node.value.id == name and isinstance(parents[node].func, ast.Name)
            and bind_counts.get(parents[node].func.id) == 1
        ]

    def _passed_display(self, call, index):
        """The kind and length of the display `call` passes at `index`, or None."""
        if call.keywords or index >= len(call.args) or any(
                isinstance(arg, ast.Starred) for arg in call.args):
            return None
        return type(call.args[index]), self._sequence_display_length(call.args[index])

    def _record_fixed_length_starred(self, stmt, later, bind_counts, unsafe, fixed):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target = stmt.target
        else:
            return
        length = self._sequence_display_length(stmt.value)
        if (not isinstance(target, ast.Name) or length is None or target.id in unsafe
                or bind_counts.get(target.id) != 1):
            return
        for node in (n for s in later for n in ast.walk(s)):
            if (isinstance(node, ast.Starred) and isinstance(node.value, ast.Name)
                    and node.value.id == target.id):
                fixed[node] = length

    def _expand_starred_call_args(self, node):
        """Splice `*seq` into the call when `seq` has a statically known length."""
        args = []
        for arg in node.args:
            if not isinstance(arg, ast.Starred):
                args.append(arg)
            elif self._sequence_display_length(arg.value) is not None:
                args.extend(arg.value.elts)
            elif arg in self._fixed_length_starred:
                args.extend(
                    ast.copy_location(
                        ast.Subscript(value=ast.Name(id=arg.value.id, ctx=ast.Load()),
                                      slice=ast.Constant(value=i),
                                      ctx=ast.Load()), arg)
                    for i in range(self._fixed_length_starred[arg]))
            else:
                args.append(arg)
        node.args = args
        ast.fix_missing_locations(node)

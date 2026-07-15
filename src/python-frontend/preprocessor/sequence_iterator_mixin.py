"""Preprocessor lowering for the explicit iterator protocol on statically-sized
immutable sequences (``it = seq.__iter__()`` then ``next(it)`` / ``it.__next__()``)."""

import ast


class SequenceIteratorMixin:
    """Lowering for the explicit iterator protocol on statically-sized
    sequences.

    Handles the pattern ``it = seq.__iter__()`` followed by ``next(it)`` or
    ``it.__next__()``. The ``__iter__`` call already returns the sequence
    object itself (see ``__iter__ on builtin iterables`` in the C/C++ frontend),
    so ``it`` aliases the sequence and ``it[k]`` reads its k-th element. Each
    ``next``/``__next__`` is rewritten to ``it[k]`` with a per-iterator
    compile-time index, raising ``StopIteration`` once the captured length is
    exhausted (CPython semantics).

    Because the consumption index is tracked statically in source order, the
    rewrite is only sound for iterators consumed in straight-line code. The
    pre-scan (``_scan_sequence_iterators``) blacklists any iterator whose
    binding or consumption is inside a loop, comprehension, or conditional, or
    that is rebound, consumed in a non-rewritable position, or closed over
    across scopes; blacklisted iterators are left untouched (falling back to the
    frontend's existing behaviour, so there is no regression).
    """

    # ---- constant / length helpers -------------------------------------

    def _seq_const_int(self, node):
        """Best-effort evaluation of ``node`` to a Python int, or None.

        Resolves integer constants, negations thereof, and names bound to an
        integer literal (via ``_known_literal_values``)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)):
            inner = self._seq_const_int(node.operand)
            return None if inner is None else -inner
        if isinstance(node, ast.Name):
            bound = self._known_literal_values.get(node.id)
            if bound is not None and bound is not node:
                return self._seq_const_int(bound)
        return None

    def _range_static_len(self, call):
        """Length of a ``range(...)`` call with constant integer args, or None."""
        if call.keywords or not 1 <= len(call.args) <= 3:
            return None
        args = [a for a in (self._seq_const_int(x) for x in call.args) if a is not None]
        if len(args) != len(call.args):
            return None
        if len(args) == 1:
            start, stop, step = 0, args[0], 1
        elif len(args) == 2:
            start, stop, step = args[0], args[1], 1
        else:
            start, stop, step = args[0], args[1], args[2]
        if step == 0:
            return None
        if step > 0:
            return max(0, (stop - start + step - 1) // step)
        return max(0, (start - stop - step - 1) // (-step))

    def _static_len_of(self, node):
        """Static length of ``node`` as a sequence, or None if not known.

        Only *immutable* sequences qualify: ``range(<const>)`` calls, tuple
        literals, and names previously bound to either. Lists are excluded on
        purpose — an iterator aliases the underlying list object, so a mutation
        between two ``next()`` calls (``xs.append(...)``) would make a captured
        static length diverge from CPython's live iterator. range and tuple
        cannot be mutated in valid Python, so their length is fixed at binding."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "range":
                return self._range_static_len(node)
            if node.func.id == "tuple" and len(node.args) == 1 \
                    and not node.keywords:
                return self._static_len_of(node.args[0])
        if isinstance(node, ast.Tuple):
            if any(isinstance(e, ast.Starred) for e in node.elts):
                return None
            return len(node.elts)
        if isinstance(node, ast.Name):
            return self._static_seq_len.get(node.id)
        return None

    # ---- pattern matchers ----------------------------------------------

    @staticmethod
    def _is_dunder_iter_call(node):
        """True for ``RECV.__iter__()`` with no arguments."""
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "__iter__" and not node.args and not node.keywords)

    @staticmethod
    def _seq_next_consumed_name(call):
        """Name of the iterator consumed by ``next(it)`` / ``it.__next__()``."""
        if not isinstance(call, ast.Call) or call.keywords:
            return None
        func = call.func
        if (isinstance(func, ast.Name) and func.id == "next" and len(call.args) == 1
                and isinstance(call.args[0], ast.Name)):
            return call.args[0].id
        if (isinstance(func, ast.Attribute) and func.attr == "__next__" and not call.args
                and isinstance(func.value, ast.Name)):
            return func.value.id
        return None

    # ---- pre-scan (soundness gate) -------------------------------------

    @staticmethod
    def _ast_types(*names):
        """Tuple of the named ``ast`` node classes that exist in this Python
        (some — ``Match``, ``TryStar`` — are version-dependent)."""
        return tuple(t for t in (getattr(ast, n, None) for n in names) if t is not None)

    def _scan_sequence_iterators(self, node):
        """Blacklist iterator names that cannot be lowered with a static index.

        Static index tracking is only sound when the binding and every
        consumption of an iterator run exactly once, in source order, in the
        same scope. An iterator is disqualified when any of the following holds:

          * it is bound or consumed inside a *non-linear* construct — a loop,
            comprehension, or conditional (`if`/`try`/`with`/`match`) — where
            source order no longer matches runtime execution order;
          * a consumption appears in a position the rewriter does not handle
            (anything other than a bare ``x = next(it)`` / ``next(it)``
            statement), e.g. a call argument, ``return``, or nested expression,
            because the real ``next()`` left in place would advance the live
            iterator out of step with the static index;
          * it is rebound (assigned more than once); or
          * its binding and a consumption are in different scopes (a closure
            over an outer iterator, which may run any number of times).

        Blacklisted iterators are left untouched, falling back to the frontend's
        existing behaviour."""
        self._seq_iter_blacklist = set()
        assign_counts = {}
        binding_scope = {}
        consume_scopes = {}
        nonlinear_types = self._ast_types("For", "AsyncFor", "While", "ListComp", "SetComp",
                                          "DictComp", "GeneratorExp", "If", "IfExp", "Try",
                                          "TryStar", "With", "AsyncWith", "Match")
        scope_types = self._ast_types("FunctionDef", "AsyncFunctionDef", "Lambda", "ClassDef")

        # Consumption calls sitting in a directly-rewritable statement position:
        # the call is the entire value of an ``Assign`` or an ``Expr``.
        rewritable = set()
        for stmt in ast.walk(node):
            if isinstance(stmt, (ast.Assign, ast.Expr)) \
                    and self._seq_next_consumed_name(stmt.value) is not None:
                rewritable.add(id(stmt.value))

        counter = [0]

        def recurse(n, in_nonlinear, scope):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        assign_counts[t.id] = assign_counts.get(t.id, 0) + 1
                        if self._is_dunder_iter_call(n.value):
                            binding_scope[t.id] = scope
                            if in_nonlinear:
                                self._seq_iter_blacklist.add(t.id)
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) \
                    and isinstance(n.target, ast.Name):
                assign_counts[n.target.id] = \
                    assign_counts.get(n.target.id, 0) + 1
            if isinstance(n, ast.Call):
                consumed = self._seq_next_consumed_name(n)
                if consumed is not None:
                    consume_scopes.setdefault(consumed, set()).add(scope)
                    if in_nonlinear or id(n) not in rewritable:
                        self._seq_iter_blacklist.add(consumed)
            if isinstance(n, scope_types):
                counter[0] += 1
                child_scope, child_nonlinear = counter[0], False
            else:
                child_scope = scope
                child_nonlinear = in_nonlinear or isinstance(n, nonlinear_types)
            for child in ast.iter_child_nodes(n):
                recurse(child, child_nonlinear, child_scope)

        recurse(node, False, 0)
        for name, count in assign_counts.items():
            if count > 1:
                self._seq_iter_blacklist.add(name)
        for name, scopes in consume_scopes.items():
            if scopes != {binding_scope.get(name)}:
                self._seq_iter_blacklist.add(name)

    # ---- tracking + rewriting ------------------------------------------

    def _track_static_seq_binding(self, node):
        """Record/clear ``_static_seq_len`` for ``name = <sized sequence>``."""
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            return
        name = node.targets[0].id
        length = self._static_len_of(node.value)
        if length is None:
            self._static_seq_len.pop(name, None)
        else:
            self._static_seq_len[name] = length

    def _maybe_track_seq_iterator(self, node):
        """Record iterator state for ``it = RECV.__iter__()`` when RECV is a
        statically-sized sequence and ``it`` is not blacklisted."""
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            return
        if not self._is_dunder_iter_call(node.value):
            return
        # Only a variable receiver (``r.__iter__()``) yields a valid iterator
        # binding in the frontend; ``range(4).__iter__()`` on a call expression
        # is unsupported, so tracking it would be dead. The name's length is
        # resolved via ``_static_seq_len``.
        receiver = node.value.func.value
        if not isinstance(receiver, ast.Name):
            return
        name = node.targets[0].id
        if name in self._seq_iter_blacklist:
            return
        length = self._static_len_of(receiver)
        if length is None:
            return
        self.seq_iterator_length[name] = length
        self.seq_iterator_index[name] = 0

    def _make_seq_next_stop(self, template):
        raise_node = ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="StopIteration", ctx=ast.Load()),
                args=[ast.Constant(value="StopIteration")],
                keywords=[],
            ),
            cause=None,
        )
        ast.copy_location(raise_node, template)
        ast.fix_missing_locations(raise_node)
        return raise_node

    def _build_seq_next(self, iter_var, targets, template):
        """Lower a single ``next``/``__next__`` consumption of ``iter_var``.

        Returns a list of statements (the indexed assignment, or a
        ``StopIteration`` raise when the sequence is exhausted)."""
        idx = self.seq_iterator_index[iter_var]
        length = self.seq_iterator_length[iter_var]
        if idx >= length:
            return [self._make_seq_next_stop(template)]
        self.seq_iterator_index[iter_var] = idx + 1
        element = ast.Subscript(
            value=ast.Name(id=iter_var, ctx=ast.Load()),
            slice=ast.Constant(value=idx),
            ctx=ast.Load(),
        )
        if targets is not None:
            stmt = ast.Assign(targets=targets, value=element, type_comment=None)
        else:
            stmt = ast.Expr(value=element)
        ast.copy_location(stmt, template)
        self.ensure_all_locations(stmt, template)
        ast.fix_missing_locations(stmt)
        return [stmt]

    def _maybe_rewrite_seq_next_assign(self, node):
        """Rewrite ``x = next(it)`` / ``x = it.__next__()`` for a tracked
        iterator, or return None to leave the node unchanged."""
        iter_var = self._seq_next_consumed_name(node.value)
        if iter_var is None or iter_var not in self.seq_iterator_index:
            return None
        return self._build_seq_next(iter_var, node.targets, node)

    def _maybe_rewrite_seq_next_expr(self, node):
        """Rewrite a standalone ``next(it)`` / ``it.__next__()`` statement."""
        iter_var = self._seq_next_consumed_name(node.value)
        if iter_var is None or iter_var not in self.seq_iterator_index:
            return None
        return self._build_seq_next(iter_var, None, node)

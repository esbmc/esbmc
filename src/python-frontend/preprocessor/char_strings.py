"""Nondet strings held as typed lists of their character codes.

A str bound only to `nondet_str()` and used only through `len(s)`, `s[i]`
compared with `==` or `!=` against another such character or a one-character
ASCII literal, or passed to a followed function, behaves the same when it is
the list of its character codes: comparing two characters for equality is
comparing their codes, and the length and IndexError are the list's. The
list copies the same bounded buffer `nondet_str()` builds today, so its
capacity is the model's own and nothing is truncated. The typed list model
(#10451) then holds it. A character read into a name, as a `for c in s` loop
does, makes that name a character too, used only in the same comparisons.
Any other use of the str or of the character keeps today's str model.
"""

import ast

from .typed_lists import MODULE, TypedListGate, _parent, _unshare, _walk_scope

# The call the gate types as a list allocation and the converter lowers.
STR_CODES = "__ESBMC_str_codes"


class _CharStrings:

    def __init__(self, module):
        self.module = module
        self.gate = TypedListGate(module)
        self.edges = {}  # location: locations bound to or from it

    def run(self):
        _unshare(self.module)
        for node in ast.walk(self.module):
            for child in ast.iter_child_nodes(node):
                setattr(child, "_typed_parent", node)
        gate = self.gate
        gate.verifier_modules = gate._verifier_modules()
        gate._build_scopes()
        gate._collect_bindings()
        strs, chars = self._decide()
        if strs:
            self._rewrite(strs, chars)
        for node in ast.walk(self.module):
            node.__dict__.pop("_typed_parent", None)

    # --- which strs ------------------------------------------------------------

    def _decide(self):
        """The str and the character locations, largest sets that pass."""
        strs = self._str_candidates()
        chars = {
            loc
            for loc, values in self.gate.bindings.items()
            if values and all(self._read_of(v, s, strs) for v, s in values)
        }
        while True:
            bad = self._failing(strs, chars)
            if not bad:
                return strs, chars
            strs -= self._closure(bad & strs)
            chars -= bad

    def _failing(self, strs, chars):
        """The str and character locations with a use or binding the
        rewrite does not cover, given @p strs and @p chars."""
        uses = self._uses(strs | chars)

        def all_uses(loc, ok):
            return all(ok(n, s, strs, chars) for s, n in uses.get(loc, []))

        bad = {loc for loc in strs if not all_uses(loc, self._str_use)}
        bad |= {loc for loc in chars if not all_uses(loc, self._char_use)}
        bad |= {
            loc
            for loc in chars
            if not all(self._read_of(v, s, strs) for v, s in self.gate.bindings[loc])
        }
        return bad | self._nested_or_declared(strs | chars)

    def _str_candidates(self):
        """Locations bound only to `nondet_str()` or to one another, closed
        both ways; any other binding rejects the whole group."""
        bindings, edges = self.gate.bindings, self.edges
        for loc, values in bindings.items():
            for value, scope in values:
                if isinstance(value, ast.Name):
                    edges.setdefault(loc, set()).add(scope.loc(value.id))
                    edges.setdefault(scope.loc(value.id), set()).add(loc)
        strs, seen = set(), set()
        for seed, values in bindings.items():
            if seed in seen or not any(self._nondet_str(v) for v, _ in values):
                continue
            group, todo, ok = set(), [seed], True
            while todo:
                loc = todo.pop()
                if loc in group:
                    continue
                group.add(loc)
                todo.extend(edges.get(loc, ()))
                ok = ok and all(
                    isinstance(v, ast.Name) or self._nondet_str(v)
                    for v, _ in bindings.get(loc, [(None, None)]))
            seen |= group
            if ok:
                strs |= group
        return strs

    def _closure(self, locs):
        """@p locs and every str location bound to or from one of them."""
        out, todo = set(), list(locs)
        while todo:
            loc = todo.pop()
            if loc not in out:
                out.add(loc)
                todo.extend(self.edges.get(loc, ()))
        return out

    def _nondet_str(self, value):
        if not (isinstance(value, ast.Call) and not value.args and not value.keywords):
            return False
        func = value.func
        if isinstance(func, ast.Attribute):
            return (func.attr == "nondet_str" and isinstance(func.value, ast.Name)
                    and func.value.id in self.gate.verifier_modules)
        return (isinstance(func, ast.Name) and func.id == "nondet_str"
                and "nondet_str" not in self.gate.shadowed)

    def _uses(self, locs):
        uses = {}
        for scope in self.gate.scopes:
            for node in _walk_scope(scope.body):
                if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        and scope.loc(node.id) in locs):
                    uses.setdefault(scope.loc(node.id), []).append((scope, node))
        return uses

    def _nested_or_declared(self, locs):
        bad = set()
        for scope in self.gate.scopes:
            for name in scope.declared:
                bad |= {scope.loc(name), (MODULE, name)} & locs
            for node in _walk_scope(scope.body):
                if node is not self.module and isinstance(
                        node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                               ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    for name in self.gate._nested_names(node, scope):
                        bad |= {scope.loc(name), (MODULE, name)} & locs
        return bad

    def _read_of(self, value, scope, strs):
        """Whether @p value is `s[i]` of a str location in @p strs."""
        return (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)
                and isinstance(value.ctx, ast.Load) and not isinstance(value.slice, ast.Slice)
                and scope.loc(value.value.id) in strs and not _has_len_call(value.slice))

    def _char(self, node, scope, strs, chars):
        """A character: a read of a str, a character name, or a one-character
        ASCII literal."""
        if isinstance(node, ast.Constant):
            return isinstance(node.value, str) and len(node.value) == 1 and ord(node.value) < 128
        if isinstance(node, ast.Name):
            return scope.loc(node.id) in chars
        return self._read_of(node, scope, strs)

    def _char_compare(self, operand, scope, strs, chars):
        """@p operand is one side of `a == b` or `a != b` of two characters."""
        compare = _parent(operand)
        if not (isinstance(compare, ast.Compare) and len(compare.ops) == 1
                and isinstance(compare.ops[0], (ast.Eq, ast.NotEq))):
            return False
        return all(
            self._char(o, scope, strs, chars) for o in (compare.left, compare.comparators[0]))

    def _str_use(self, name, scope, strs, chars):
        parent = _parent(name)
        if self._len_of(parent, name):
            return True
        if self._read_of(parent, scope, strs) and parent.value is name:
            return self._char_read_use(parent, scope, strs, chars)
        if isinstance(parent,
                      ast.Call) and name in parent.args and self.gate._analysed_call(parent):
            func = self.gate.functions[parent.func.id]
            param = func.args.args[parent.args.index(name)]
            return (func.name, param.arg) in strs and param.annotation is None
        target = _single_target(parent)
        return target is not None and parent.value is name and scope.loc(target.id) in strs

    def _len_of(self, call, name):
        """Whether @p call is `len(name)` of the builtin len."""
        return (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "len" and "len" not in self.gate.shadowed
                and call.args == [name] and not call.keywords)

    def _char_read_use(self, read, scope, strs, chars):
        """`s[i]` compared with a character, or bound to a character name."""
        if self._char_compare(read, scope, strs, chars):
            return True
        holder = _parent(read)
        target = _single_target(holder)
        return target is not None and holder.value is read and scope.loc(target.id) in chars

    def _char_use(self, name, scope, strs, chars):
        return self._char_compare(name, scope, strs, chars)

    # --- the rewrite -----------------------------------------------------------

    def _rewrite(self, strs, chars):
        for scope in self.gate.scopes:
            for node in list(_walk_scope(scope.body)):
                target = _single_target(node)
                if (target is not None and scope.loc(target.id) in strs
                        and self._nondet_str(node.value)):
                    node.value = ast.copy_location(
                        ast.Call(func=ast.Name(id=STR_CODES, ctx=ast.Load()),
                                 args=[node.value],
                                 keywords=[]), node.value)
                elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                      and self._char_compare(node, scope, strs, chars)):
                    node.value = ord(node.value)


def _has_len_call(index):
    """An index holding `len(...)`: its value is unsigned, and the list read
    of an unsigned index raises no IndexError (today's str model does)."""
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len"
        for n in ast.walk(index))


def _single_target(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name):
        return node.targets[0]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target
    return None


def type_char_strings(module):
    """Rewrite each nondet str of @p module the rewrite covers into the
    list of its character codes."""
    _CharStrings(module).run()

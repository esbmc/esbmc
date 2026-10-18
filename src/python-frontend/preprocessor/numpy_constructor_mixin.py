"""Binding of the arguments of the NumPy array constructors."""
import ast
import copy


class NumpyConstructorMixin:
    """Spell a NumPy constructor call in the one form the frontend reads.

    The frontend reads the leading value parameters of a constructor
    positionally and the others as keywords. Binding them here gives every
    reader the same call; a call CPython rejects raises its TypeError when it
    runs, after its arguments are evaluated.
    """

    # Parameters in declaration order (NumPy 2), how many of them the frontend
    # reads positionally, and the keyword-only ones.
    _NUMPY_CONSTRUCTOR_SIGNATURES = {
        "arange": (("start", "stop", "step", "dtype"), 3, ("device", "like")),
        "linspace":
        (("start", "stop", "num", "endpoint", "retstep", "dtype", "axis"), 3, ("device", )),
        "full": (("shape", "fill_value", "dtype", "order"), 2, ("device", "like")),
        "zeros": (("shape", "dtype", "order"), 1, ("device", "like")),
        "ones": (("shape", "dtype", "order"), 1, ("device", "like")),
        "eye": (("N", "M", "k", "dtype", "order"), 2, ("device", "like")),
        "identity": (("n", "dtype"), 1, ("like", )),
        "array": (("object", "dtype"), 1, ("copy", "order", "subok", "ndmin", "like")),
    }
    _NUMPY_REQUIRED_PARAMS = {
        "linspace": ("start", "stop"),
        "full": ("shape", "fill_value"),
        "zeros": ("shape", ),
        "ones": ("shape", ),
        "eye": ("N", ),
        "identity": ("n", ),
        "array": ("object", ),
    }

    def _record_numpy_import(self, node):
        """Remember the names a module binds to numpy and its constructors."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy":
                    self.numpy_module_names.add(alias.asname or "numpy")
        elif node.module == "numpy":
            for alias in node.names:
                if alias.name in self._NUMPY_CONSTRUCTOR_SIGNATURES:
                    self.numpy_constructor_names[alias.asname or alias.name] = alias.name

    def _numpy_constructor_name(self, node):
        func = node.func
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id in self.numpy_module_names
                and func.attr in self._NUMPY_CONSTRUCTOR_SIGNATURES):
            return func.attr
        if isinstance(func, ast.Name):
            return self.numpy_constructor_names.get(func.id)
        return None

    def _bind_numpy_constructor_call(self, node):
        """Rewrite a NumPy constructor call to its canonical form in place.

        Returns the call lowered to a raising expression when its arguments
        do not bind, and None otherwise.
        """
        name = self._numpy_constructor_name(node)
        if name is None or any(isinstance(arg, ast.Starred) for arg in node.args):
            return None
        positional = self._NUMPY_CONSTRUCTOR_SIGNATURES[name][1]
        if not node.keywords and len(node.args) <= positional:
            return None  # already in the canonical form
        if any(kw.arg is None for kw in node.keywords):
            return None  # **kwargs: the names are not statically known
        try:
            bound = self._bind_numpy_parameters(name, node)
        except TypeError as error:
            return self.visit(self._raising_call(node, str(error)))
        self._spell_numpy_constructor_call(name, node, bound)
        return None

    def _bind_numpy_parameters(self, name, node):
        params, _, keyword_only = self._NUMPY_CONSTRUCTOR_SIGNATURES[name]
        if len(node.args) > len(params):
            raise TypeError(f"{name}() takes at most {len(params)} positional arguments "
                            f"({len(node.args)} given)")
        bound = dict(zip(params, node.args))
        for kw in node.keywords:
            if kw.arg in bound:
                raise TypeError(f"{name}() got multiple values for argument '{kw.arg}'")
            if kw.arg not in params and kw.arg not in keyword_only:
                raise TypeError(f"{name}() got an unexpected keyword argument '{kw.arg}'")
            bound[kw.arg] = kw.value
        if name == "arange":
            self._bind_arange_stop(node, bound)
        for param in self._NUMPY_REQUIRED_PARAMS.get(name, ()):
            if param not in bound:
                raise TypeError(f"{name}() missing required argument '{param}'")
        return bound

    @staticmethod
    def _bind_arange_stop(node, bound):
        """`arange(n)` is `arange(0, n)`: a lone start given by position is the stop."""
        if "stop" in bound:
            bound.setdefault("start", ast.Constant(value=0))
            return
        if "start" not in bound or not node.args:
            raise TypeError("arange() requires stop to be specified.")
        bound["stop"] = bound.pop("start")
        bound["start"] = ast.Constant(value=0)

    def _spell_numpy_constructor_call(self, name, node, bound):
        params, positional, keyword_only = self._NUMPY_CONSTRUCTOR_SIGNATURES[name]
        args = []
        for param in params[:positional]:
            if param not in bound:
                break
            args.append(bound.pop(param))
        keywords = [
            ast.keyword(arg=param, value=bound[param]) for param in params + keyword_only
            if param in bound
        ]
        source_order = list(node.args) + [kw.value for kw in node.keywords]
        bound_order = args + [kw.value for kw in keywords]
        source_ids = [id(value) for value in source_order]
        reordered = [id(value) for value in bound_order if id(value) in source_ids] != source_ids
        if reordered and not all(self._is_pure_numpy_argument(value) for value in source_order):
            raise NotImplementedError(
                f"numpy.{name}() arguments with side effects bound out of their "
                "evaluation order are not supported")
        node.args = args
        node.keywords = keywords
        ast.fix_missing_locations(node)

    @classmethod
    def _is_pure_numpy_argument(cls, node):
        if isinstance(node, (ast.Constant, ast.Name)):
            return True
        if isinstance(node, ast.UnaryOp):
            return cls._is_pure_numpy_argument(node.operand)
        if isinstance(node, ast.Attribute):
            return isinstance(node.value, ast.Name)
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(cls._is_pure_numpy_argument(elt) for elt in node.elts)
        return False

    def _unpacked_linspace_call(self, node):
        """The linspace call of `a, s = np.linspace(...)`, or None."""
        target = node.targets[0] if len(node.targets) == 1 else None
        if not (isinstance(target, ast.Tuple) and len(target.elts) == 2
                and all(isinstance(elt, ast.Name) for elt in target.elts)):
            return None
        call = node.value
        if not isinstance(call, ast.Call) or self._numpy_constructor_name(call) != "linspace":
            return None
        if any(isinstance(arg, ast.Starred) for arg in call.args):
            return None
        return call if all(kw.arg is not None for kw in call.keywords) else None

    def _split_linspace_retstep(self, node):
        """`a, s = np.linspace(..., retstep=True)` as `a = np.linspace(...)`
        followed by `s = step`, the step NumPy returns: (stop - start) / div,
        or NaN when div, num - 1 with the endpoint and num without it, is not
        positive. Only for arguments that are safe to read twice."""
        call = self._unpacked_linspace_call(node)
        bound = self._retstep_bound_arguments(call) if call is not None else None
        if bound is None or self._reads_unpacked_target(node, bound):
            return None
        bound.pop("retstep")
        num = bound.get("num", ast.Constant(value=50))
        endpoint = bound.get("endpoint", ast.Constant(value=True))
        step = self._linspace_step(bound, num.value - 1 if endpoint.value else num.value)
        call.args = []
        call.keywords = [ast.keyword(arg=param, value=value) for param, value in bound.items()]
        target = node.targets[0]
        array_assign = ast.Assign(targets=[target.elts[0]], value=call)
        step_assign = ast.Assign(targets=[target.elts[1]], value=step)
        for statement in (array_assign, step_assign):
            ast.copy_location(statement, node)
            ast.fix_missing_locations(statement)
        return [array_assign, step_assign]

    def _retstep_bound_arguments(self, call):
        """The bound arguments of a `retstep=True` call whose step can be
        computed here: constant num and endpoint, start and stop safe to read
        twice. None otherwise."""
        try:
            bound = self._bind_numpy_parameters("linspace", call)
        except TypeError:
            return None  # the call raises its TypeError where it stands
        retstep = bound.get("retstep")
        if not (isinstance(retstep, ast.Constant) and retstep.value is True):
            return None
        num = bound.get("num", ast.Constant(value=50))
        endpoint = bound.get("endpoint", ast.Constant(value=True))
        if not (isinstance(num, ast.Constant) and isinstance(num.value, int)
                and not isinstance(num.value, bool) and isinstance(endpoint, ast.Constant)):
            return None
        if not all(self._is_pure_numpy_argument(bound[param]) for param in ("start", "stop")):
            return None
        return bound

    @staticmethod
    def _reads_unpacked_target(node, bound):
        """Whether start or stop reads a name the unpacking rebinds: the step,
        computed after the array, would read the rebound name."""
        assigned = {elt.id for elt in node.targets[0].elts}
        return any(
            isinstance(sub, ast.Name) and sub.id in assigned for param in ("start", "stop")
            for sub in ast.walk(bound[param]))

    @staticmethod
    def _linspace_step(bound, div):
        if div <= 0:
            return ast.Call(func=ast.Name(id="float", ctx=ast.Load()),
                            args=[ast.Constant(value="nan")],
                            keywords=[])
        return ast.BinOp(left=ast.BinOp(left=copy.deepcopy(bound["stop"]),
                                        op=ast.Sub(),
                                        right=copy.deepcopy(bound["start"])),
                         op=ast.Div(),
                         right=ast.Constant(value=div))

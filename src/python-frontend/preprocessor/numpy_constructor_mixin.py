"""Binding of the arguments of the NumPy array constructors."""
import ast


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
        "full": (("shape", "fill_value", "dtype", "order"), 2, ("device", "like")),
        "zeros": (("shape", "dtype", "order"), 1, ("device", "like")),
        "ones": (("shape", "dtype", "order"), 1, ("device", "like")),
        "eye": (("N", "M", "k", "dtype", "order"), 2, ("device", "like")),
        "identity": (("n", "dtype"), 1, ("like", )),
        "array": (("object", "dtype"), 1, ("copy", "order", "subok", "ndmin", "like")),
    }
    _NUMPY_REQUIRED_PARAMS = {
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

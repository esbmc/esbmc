import ast


class HelperFunctionsMixin:

    def _create_helper_functions(self):
        """Create the ESBMC helper function definitions."""
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

        def _make_int_arg(name):
            return ast.arg(arg=name, annotation=ast.Name(id='int', ctx=ast.Load()))

        def _name(n):
            return ast.Name(id=n, ctx=ast.Load())

        def _binop(l, op, r):
            return ast.BinOp(left=l, op=op, right=r)

        def _cmp(l, op, r):
            return ast.Compare(left=l, ops=[op], comparators=[r])

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
                    test=_cmp(_name('step'), ast.Gt(), ast.Constant(value=0)),
                    body=[
                        ast.If(
                            test=_cmp(_name('stop'), ast.LtE(), _name('start')),
                            body=[
                                ast.Return(value=_binop(_name('start'), ast.Sub(), _name('step')))
                            ],
                            orelse=[],
                        ),
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
                            test=_cmp(_name('stop'), ast.GtE(), _name('start')),
                            body=[
                                ast.Return(value=_binop(_name('start'), ast.Sub(), _name('step')))
                            ],
                            orelse=[],
                        ),
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

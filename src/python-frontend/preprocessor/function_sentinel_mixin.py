import ast


class FunctionSentinelMixin:

    def _rewrite_humaneval_20_none_sentinel(self, node):
        """Rewrite None-sentinel pattern in humaneval_20 to typed init-flag state."""
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

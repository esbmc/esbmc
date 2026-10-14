"""Position a preprocessor rejection in the user's program (#7547)."""

import ast


def attach_location(exc: BaseException, node: ast.AST, module_name: str) -> None:
    """Record ``node``'s position on ``exc``, unless it already carries one.

    The innermost visit frame wins: it attaches first as the exception
    unwinds, so the position is the narrowest node that could not be handled.
    """
    if getattr(exc, "esbmc_location", None):
        return
    if isinstance(exc, SyntaxError) and exc.filename:
        return
    line = getattr(node, "lineno", None)
    if line is None:
        return
    exc.esbmc_location = (module_name, line, getattr(node, "col_offset", 0))

import ast
from dataclasses import dataclass


class Block:

    def __init__(self):
        self.stmts = [
        ]  # list of AST nodes; first item may be a condition object stored as ('cond', node)
        self.succ = []  # successor block indices
        self.pred = []


class CFG:

    def __init__(self):
        self.blocks = []


@dataclass
class LoopContext:
    header: int
    break_target: int
    continue_target: int


#look for the set, append etc methods fro dataflow
def add_edge(cfg: CFG, src_idx: int | None, dst_idx: int | None):

    if src_idx is None or dst_idx is None:
        return

    if dst_idx not in cfg.blocks[src_idx].succ:
        cfg.blocks[src_idx].succ.append(dst_idx)

    if src_idx not in cfg.blocks[dst_idx].pred:
        cfg.blocks[dst_idx].pred.append(src_idx)


def build_cfg_for_function(func_node: ast.FunctionDef) -> CFG:
    cfg = CFG()

    entry_idx = new_block(cfg)
    exit_idx = new_block(cfg)

    tail = build_statements(func_node.body, cfg, entry_idx, None)

    if tail is not None:
        add_edge(cfg, tail, exit_idx)

    return cfg


def new_block(cfg):
    block = Block()
    cfg.blocks.append(block)
    return len(cfg.blocks) - 1


def build_while(cfg: CFG,
                stmt: ast.While,
                current_idx: int,
                outer_loop: LoopContext | None = None) -> int:

    header_idx = len(cfg.blocks)
    cfg.blocks.append(Block())
    add_edge(cfg, current_idx, header_idx)

    cfg.blocks[header_idx].stmts.append(("cond", stmt.test))

    body_idx = len(cfg.blocks)
    cfg.blocks.append(Block())

    exit_idx = len(cfg.blocks)
    cfg.blocks.append(Block())

    add_edge(cfg, header_idx, body_idx)

    if stmt.orelse:
        else_idx = len(cfg.blocks)
        cfg.blocks.append(Block())

        add_edge(cfg, header_idx, else_idx)
    else:
        else_idx = None
        add_edge(cfg, header_idx, exit_idx)

    loop_ctx = LoopContext(
        header=header_idx,
        break_target=exit_idx,
        continue_target=header_idx,
    )

    body_tail = build_statements(
        stmt.body,
        cfg,
        body_idx,
        loop_ctx,
    )

    if body_tail is not None:
        add_edge(cfg, body_tail, header_idx)

    if else_idx is not None:
        else_tail = build_statements(
            stmt.orelse,
            cfg,
            else_idx,
            outer_loop,
        )

        add_edge(cfg, else_tail, exit_idx)

    return exit_idx


def build_for(cfg: CFG, stmt: ast.For, current_idx: int, outer_loop: LoopContext | None) -> int:

    header_idx = len(cfg.blocks)
    cfg.blocks.append(Block())
    add_edge(cfg, current_idx, header_idx)

    cfg.blocks[header_idx].stmts.append(('for', stmt.target, stmt.iter))

    body_idx = len(cfg.blocks)
    cfg.blocks.append(Block())

    exit_idx = len(cfg.blocks)
    cfg.blocks.append(Block())

    add_edge(cfg, header_idx, body_idx)

    if stmt.orelse:
        else_idx = new_block(cfg)
        add_edge(cfg, header_idx, else_idx)
    else:
        else_idx = None
        add_edge(cfg, header_idx, exit_idx)

    loop_ctx = LoopContext(header=header_idx, break_target=exit_idx, continue_target=header_idx)

    body_tail = build_statements(
        stmt.body,
        cfg,
        body_idx,
        loop_ctx,
    )

    if body_tail is not None:
        add_edge(cfg, body_tail, header_idx)

    if else_idx is not None:
        else_tail = build_statements(stmt.orelse, cfg, else_idx, outer_loop)
        add_edge(cfg, else_tail, exit_idx)

    return exit_idx


def build_statements(statements, cfg, current_idx: int, loop_ctx: LoopContext | None = None):
    current = current_idx

    for stmt in statements:
        if current is None:
            break
        if isinstance(stmt, ast.If):
            current = build_if(stmt, cfg, current, loop_ctx)
        elif isinstance(stmt, ast.While):
            current = build_while(cfg, stmt, current, loop_ctx)
        elif isinstance(stmt, ast.For):
            current = build_for(cfg, stmt, current, loop_ctx)
        elif isinstance(stmt, ast.Break):
            if loop_ctx is None:
                raise RuntimeError("break outside loop")
            add_edge(cfg, current, loop_ctx.break_target)
            #return None
            current = None
            continue
        elif isinstance(stmt, ast.Continue):
            if loop_ctx is None:
                raise RuntimeError("continue outside loop")
            add_edge(cfg, current, loop_ctx.continue_target)
            current = None
            continue
            #return None
        elif isinstance(stmt, ast.With):
            current = build_with(cfg, stmt, current, loop_ctx)
        elif isinstance(stmt, ast.Try):
            current = build_try_statement(stmt, cfg, current, loop_ctx)
        else:
            cfg.blocks[current].stmts.append(stmt)

        if is_terminal_statement(stmt):
            current = None
            #continue

    return current


def build_try_statement(stmt, cfg, current_idx, outer_loop):
    """
    Build the CFG for:

        try:
            BODY
        except E1:
            HANDLER1
        except E2:
            HANDLER2
        else:
            ORELSE
        finally:
            FINALBODY
    """

    try_entry = new_block(cfg)
    after_try = new_block(cfg)

    add_edge(cfg, current_idx, try_entry)

    try_tail = build_statements(stmt.body, cfg, try_entry, outer_loop)

    # Normal completion of try -> try_end.
    #cfg.blocks[try_tail_idx].succ.append(try_end_idx)
    handler_entries: list[int] = []
    handler_tails: list[int] = []

    # Exception handlers.
    for handler in stmt.handlers:
        handler_idx = new_block(cfg)
        handler_entries.append(handler_idx)
        handler_block = cfg.blocks[handler_idx]
        # `except X as e:`
        if handler.name is not None:
            handler_block.stmts.append(
                ast.Assign(
                    targets=[ast.Name(
                        id=handler.name,
                        ctx=ast.Store(),
                    )],
                    value=ast.Name(
                        id="Exception",
                        ctx=ast.Load(),
                    ),
                ))

        #handler_idx = handler_block.index
        handler_tail = build_statements(
            handler.body,
            cfg,
            handler_idx,
            outer_loop,
        )
        if handler_tail is not None:
            handler_tails.append(handler_tail)

        # Exception edge from the try region to the handler.
    for handler_idx in handler_entries:
        add_edge(cfg, try_entry, handler_idx)

    if try_tail is not None:
        add_edge(cfg, try_tail, after_try)

    for handler_tail in handler_tails:
        add_edge(cfg, handler_tail, after_try)

    return after_try


def is_terminal_statement(stmt):
    return isinstance(stmt, (ast.Return, ast.Raise))


def build_if(stmt: ast.If, cfg: CFG, current_idx: int, loop_ctx: LoopContext | None = None) -> int:
    cfg.blocks[current_idx].stmts.append(("cond", stmt.test))

    then_idx = new_block(cfg)

    else_idx = new_block(cfg)

    join_idx = new_block(cfg)

    add_edge(cfg, current_idx, then_idx)
    add_edge(cfg, current_idx, else_idx)

    then_tail = build_statements(
        stmt.body,
        cfg,
        then_idx,
        loop_ctx,
    )

    else_tail = build_statements(stmt.orelse, cfg, else_idx, loop_ctx)

    add_edge(cfg, then_tail, join_idx)

    add_edge(cfg, else_tail, join_idx)

    return join_idx


def build_with(
    cfg: CFG,
    stmt: ast.With,
    current_idx: int,
    loop_ctx: LoopContext | None = None,
) -> int:

    body_idx = new_block(cfg)
    exit_idx = new_block(cfg)

    add_edge(cfg, current_idx, body_idx)

    cfg.blocks[body_idx].stmts.append(stmt)

    body_tail = build_statements(stmt.body, cfg, body_idx, loop_ctx)

    add_edge(cfg, body_tail, exit_idx)

    return exit_idx

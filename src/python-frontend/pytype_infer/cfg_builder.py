import ast
from dataclasses import dataclass

class Block:
    def __init__(self):
        self.stmts = []   # list of AST nodes; first item may be a condition object stored as ('cond', node)
        self.succ = []    # successor block indices
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
    return len(cfg.blocks) -1

def build_while(cfg: CFG, stmt: ast.While, current_idx: int, outer_loop: LoopContext | None = None) -> int:
    
    header_idx = len(cfg.blocks)
    cfg.blocks.append(Block())
    add_edge(cfg, current_idx, header_idx)

    cfg.blocks[header_idx].stmts.append(
        ("cond", stmt.test)
    )

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

    cfg.blocks[header_idx].stmts.append(
        ('for', stmt.target, stmt.iter)
    )

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

    loop_ctx = LoopContext(
        header=header_idx,
        break_target=exit_idx,
        continue_target=header_idx
    )

    body_tail = build_statements(
        stmt.body,
        cfg,
        body_idx,
        loop_ctx,
    )

    add_edge(cfg, body_tail, header_idx)

    if else_idx is not None:
        else_tail = build_statements(
            cfg,
            stmt.orelse,
            else_idx,
            outer_loop
        )
        add_edge(cfg, else_tail, exit_idx)

    return exit_idx



def build_statements(statements, cfg, current_idx: int, loop_ctx: LoopContext | None = None):
    current = current_idx

    for stmt in statements:
        if isinstance(stmt, ast.If):
            current = build_if(stmt, cfg, current)

        elif isinstance(stmt, ast.While):
            current = build_while(cfg, stmt, current, loop_ctx)    
        elif isinstance(stmt, ast.For):
            current = build_for(cfg, stmt, current, loop_ctx)
        elif isinstance(stmt, ast.Break):
            if loop_ctx is not None:
                add_edge(cfg, current_idx,loop_ctx.break_target)
            return None
        elif isinstance(stmt, ast.Continue):
            if loop_ctx is not None:
                add_edge(cfg, current_idx, loop_ctx.continue_target)
            return None 
                   
        else:
            cfg.blocks[current].stmts.append(stmt)

            if is_terminal_statement(stmt):
                return None
    return current

def is_terminal_statement(stmt):
    return isinstance(
        stmt,
        (ast.Return, ast.Raise)
    )

def build_if(stmt: ast.If, cfg:CFG, current_idx: int) -> int: 
    cfg.blocks[current_idx].stmts.append(("cond", stmt.test))

    then_idx = len(cfg.blocks)
    cfg.blocks.append(Block())

    else_idx = len(cfg.blocks)
    cfg.blocks.append(Block())

    join_idx = len(cfg.blocks)
    cfg.blocks.append(Block())

    add_edge(cfg, current_idx, then_idx)
    add_edge(cfg, current_idx, else_idx)

    then_tail = build_statements(
        stmt.body,
        cfg,
        then_idx
    )

    else_tail = build_statements(stmt.orelse, cfg, else_idx)

    add_edge(cfg, then_tail, join_idx)

    add_edge(cfg, else_tail, join_idx)

    return join_idx            
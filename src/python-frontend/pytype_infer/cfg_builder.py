import ast

class Block:
    def __init__(self):
        self.stmts = []   # list of AST nodes; first item may be a condition object stored as ('cond', node)
        self.succ = []    # successor block indices
        self.pred = []


class CFG:
    def __init__(self):
        self.blocks = []

#look for the set, append etc methods fro dataflow
def build_cfg_for_function(func_node: ast.FunctionDef) -> CFG:
    cfg = CFG()
    entry = Block(); 

    cfg.blocks.append(entry)

    build_statements(func_node, CFG, 0)

    return cfg

def build_statements(statements, cfg, current_idx):
    current = current_idx

    for stmt in statements:
        if isinstance(stmt, ast.If):
            current = build_if(stmt, cfg, current)
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
    current = cfg.blocks[current_idx]

    current.stmts.append(("cond", stmt.test))

    then_idx = len(cfg.blocks)
    cfg.blocks.append(Block())

    else_idx = len(cfg.blocks)
    cfg.blocks.append(Block())

    join_idx = len(cfg.blocks)
    cfg.blocks.append(Block())

    current.succ.extend([then_idx, else_idx])

    cfg.blocks[then_idx].pred.append(current_idx)
    cfg.blocks[else_idx].pred.append(current_idx)

    then_tail = build_statements(
        stmt.body,
        cfg,
        then_idx
    )

    else_tail = build_statements(stmt.orelse, cfg, else_idx)

    if then_tail is not None:
        cfg.blocks[then_tail].succ.append(join_idx)
        cfg.blocks[join_idx].pred.append(then_tail)

    if else_tail is not None:
        cfg.blocks[else_tail].succ.append(join_idx)
        cfg.blocks[join_idx].pred.append(else_tail)

    return join_idx            
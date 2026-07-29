import ast

class Block:
    def __init__(self):
        self.stmts = []   # list of AST nodes; first item may be a condition object stored as ('cond', node)
        self.succ = []    # successor block indices
        self.pred = []


class CFG:
    def __init__(self):
        self.blocks = []


def build_cfg_for_function(func_node: ast.FunctionDef) -> CFG:
    cfg = CFG()
    entry = Block(); cfg.blocks.append(entry)
    cur = entry
    for stmt in func_node.body:
        if isinstance(stmt, ast.If):
            # current block gets condition entry marker (store test)
            cur.stmts.append(('cond', stmt.test))
            # then block
            then_block = Block()
            then_block.stmts.extend(stmt.body)
            # else block
            else_block = Block()
            else_block.stmts.extend(stmt.orelse)
            # join block
            join_block = Block()
            idx_then = len(cfg.blocks); cfg.blocks.append(then_block)
            idx_else = len(cfg.blocks); cfg.blocks.append(else_block)
            idx_join = len(cfg.blocks); cfg.blocks.append(join_block)
            cur.succ.extend([idx_then, idx_else])
            cfg.blocks[idx_then].pred.append(cfg.blocks.index(cur))
            cfg.blocks[idx_else].pred.append(cfg.blocks.index(cur))
            cfg.blocks[idx_then].succ.append(idx_join)
            cfg.blocks[idx_else].succ.append(idx_join)
            cfg.blocks[idx_join].pred.extend([idx_then, idx_else])
            cur = join_block
        else:
            cur.stmts.append(stmt)
    return cfg

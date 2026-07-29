import ast
from typing import Dict, Any
from .dataflow_solver import analyze_function, infer_type_from_expr
from .lattice import *

def derive_param_and_return_types(func_node: ast.FunctionDef, out_envs: list):
    # derive return types by scanning Return nodes in the function body and infer their expr types
    ret_type = Unknown()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value is not None:
            t = infer_type_from_expr(node.value, {})  # limited: no env here
            ret_type = t if isinstance(ret_type, Unknown) else ret_type.join(t)
    # merge out_envs for params
    merged = {}
    for be in out_envs:
        for k,v in be.items():
            if k in merged:
                merged[k] = merged[k].join(v)
            else:
                merged[k] = v
    param_types = {}
    for arg in func_node.args.args:
        param_types[arg.arg] = merged.get(arg.arg, Unknown())
    return param_types, ret_type

def annotate_function_with_env_and_signatures(func_node: ast.FunctionDef, out_envs: list):
    param_types, ret_type = derive_param_and_return_types(func_node, out_envs)
    # annotate params
    for arg in func_node.args.args:
        t = param_types.get(arg.arg, Unknown())
        if not isinstance(t, Unknown):
            arg.annotation = ast.Name(id=t.to_ann_name(), ctx=ast.Load())
    # annotate return
    if not isinstance(ret_type, Unknown):
        func_node.returns = ast.Name(id=ret_type.to_ann_name(), ctx=ast.Load())
    # annotate locals: merge out_envs
    merged = {}
    for e in out_envs:
        for k,v in e.items():
            if k in merged:
                merged[k] = merged[k].join(v)
            else:
                merged[k] = v
    print("Merged", merged)            
    new_body = []
    for stmt in func_node.body:
        # convert name = lambda ... to AnnAssign(name: callable = lambda ...)
        print(type(stmt))
        if isinstance(stmt, ast.Assign) and len(stmt.targets)>=1 and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Lambda):
            name = stmt.targets[0].id
            lam = stmt.value
            ret_t = infer_type_from_expr(lam.body, {}) 
            ann_name = 'callable' if isinstance(ret_t, Unknown) else 'callable'
            ann = ast.AnnAssign(target=ast.Name(id=name, ctx=ast.Store()),
                                 annotation=ast.Name(id=ann_name, ctx=ast.Load()),
                                 value=stmt.value,
                                 simple=1)
            ast.copy_location(ann, stmt)
            new_body.append(ann)
            continue
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) >= 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                if name in merged and not isinstance(merged[name], Unknown):
                    t = merged[name]
                    ann_name = t.to_ann_name()
                    ann = ast.AnnAssign(
                        target=ast.Name(id=name, ctx=ast.Store()),
                        annotation=ast.Name(id=ann_name, ctx=ast.Load()),
                        value=stmt.value,
                        simple=1
                    )
                    ast.copy_location(ann, stmt)
                    new_body.append(ann)
                    continue
        new_body.append(stmt)
    func_node.body = new_body
    return func_node

def annotate_module_with_outenvs(module_node: ast.Module, out_envs_map):
    for node in module_node.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in out_envs_map:
                out_envs = out_envs_map[node.name]
                annotate_function_with_env_and_signatures(node, out_envs)
    return module_node

def annotate_ast(ast_node, opts=None):
    out_envs_by_func = {}
    for node in ast_node.body:
        if isinstance(node, ast.FunctionDef):
            cfg, in_envs, out_envs = analyze_function(node)
            print(cfg)
            print(out_envs)
            out_envs_by_func[node.name] = out_envs
    annotate_module_with_outenvs(ast_node, out_envs_by_func)
    ast.fix_missing_locations(ast_node)
    return ast_node

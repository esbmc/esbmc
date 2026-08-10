import ast
from typing import Dict, Any
from .dataflow_solver import analyze_function, infer_type_from_expr
from .lattice import *

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

def annotate_parameters(funct_node: ast.FunctionDef, param_types):
    for arg in funct_node.args.args:
        t = param_types.get(arg.arg, Unknown())

        if isinstance(t, Unknown):
            continue

        arg.annotation = ast.Name(
            id=t.to_ann_name(),
            ctx=ast.Load()
        )

def annotate_return(func_node: ast.FunctionDef, ret_type):
    if isinstance(ret_type, Unknown):
        return

    func_node.returns = ast.Name(
        id = ret_type.to_ann_name(),
        ctx=ast.Load()
    )   

def merge_out_envs(out_envs):
    merged = {}

    for env in out_envs:
        for name, typ in env.items():
            if name in merged:
                merged[name] = merged[name].join(typ)
            else:
                merged[name] = typ 

    return merged        

def convert_assign_to_annassign(stmt, inferred_type):
    ann = ast.AnnAssign(
        target=ast.Name(
            id=stmt.targets[0].id,
            ctx=ast.Store()
        ),
        annotation=ast.Name(
            id=inferred_type.to_ann_name(),
            ctx=ast.Load()
        ),
        value=stmt.value,
        simple=1
    )

    ast.copy_location(ann, stmt)
    return ann

def convert_lambda_assignment(stmt):
        lam = stmt.value

        ann = ast.AnnAssign(
            target=ast.Name(
                id=stmt.targets[0].id,
                ctx=ast.Store()
            ),
            annotation=ast.Name(
                id="callable",
                ctx=ast.Load()
            ),
            value=lam,
            simple=1
        )

        ast.copy_location(ann, stmt)

        return ann

def is_simple_assignment(stmt):
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    )

def is_lambda_assignment(stmt):
    return (
        is_simple_assignment(stmt)
        and isinstance(stmt.value, ast.Lambda)
    )    

def annotate_function_with_env_and_signatures(func_node, out_envs):
    param_types, ret_type = derive_param_and_return_types(func_node, out_envs)

    annotate_parameters(func_node, param_types)
    annotate_return(func_node, ret_type)

    merged = merge_out_envs(out_envs)

    new_body = []

    for stmt in func_node.body:

        if is_lambda_assignment(stmt):
            new_body.append(convert_lambda_assignment(stmt))
            continue

        if is_typed_assignment(stmt, merged):
            t = merged[stmt.targets[0].id]

            new_body.append(convert_assign_to_annassign(stmt, t))
            continue
        
        new_body.append(stmt)

    func_node.body = new_body

    return func_node            

def is_typed_assignment(stmt, merged):
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id in merged
        and not isinstance(merged[stmt.targets[0].id], Unknown)
    )

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
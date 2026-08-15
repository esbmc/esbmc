import ast
from typing import Dict, Any
from .dataflow_solver import analyze_function, infer_type_from_expr, collect_known_classes, InferenceContext
from .lattice import *




def derive_param_and_return_types(func_node: ast.FunctionDef, out_envs: list, return_types:dict, context: InferenceContext):
    # derive return types by scanning Return nodes in the function body and infer their expr types
    
    # merge out_envs for params
    merged = {}
    for be in out_envs:
        for k,v in be.items():
            if k in merged:
                merged[k] = merged[k].join(v)
            else:
                merged[k] = v

    ret_type = Unknown()
    #context = InferenceContext(known_classes=known_classes,)

    #for node in ast.walk(func_node):
       # if isinstance(node, ast.Return) and node.value is not None:
        #    t = infer_type_from_expr(node.value, {}, context)  # limited: no env here
         #   ret_type = t if isinstance(ret_type, Unknown) else ret_type.join(t)
    for typ in return_types.values():
        if isinstance(ret_type, Unknown):
            ret_type = typ
        else:
            ret_type = ret_type.join(typ)

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

def annotate_function_with_env_and_signatures(func_node, out_envs, return_types, context: InferenceContext):
    #context = InferenceContext(known_classes=known_classes)
    
    param_types, ret_type = derive_param_and_return_types(func_node, out_envs, return_types, context,)

    annotate_parameters(func_node, param_types)
    annotate_return(func_node, ret_type)

    merged = merge_out_envs(out_envs)

    new_body = []

    already_annotated = set()

    func_node.body = process_statements(func_node.body, merged, already_annotated)

    for stmt in func_node.body:

        if is_lambda_assignment(stmt):
            new_body.append(convert_lambda_assignment(stmt))
            continue

        if is_typed_assignment(stmt, merged, already_annotated):
            t = merged[stmt.targets[0].id]

            print(f"[ANNOTATE DEBUG] converting assignment "
                f"{stmt.targets[0].id} -> {merged[stmt.targets[0].id].to_ann_name()}"
                )

            new_body.append(convert_assign_to_annassign(stmt, t))
            already_annotated.add(stmt.targets[0].id)
            continue
        
        new_body.append(stmt)

    func_node.body = new_body

    return func_node            

# def is_typed_assignment(stmt, merged):
#     return (
#         isinstance(stmt, ast.Assign)
#         and len(stmt.targets) == 1
#         and isinstance(stmt.targets[0], ast.Name)
#         and stmt.targets[0].id in merged
#         and not isinstance(merged[stmt.targets[0].id], Unknown)
#     )
def is_typed_assignment(stmt, merged, already_annotated):
    if not isinstance(stmt, ast.Assign):
        return False

    if len(stmt.targets) != 1:
        return False

    target = stmt.targets[0]

    if not isinstance(target, ast.Name):
        return False

    name = target.id

    if name in already_annotated:
        return False

    if name not in merged:
        return False

    inferred_type = merged[name]

    if isinstance(inferred_type, Unknown):
        return False

    return True

def process_statements(statements, merged, already_annotated):
    new_body = []

    for stmt in statements:
        if isinstance(stmt, ast.If):

            stmt.body = process_statements(stmt.orelse, merged, already_annotated.copy())

            stmt.orelse = process_statements(stmt.orelse, merged, already_annotated.copy())

            new_body.append(stmt)
            continue

        if is_typed_assignment(stmt, merged, already_annotated):
            name = stmt.targets[0].id
            t = merged[name]
            
            print(
                f"[ANNOTATE DEBUG] converting assignment "
                f"{name} -> {t.to_ann_name()}"
            )            

            new_body.append(convert_assign_to_annassign(stmt, t))
            already_annotated.add(name)
            continue

        if isinstance(stmt, (ast.For, ast.AsyncFor)):

            stmt.body = process_statements(stmt.body, merged, already_annotated.copy())

            stmt.orelse = process_statements(stmt.orelse, merged, already_annotated.copy())

            new_body.append(stmt)
            continue

        if isinstance(stmt, ast.While):

            stmt.body = process_statements(stmt.body, merged, already_annotated.copy())

            stmt.orelse = process_statements(stmt.orelse, merged, already_annotated.copy())

            new_body.append(stmt)
            continue

        if is_lambda_assignment(stmt):
            new_body.append(convert_lambda_assignment(stmt))
            continue

        new_body.append(stmt)

    return new_body
def is_container_type(t):
    return isinstance(
        t,
        (
            ListType,
            TupleType,
            DictType,
            SetType,
            CallableType,
            UnionType,
            InstanceType,

        )
    )

def annotate_module_with_outenvs(module_node: ast.Module, out_envs_by_func, return_types_by_func, context: InferenceContext, ):
    for node in module_node.body:
        if not isinstance(node, ast.FunctionDef):
                continue
        if node.name not in out_envs_by_func:
                continue
        
        out_envs = out_envs_by_func[node.name]
        return_types = return_types_by_func.get(node.name, {})
        #print(ast.dump(module_node, indent=4))
        print(ast.unparse(module_node))

        print(f"\n[ANNOTATE DEBUG] BEFORE function: {node.name}")
        print(ast.unparse(node))

        print("[ANNOTATE DEBUG] outenvs:")
        print(out_envs)
        annotate_function_with_env_and_signatures(node, out_envs, return_types, context)

        print(f"[ANNOTATE DEBUG] AFTER function: {node.name}")
        print(ast.unparse(node))

    return module_node

def annotate_ast(ast_node, opts=None):
    known_classes = collect_known_classes(ast_node)
    context = InferenceContext(known_classes=known_classes, )
    out_envs_by_func = {}
    return_types_by_func = {}
    for node in ast_node.body:
        if isinstance(node, ast.FunctionDef):
            cfg, in_envs, out_envs, return_types = analyze_function(node, context)
            print(cfg)
            print(out_envs)
            out_envs_by_func[node.name] = out_envs
            return_types_by_func[node.name] = return_types

           # param_types, ret_type = derive_param_and_return_types(node, out_envs, return_types, context,)
    annotate_module_with_outenvs(ast_node, out_envs_by_func, return_types_by_func, context, )

    ast.fix_missing_locations(ast_node)
    return ast_node

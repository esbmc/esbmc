import ast
from typing import Dict, List, Any, Tuple
from .lattice import *
from .cfg_builder import *

def parse_isinstance_condition(cond):
    if not (isinstance(cond, ast.Call)
    and isinstance(cond.func, ast.Name)
    and cond.func.id == "isinstance"
    and len(cond.args) >=2
    and isinstance(cond.args[0], ast.Name)
    ):
        return None
    var = cond.args[0].id
    type_expr = cond.args[1]

    if isinstance(type_expr, ast.Name):
        return ("isinstance", var, type_expr.id)

    if(isinstance(type_expr, ast.Subscript)
       and isinstance(type_expr.value, ast.Name)):

       return ("isinstance", var, type_expr.value.id)

    return None

def parse_none_condition(cond):
    if not isinstance(cond, ast.Compare):
        return None

    if not isinstance(cond.left, ast.Name):
        return None

    comparator = cond.comparators[0]

    if not (isinstance(comparator, ast.Constant)
            and comparator.value is None):
        return None

    if isinstance(cond.ops[0], ast.Is):
        return ("is_none", cond.left.id, False)

    if isinstance(cond.ops[0], ast.IsNot):
        return ("is_none", cond.left.id, True)

    return None                               

def parse_condition(cond):
    result = parse_isinstance_condition(cond)
    if result is not None:
        return result

    return parse_none_condition(cond)    

def analyze_function(func_node: ast.FunctionDef, max_iters=50):
    cfg = build_cfg_for_function(func_node)
    in_envs, out_envs = initialize_environments(func_node, cfg)

    changed = True
    iteration = 0

    while changed and iteration < max_iters:
        changed = False
        iteration +=1

        for i, block in enumerate(cfg.blocks):
            env, cond, stmts = prepare_block(block, in_envs, i)

            for stmt in stmts:
                env = transfer_statement(stmt, env)

            merged = merge_environment(out_envs[i], env)

            if merged != out_envs[i]:
                out_envs[i] = merged
                changed = True

            propagate_block(block,merged, cond, in_envs)

    return cfg, in_envs, out_envs  

def propagate_block(block, env, cond, in_envs):
    if cond is not None and block.succ:
        propagate_condition(
            block,
            env,
            cond,
            in_envs,
        )
    else:
        propagate_to_successors(
            block.succ,
            env,
            in_envs,
        )       
def propagate_condition(block, env, cond, in_envs):
    parsed = parse_condition(cond)

    if parsed is None or len(block.succ) < 2:
        propagate_to_successors(
            block.succ,
            env,
            in_envs,
        )
        return

    true_env, false_env = narrow_condition(
        parsed,
        env,
    )

    merge_successor_environment(
        block.succ[0],
        true_env,
        in_envs,
    )

    merge_successor_environment(
        block.succ[1],
        false_env,
        in_envs,
    )
def narrow_condition(parsed, env):
    true_env = env.copy()
    false_env = env.copy()

    kind = parsed[0]

    if kind == "isinstance":
        narrow_isinstance(
            parsed,
            true_env,
            false_env,
        )
    elif kind == "is_none":
        narrow_none(
            parsed,
            true_env,
            false_env,
        )

    return true_env, false_env

def narrow_isinstance(parsed, true_env, false_env):
    _, var, type_name = parsed

    if var in true_env:
        true_env[var] = (
            true_env[var].narrow_with_isinstance(type_name)
        )

    if var in false_env:
        false_env[var] = (
            false_env[var].remove_isinstance(type_name)
        )

def narrow_none(parsed, true_env, false_env):
    _, var, is_not = parsed

    if is_not:
        narrow_not_none(true_env, false_env, var)
    else:
        narrow_is_none(true_env, false_env, var)

def narrow_is_none(true_env, false_env, var):
    if var in true_env:
        true_env[var] = true_env[var].narrow_with_isinstance("None")

    if var in false_env:
        false_env[var] = false_env[var].remove_isinstance("None")

def narrow_not_none(true_env, false_env, var):
    if var in true_env:
        true_env[var] = true_env[var].remove_isinstance("None")

    if var in false_env:
        false_env[var] = false_env[var].narrow_with_isinstance("None")
           

def initialize_environments(func_node, cfg):
    n = len(cfg.blocks)

    in_envs = [dict() for _ in range(n)]
    out_envs = [dict() for _ in range(n)]

    entry_env = { arg.arg: Unknown()
                  for arg in func_node.args.args
                  }      

    if n > 0:
        in_envs[0] = entry_env.copy()

    return in_envs, out_envs

def compute_input_envirment(block, block_index, in_envs, out_envs):
    if not block.pred:
        return in_envs[block_index].copy()

    pred_env = []

    for predecessor in block.pred:
        merge_into_environment(
            pred_env,
            out_envs[predecessor],
        )

    return pred_env

def merge_into_environment(target, source):
    for name, typ in source.items():
        if name in target:
            target[name] = target[name].join(typ)
        else:
            target[name] = typ

def prepare_block(block, in_envs, block_index):
    env = in_envs[block_index].copy()

    if block.stmts and is_condition_marker(block.stmts[0]):
        cond = block.stmts[0][1]
        stmts = block.stmts[1:]
    else:
        cond = None
        stmts = block.stmts

    return env, cond, stmts

def is_condition_marker(stmt):
    return (isinstance(stmt, tuple)
             and len(stmt) >=2
             and stmt[0] == "cond")

def transfer_statement(stmt, env):
    if isinstance(stmt, ast.Assign):
        return transfer_assign(stmt, env)

    if isinstance(stmt, ast.AnnAssign):
        return transfer_annassign(stmt, env)

    if isinstance(stmt, ast.Expr):
        return transfer_expr(stmt, env)

    return env 

def transfer_assign(stmt, env):
    for target in stmt.targets:
        if isinstance(target, ast.Name):
            assign_name(target, stmt.value, env)

        elif isinstance(target, ast.Subscript):
            assign_subscript(target, stmt.value, env)

    return env 

def assign_name(target, value, env):
    name = target.id
    typ = infer_type_from_expr(value, env)

    if typ is None:
        typ = Unknown()

    if name in env:
        env[name] = env[name].join(typ)
    else:
        env[name] = typ 

def assign_subscript(target, value, env):
    if not isinstance(target.value, ast.Name):
        return

    container = target.value.id
    key_type = infer_type_from_expr(target.slice,env) 
    value_type = infer_type_from_expr(value, env)

    update_subscript_assignment(
        container,
        key_type,
        value_type,
        env,
    )
def update_subscript_assignment(container, key_type, value_type, env):
    current = env.get(container, Unknown())

    if isinstance(current, DictType):
        env[container] = DictType(
            current.key_t.join(key_type),
            current.val_t.join(value_type),
        )    
        return

    if isinstance(current, ListType):
        env[container] = ListType(current.elem.join(value_type))
        return

    if isinstance(current, Unknown):
        env[container] = infer_subscript_container(key_type, value_type,)
        return

    env[container] = Unknown()

def infer_subscript_container(key_type, value_type):
    if isinstance(key_type, StrType):
        return DictType(key_type, value_type)

    if isinstance(key_type, IntType):
        return ListType(value_type)

    return Unknown()

def transfer_annassign(stmt, env):
    if not isinstance(stmt.target, ast.Name):
        return env

    name = stmt.target.id
    annotation = stmt.annotation

    if isinstance(annotation, ast.Name):
        env[name] = mk_type_from_name(annotation.id)
    else:
        env[name] = Unknown()

    return env

def transfer_expr(stmt, env):
    expr = stmt.value

    if isinstance(expr, ast.Call):
        handle_call_effect(expr, env)

    return env    

def handle_call_effect(call, env):
    if not isinstance(call.func, ast.Attribute):
        return

    if call.func.attr == "append":
        handle_append(call, env) 

def handle_append(call, env):
    if not isinstance(call.func.vlaue, ast.Name):
        return

    if not call.args:
        return

    listname = call.func.value.id
    arg_type = infer_type_from_expr(call.args[0], env)
    current = env.get(listname, Unknown())

    if isinstance(current, ListType):
        env[listname] = ListType(
            current.elem.join(arg_type)
        )                            
        return

    env[listname] = ListType(arg_type)

def merge_environment(previous, current):
    merged= previous.copy()
    merge_into_environment(merged, current)
    return merged

def merge_types(old_t, new_type, use_widen=False):
    if old_t is None:
        return new_type
    return old_t.widen(new_type) if use_widen else new_type 

def merge_successor_environment(successor, env, in_envs):
    merge_into_environment(in_envs[successor], env)

def propagate_to_successors(successors, env, in_envs):
    for successor in successors:
        merge_successor_environment(
            successor,
            env,
            in_envs,
        )               
    
def resolve_method_call(obj_type, method, arg_types):
    """
    Infer the return type of a Python method call.
    Always returns a lattice Type.
    """


    if isinstance(obj_type, StrType):

        if method in {
            "upper", "lower", "capitalize", "title",
            "casefold", "swapcase",
            "strip", "lstrip", "rstrip",
            "replace", "removeprefix", "removesuffix",
            "expandtabs", "center", "ljust", "rjust",
            "zfill", "join", "translate",
        }:
            return StrType()

        if method in {
            "split",
            "rsplit",
            "splitlines",
        }:
            return ListType(StrType())

        if method in {
            "partition",
            "rpartition",
        }:
            return TupleType([
                StrType(),
                StrType(),
                StrType()
            ])

        if method in {
            "find",
            "rfind",
            "index",
            "rindex",
            "count",
        }:
            return IntType()

        if method in {
            "startswith",
            "endswith",
            "isalnum",
            "isalpha",
            "isascii",
            "isdecimal",
            "isdigit",
            "isidentifier",
            "islower",
            "isnumeric",
            "isprintable",
            "isspace",
            "istitle",
            "isupper",
        }:
            return BoolType()

        return Unknown()


    if isinstance(obj_type, ListType):

        if method == "copy":
            return ListType(obj_type.elem)

        if method == "pop":
            return obj_type.elem

        if method in {
            "append",
            "extend",
            "insert",
            "remove",
            "clear",
            "reverse",
            "sort",
        }:
            return NoneType()

        if method in {
            "count",
            "index",
        }:
            return IntType()

        return Unknown()

    if isinstance(obj_type, TupleType):

        if method in {
            "count",
            "index",
        }:
            return IntType()

        return Unknown()

    if isinstance(obj_type, DictType):

        if method == "copy":
            return DictType(
                obj_type.key_t,
                obj_type.val_t
            )

        if method == "get":
            return obj_type.val_t

        if method == "pop":
            return obj_type.val_t

        if method == "popitem":
            return TupleType([
                obj_type.key_t,
                obj_type.val_t
            ])

        if method == "keys":
            return ListType(obj_type.key_t)

        if method == "values":
            return ListType(obj_type.val_t)

        if method == "items":
            return ListType(
                TupleType([
                    obj_type.key_t,
                    obj_type.val_t
                ])
            )

        if method in {
            "update",
            "clear",
            "setdefault",
        }:
            return NoneType()

        return Unknown()

    # if isinstance(obj_type, SetType):

    #     if method == "copy":
    #         return SetType(obj_type.elem)

    #     if method == "pop":
    #         return obj_type.elem

    #     if method in {
    #         "union",
    #         "intersection",
    #         "difference",
    #         "symmetric_difference",
    #     }:
    #         return SetType(obj_type.elem)

    #     if method in {
    #         "add",
    #         "clear",
    #         "discard",
    #         "remove",
    #         "update",
    #         "intersection_update",
    #         "difference_update",
    #         "symmetric_difference_update",
    #     }:
    #         return NoneType()

    #     if method in {
    #         "issubset",
    #         "issuperset",
    #         "isdisjoint",
    #     }:
    #         return BoolType()

    #     return Unknown()

    if isinstance(obj_type, CallableType):

        return obj_type.ret

    return Unknown()    
                                                                          

def infer_type_from_expr(expr, env):
    if isinstance(expr, ast.Constant):
        v = expr.value
        if isinstance(v, bool):
            return BoolType()
        if isinstance(v, int):
            return IntType()
        if isinstance(v, float):
            return FloatType()
        if v is None:
            return NoneType()
        if isinstance(v, str):
            return StrType()
    if isinstance(expr, ast.List):
        elem = Unknown()
        for e in expr.elts:
            elem = elem.join(infer_type_from_expr(e, env))
        return ListType(elem)
    if isinstance(expr, ast.Name):
        return env.get(expr.id, Unknown())
    if isinstance(expr, ast.BinOp):
        left = infer_type_from_expr(expr.left, env)
        right = infer_type_from_expr(expr.right, env)
        
        if isinstance(expr.op, ast.Add):

            if isinstance(left, ListType) and isinstance(right, ListType):
                return ListType(left.elem.join(right.elem))

            if isinstance(left, StrType) and isinstance(right, StrType):
                return StrType()

            if (isinstance(left, TupleType) and isinstance(right, TupleType)):
                return TupleType(left.elems + right.elems)
        
        if isinstance(left, FloatType) or isinstance(right, FloatType):
            return FloatType()

        if isinstance(left, IntType) or isinstance(right, IntType):
            return IntType()                     
        return Unknown()

    if isinstance(expr, ast.Subscript):
        if isinstance(expr.value, ast.Name):
            name = expr.value.id
            t = env.get(name, Unknown())
            if isinstance(t, ListType):
                return t.elem
            if isinstance(t, DictType):
                return t.val_t
        return Unknown()
    if isinstance(expr, ast.Call):
       func = expr.func

       if isinstance(func, ast.Name):
            f_name = func.id

            if f_name == "len":
                return IntType()
            if f_name == "str":
                return StrType()
            if f_name == "int":
                return IntType()
            if f_name == "bool":
                 return BoolType()
            if f_name == "float":
                 return FloatType()
            if f_name == "list":
                 return ListType(Unknown())
            if f_name == "dict":
                 return DictType(Unknown(), Unknown())
            if f_name == "tuple":
                 return TupleType([])
            #if f_name in known_classes:
             #   return InstanceType(f_name)

            return Unknown()                             
       
       if isinstance(func, ast.Attribute):

            obj_type = infer_type_from_expr(func.value, env)

            arg_types = [
                infer_type_from_expr(arg, env)
                for arg in expr.args
            ]

            t = resolve_method_call(
                obj_type,
                func.attr,
                arg_types
            )

            if t is None:
                return Unknown()

            return t
    
    elif isinstance(expr, ast.Lambda):
        # attempt to infer lambda return via body
        
        param_types: List[Type] = [Unknown() for _ in expr.args.args]
        ret_type = infer_type_from_expr(expr.body, env)
        return CallableType(param_types, ret_type)
    
    if isinstance(expr, ast.Tuple):
        elems = [infer_type_from_expr(e, env) for e in expr.elts]
        return TupleType(elems)
    if isinstance(expr, ast.Dict):
        k = Unknown()
        v = Unknown()
        for key, val in zip(expr.keys, expr.values):
            k = k.join(infer_type_from_expr(key, env))
            v = v.join(infer_type_from_expr(val,env))
        return DictType(k,v)
    return Unknown()        

def repr_dict(d):
    return {k:repr(v) for k,v in d.items()}
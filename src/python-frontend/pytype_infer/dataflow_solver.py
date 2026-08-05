import ast
from typing import Dict, List, Any, Tuple
from .lattice import *

def parse_condition(cond):
    if isinstance(cond, ast.Call) and isinstance(cond.func, ast.Name) and cond.func.id == 'isinstance':
        if len(cond.args)>=2 and isinstance(cond.args[0], ast.Name):
            var = cond.args[0].id
            arg = cond.args[1]
            if isinstance(arg, ast.Name):
                return ('isinstance', var, arg.id)
            if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
                return ('isinstance', var, arg.value.id)
    if isinstance(cond, ast.Compare) and isinstance(cond.left, ast.Name):
        if len(cond.ops)==1 and isinstance(cond.ops[0], ast.Is):
            if len(cond.comparators)==1 and isinstance(cond.comparators[0], ast.Constant) and cond.comparators[0].value is None:
                return ('is_none', cond.left.id, False)
        if len(cond.ops)==1 and isinstance(cond.ops[0], ast.IsNot):
            if len(cond.comparators)==1 and isinstance(cond.comparators[0], ast.Constant) and cond.comparators[0].value is None:
                return ('is_none', cond.left.id, True)
    return None

def analyze_function(func_node: ast.FunctionDef, max_iters=50):
    from .cfg_builder import build_cfg_for_function
    cfg = build_cfg_for_function(func_node)
    for i, block in enumerate(cfg.blocks):
        print(f"BLOCK {i}")
        for s in block.stmts:
            print(type(s))
            print(ast.dump(s, indent=2) if isinstance(s, ast.AST) else s)
        print("pred =", block.pred)
        print("succ = ", block.succ)
            
    n = len(cfg.blocks)
    in_envs = [dict() for _ in range(n)]
    out_envs = [dict() for _ in range(n)]
    entry_env = {}
    for arg in func_node.args.args:
        entry_env[arg.arg] = Unknown()
    if n>0:
        in_envs[0] = entry_env.copy()
    changed = True
    iteration = 0
    while changed and iteration < max_iters:
        changed = False
        iteration += 1
        for i, block in enumerate(cfg.blocks):
            if block.pred:
                pred_env = {}
                for p in block.pred:
                    pred = out_envs[p]
                    for k,v in pred.items():
                        if k in pred_env:
                            pred_env[k] = pred_env[k].join(v)
                        else:
                            pred_env[k] = v
            else:
                pred_env = in_envs[i] if in_envs[i] else {}
            in_envs[i] = pred_env.copy()
            env = pred_env.copy()
            cond = None
            if block.stmts and isinstance(block.stmts[0], tuple) and block.stmts[0][0]=='cond':
                cond = block.stmts[0][1]
                stmts = block.stmts[1:]
            else:
                stmts = block.stmts
            for stmt in stmts:
                if isinstance(stmt, ast.Assign):
                    print(ast.dump(stmt, indent=2))
                    targets = stmt.targets
                    if len(targets)>=1 and isinstance(targets[0], ast.Name):
                        name = targets[0].id
                        val = stmt.value
                        t = infer_type_from_expr(val, env)
                        if t is None:
                           t = Unknown()
                        if name in env:
                            env[name] = env[name].join(t)
                        else:
                            env[name] = t
                    elif (len(targets) >=1 and isinstance(targets[0], ast.Subscript) and isinstance(targets[0].value, ast.Name)):
                        dict_name = targets[0].value.id
                        key_type = infer_type_from_expr(targets[0].slice, env)
                        value_type = infer_type_from_expr(stmt.value, env)
                        cur = env.get(dict_name)
                        if isinstance(cur, DictType):
                            env[dict_name] = DictType(cur.key_t.join(key_type), cur.val_t.join(value_type))
                        elif isinstance(cur, Unknown):
                            if isinstance(key_type, StrType):
                                env[dict_name] = DictType(key_type, value_type)
                            else:
                                env[dict_name] = Unknown()
                        else:
                            env[dict_name] = Unknown()                
                elif isinstance(stmt, ast.AnnAssign):
                    target = stmt.target
                    if isinstance(target, ast.Name):
                        name = target.id
                        ann = stmt.annotation
                        if isinstance(ann, ast.Name):
                            t = mk_type_from_name(ann.id)
                        else:
                            t = Unknown()
                        env[name] = t
                elif isinstance(stmt, ast.Expr):
                    expr = stmt.value if hasattr(stmt,'value') else stmt
                    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
                        if isinstance(expr.func.value, ast.Name) and expr.func.attr == 'append' and len(expr.args)>=1:
                            listname = expr.func.value.id
                            arg = expr.args[0]
                            at = infer_type_from_expr(arg, env)
                            cur = env.get(listname)

                            if cur is None:
                                env[listname] = ListType(at)

                            elif isinstance(cur, ListType):

                                if isinstance(cur.elem, Unknown):

                                    env[listname] = ListType(at)

                                else:

                                    env[listname] = ListType(cur.elem.join(at))

                            elif isinstance(cur, Unknown):

                                env[listname] = ListType(at)

                            else:

                                env[listname] = ListType(at)                

            prev = out_envs[i]
            merged = prev.copy()
            for k,v in env.items():
                if k in merged:
                    merged[k] = merged[k].join(v)
                else:
                    merged[k] = v
            if repr_dict(merged) != repr_dict(prev):
                out_envs[i] = merged
                changed = True
            # propagate to successors with condition narrowing
            if cond and block.succ:
                parsed = parse_condition(cond)
                if parsed and len(block.succ)>=2:
                    succ_true = block.succ[0]
                    succ_false = block.succ[1]
                    t_env = merged.copy()
                    f_env = merged.copy()
                    kind = parsed[0]
                    if kind == 'isinstance':
                        var, tname = parsed[1], parsed[2]
                        if var in t_env:
                            t_env[var] = t_env[var].narrow_with_isinstance(tname)
                        if var in f_env:
                            f_env[var] = f_env[var].remove_isinstance(tname)
                    elif kind == 'is_none':
                        var, is_not = parsed[1], parsed[2]
                        if is_not:
                            if var in t_env:
                                t_env[var] = t_env[var].remove_isinstance('None')
                            if var in f_env:
                                f_env[var] = f_env[var].narrow_with_isinstance('None')
                        else:
                            if var in t_env:
                                t_env[var] = t_env[var].narrow_with_isinstance('None')
                            if var in f_env:
                                f_env[var] = f_env[var].remove_isinstance('None')
                    # merge into successor in_envs
                    succ_prev = in_envs[succ_true]
                    succ_new = succ_prev.copy()
                    for k,v in t_env.items():
                        if k in succ_new:
                            succ_new[k] = succ_new[k].join(v)
                        else:
                            succ_new[k] = v
                    in_envs[succ_true] = succ_new
                    succ_prev = in_envs[succ_false]
                    succ_new = succ_prev.copy()
                    for k,v in f_env.items():
                        if k in succ_new:
                            succ_new[k] = succ_new[k].join(v)
                        else:
                            succ_new[k] = v
                    in_envs[succ_false] = succ_new
                else:
                    for s in block.succ:
                        succ_prev = in_envs[s]
                        succ_new = succ_prev.copy()
                        for k,v in merged.items():
                            if k in succ_new:
                                succ_new[k] = succ_new[k].join(v)
                            else:
                                succ_new[k] = v
                        in_envs[s] = succ_new
            else:
                for s in block.succ:
                    succ_prev = in_envs[s]
                    succ_new = succ_prev.copy()
                    for k,v in merged.items():
                        if k in succ_new:
                            succ_new[k] = succ_new[k].join(v)
                        else:
                            succ_new[k] = v
                    in_envs[s] = succ_new
    return cfg, in_envs, out_envs

def merge_types(old_t, new_type, use_widen=False):
    if old_t is None:
        return new_type
    return old_t.widen(new_type) if use_widen else new_type    


def resolve_method_call(onj_type, method, arg_types):
    if isinstance(obj_type, ListType):
        if method == "append":
            return NoneType()
        if method == "pop":
            return obj_type.elem
        if method == "copy":
            return obj_type
        if method == "extend":
            if arg_types:
                return ListType(obj_type.elem.join(arg_types[0]))
            return obj_type

    if isinstance(obj_type, StrType):
        if method in ["lower", "upper", "strip", "replace"]:
            return StrType()
        if method in ["startswith", "endswith", "isdigit", "isalpha"]:
            return BoolType()
        if method in ["find", "rfind", "count"]:
            return IntType()
        if method == "split":
            return ListType(StrType())

    if isinstance(obj_type, StrType):
        if method in ["lower", "upper", "strip", "replace"]:
            return StrType()
        if method in ["startswith", "endswith", "isdigit", "isalpha"]:
            return BoolType()
        if method in ["find", "rfind", "count"]:
            return IntType()
        if method == "split":
            return ListType(StrType())

    if isinstance(obj_type, DictType):
        if method in ["get", "pop"]:
            return obj_type.v 
        if method == "keys":
            return ListType(obj_type.k)

    return Unknown()         
    
def resolve_method_call(obj_type, method, arg_types):
    """
    Infer the return type of a Python method call.
    Always returns a lattice Type.
    """

    #
    # ---------- STRINGS ----------
    #

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

        if method == "encode":
            return BytesType() if "BytesType" in globals() else Unknown()

        return Unknown()

    #
    # ---------- LISTS ----------
    #

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

    #
    # ---------- TUPLES ----------
    #

    if isinstance(obj_type, TupleType):

        if method in {
            "count",
            "index",
        }:
            return IntType()

        return Unknown()

    #
    # ---------- DICTIONARIES ----------
    #

    if isinstance(obj_type, DictType):

        if method == "copy":
            return DictType(
                obj_type.key,
                obj_type.value
            )

        if method == "get":
            return obj_type.value

        if method == "pop":
            return obj_type.value

        if method == "popitem":
            return TupleType([
                obj_type.key,
                obj_type.value
            ])

        if method == "keys":
            return ListType(obj_type.key)

        if method == "values":
            return ListType(obj_type.value)

        if method == "items":
            return ListType(
                TupleType([
                    obj_type.key,
                    obj_type.value
                ])
            )

        if method in {
            "update",
            "clear",
            "setdefault",
        }:
            return NoneType()

        return Unknown()

    #
    # ---------- SETS ----------
    #

    if isinstance(obj_type, SetType):

        if method == "copy":
            return SetType(obj_type.elem)

        if method == "pop":
            return obj_type.elem

        if method in {
            "union",
            "intersection",
            "difference",
            "symmetric_difference",
        }:
            return SetType(obj_type.elem)

        if method in {
            "add",
            "clear",
            "discard",
            "remove",
            "update",
            "intersection_update",
            "difference_update",
            "symmetric_difference_update",
        }:
            return NoneType()

        if method in {
            "issubset",
            "issuperset",
            "isdisjoint",
        }:
            return BoolType()

        return Unknown()

    #
    # ---------- CALLABLES ----------
    #

    if isinstance(obj_type, CallableType):

        return obj_type.ret

    #
    # ---------- FALLBACK ----------
    #

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
            if f_name == "set":
                 return SetType(Unknown())

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

    return Unknown()
    
    if isinstance(expr, ast.Lambda):
        # attempt to infer lambda return via body
        
        param_types: List[Type] = [Unknown() for _ in expr.args.args]
        ret_type = infer_type_from_expr(expr.body, env)
        return CallableType(param_types, ret_type)
    return Unknown()
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

def repr_dict(d):
    return {k:repr(v) for k,v in d.items()}

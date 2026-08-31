import ast
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from .lattice import *
from .cfg_builder import *

@dataclass
class InferenceContext:
    
    known_classes: dict[str,Any]

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

def analyze_function(func_node: ast.FunctionDef, context: InferenceContext, max_iters: int =50):
    cfg = build_cfg_for_function(func_node)
    in_envs, out_envs = initialize_environments(func_node, cfg)

    return_types = {}

    changed = True
    iteration = 0

    while changed and iteration < max_iters:
        changed = False
        iteration +=1

        for i, block in enumerate(cfg.blocks):
            env, cond, stmts = prepare_block(block, in_envs, i)

            for stmt in stmts:
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    return_types[id(stmt)] = infer_return_type(
                        stmt,
                        env,
                        context,
                    )
                env = transfer_statement(stmt, env, context)

            merged = merge_environment(out_envs[i], env)

            if merged != out_envs[i]:
                out_envs[i] = merged
                changed = True

            propagate_block(block,merged, cond, in_envs,)

    return cfg, in_envs, out_envs, return_types  

def infer_return_type(stmt, env, context):
    if stmt.value is None:
        return None

    return infer_type_from_expr(stmt.value, env, context)

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

    entry_env = {}

    for arg in func_node.args.args:
        if arg.annotation is not None:
            entry_env[arg.arg] = infer_annotation_type(arg.annotation)
        else:
            entry_env[arg.arg] = Unknown()              

    if n > 0:
        in_envs[0] = entry_env.copy()


    return in_envs, out_envs

# def compute_input_envirment(block, block_index, in_envs, out_envs):
#     if not block.pred:
#         return in_envs[block_index].copy()

#     pred_env = []

#     for predecessor in block.pred:
#         merge_into_environment(
#             pred_env,
#             out_envs[predecessor],
#         )

#     return pred_env

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

def transfer_statement(stmt, env, context):
    if isinstance(stmt, ast.Assign):
        return transfer_assign(stmt, env, context)

    if isinstance(stmt, ast.AnnAssign):
        return transfer_annassign(stmt, env)

    if isinstance(stmt, ast.Expr):
        return transfer_expr(stmt, env, context)

    #if isinstance(stmt, ast.Return):
     #   return transfer_return(stmt, env, context)

    return env


def transfer_return(stmt, env, context):
    if stmt.value is None:
        return env

    return_type = infer_type_from_expr(stmt.value, env, context,)

    return env, return_type 

def transfer_assign(stmt, env, context):
    for target in stmt.targets:
        if isinstance(target, ast.Name):
            assign_name(target, stmt.value, env, context)

        elif isinstance(target, ast.Subscript):
            assign_subscript(target, stmt.value, env, context)

    return env 

def assign_name(target, value, env, context):
    name = target.id
    typ = infer_type_from_expr(value, env, context)

    if typ is None:
        typ = Unknown()

    if name in env:
        env[name] = env[name].join(typ)
    else:
        env[name] = typ 

def assign_subscript(target, value, env, context):
    if not isinstance(target.value, ast.Name):
        return

    container = target.value.id
    key_type = infer_type_from_expr(target.slice,env, context) 
    value_type = infer_type_from_expr(value, env, context)

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

def transfer_expr(stmt, env, context):
    expr = stmt.value

    if isinstance(expr, ast.Call):
        handle_call_effect(expr, env, context, )

    return env    

def handle_call_effect(call, env, context):
    if not isinstance(call.func, ast.Attribute):
        return

    if call.func.attr == "append":
        handle_append(call, env, context)

    elif call.func.attr == "setDefault":
        handle_setdfault(call, env, context)

def handle_setdfault(call, env, context):
    if not isinstance(call.func, ast.Attribute):
        return

    if call.func.attr != "setdefault":
        return

    if not isinstance(call.func.value, ast.Name):
        return

    if len(call.args) < 2:
        return

    dict_name = call.func.value.id

    key_type = infer_type_from_expr(call.args[0], env, context,)

    default_type = infer_type_from_expr(call.args[1], env, context,)

    current = env.get(dict_name, Unknown(),)

    if isinstance(current, DictType):
        env[dict_name] = DictType(current.key_t.join(key_type),
                                  current.val_t.join(default_type),)
        return 

    env[dict_name] = DictType(key_type, default_type,)

def handle_append(call, env, context):
    if not call.args:
        return

    receiver = call.func.value

    arg_type = infer_type_from_expr(
        call.args[0],
        env,
        context,
    )

    if isinstance(receiver, ast.Name):
        listname = receiver.id
        current = env.get(listname, Unknown())

        if isinstance(current, ListType):
            env[listname] = ListType(
                current.elem.join(arg_type)
            )
        else:
            env[listname] = ListType(arg_type)

        return

    if isinstance(receiver, ast.Attribute):
        receiver_type = infer_type_from_expr(
            receiver,
            env,
            context,
        )

        if isinstance(receiver_type, ListType):
            return

        return

    if isinstance(receiver, ast.Call):
        handle_nested_append(
            receiver,
            arg_type,
            env,
            context,
        )

def handle_nested_append(call, arg_type, env, context):
    if not isinstance(call.func, ast.Attribute):
        return

    if call.func.attr != "setdefault":
        return

    receiver = call.func.value

    if not isinstance(receiver, ast.Name):
        return

    dictname = receiver.id

    if len(call.args) < 2:
        return

    key_type = infer_type_from_expr(
        call.args[0],
        env,
        context,
    )

    default_type = infer_type_from_expr(
        call.args[1],
        env,
        context,
    )

    if not isinstance(default_type, ListType):
        return



    current = env.get(dictname, Unknown())

    if isinstance(current, DictType):
        # The default supplied to setdefault is the
        # value type that must be mutated by append().
        value_type = current.val_t

        if isinstance(value_type, ListType):
            updated_value = ListType(
                value_type.elem.join(arg_type)
            )
        # elif isinstance(default_type, ListType):
        #     updated_value = ListType(
        #         default_type.elem.join(arg_type)
        #     )
        else:
            updated_value = default_type

        env[dictname] = DictType(
            current.key_t.join(key_type),
            value_type.join(updated_value),
        )
        return

    if isinstance(default_type, ListType):
        env[dictname] = DictType(
            key_type,
            ListType(
                default_type.elem.join(arg_type)
            ),
        )

def merge_environment(previous, current):
    merged= previous.copy()
    merge_into_environment(merged, current)
    return merged

def merge_types(old_t, new_type, use_widen=False):
    if old_t is None:
        return new_type

    if use_widen:
        return old_t.widen(new_type)
    
    return old_t.join(new_type) 

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
                                                                          
def infer_class_call_type(func_name, known_classes):
    if func_name in known_classes:
        return InstanceType(func_name)

    return None

def infer_instance_attribute_type(
    obj_type,
    attr_name,
    known_classes,
):
    class_info = known_classes.get(obj_type.class_name)

    if class_info is None:
        return Unknown()

    return class_info.attributes.get(
        attr_name,
        Unknown(),
    )

def infer_dict_attribute_type(obj_type, attr_name):
    if attr_name == "copy":
        return CallableType([], DictType(obj_type.key_t, obj_type.val_t))

    if attr_name == "get":
        return CallableType([], obj_type.val_t)

    if attr_name == "pop":
        return CallableType([], obj_type.val_t)

    if attr_name == "popitem":
        return CallableType([], TupleType([obj_type.key_t, obj_type.val_t]),)

    if attr_name == "keys":
        return CallableType([], ListType(obj_type.key_t),)

    if attr_name == "values":
        return CallableType([], ListType(obj_type.val_t))

    if attr_name == "items":
        return CallableType([], ListType(TupleType([obj_type.key_t, obj_type.val_t])),)
    
    if attr_name in {
        "update",
        "clear",
    }:
        return CallableType(
            [],
            NoneType(),
        )

    if attr_name == "setdefault":
        return CallableType(
            [],
            UnionType([
                obj_type.val_t,
                NoneType(),
            ]),
        )

    return Unknown()
    
def infer_constant_type(expr):
    value = expr.value

    if isinstance(value, bool):
        return BoolType()

    if isinstance(value, int):
        return IntType()

    if isinstance(value, float):
        return FloatType()

    if isinstance(value, complex):
        return ComplexType()

    if value is None:
        return NoneType()

    if isinstance(value, str):
        return StrType()

    return Unknown()

def infer_list_literal_type(expr, env, context):
    elem_type = Unknown()

    for element in expr.elts:
        elem_type = elem_type.join(infer_type_from_expr(element, env, context))

    return ListType(elem_type)

def infer_tuple_literal_type(expr, env, context):
    elems = [
        infer_type_from_expr(element, env, context)
        for element in expr.elts
    ]    
    return TupleType(elems)

def infer_dict_literal_type(expr, env, context):
    key_type = Unknown()
    value_type = Unknown()

    for key, value in zip(expr.keys, expr.values):
        if key is None:
            continue

        key_type = key_type.join(
            infer_type_from_expr(key, env, context)
        )

        value_type = value_type.join(
            infer_type_from_expr(value, env, context)
        )

    return DictType(
        key_type,
        value_type,
    )

def infer_binop_type(expr, env, context):
    left = infer_type_from_expr(expr.left, env, context)
    right = infer_type_from_expr(expr.right, env, context)

    print(
        "[PYTYPE BINOP]",
        ast.dump(expr),
        "LEFT =", left,
        "RIGHT =", right,
        "OP =", type(expr.op).__name__,
    )

    if isinstance(expr.op, ast.Add):
        result = infer_add_type(left, right)

        if result is not None:
            return result

    if isinstance(left, FloatType) or isinstance(right, FloatType):
        return FloatType()

    if isinstance(left, IntType) and isinstance(right, IntType):
        return IntType()

    return Unknown()

def infer_add_type(left, right):
    if isinstance(left, ListType) and isinstance(right, ListType):
        return ListType(left.elem.join(right.elem))

    if isinstance(left, StrType) and isinstance(right, StrType):
        return StrType()

    if isinstance(left, TupleType) and isinstance(right, TupleType):
        return TupleType(left.elems + right.elems)

    return None

def infer_call_type(expr, env, context):
    func = expr.func

    if isinstance(func, ast.Name):
        return infer_named_call_type(func.id, expr, env, context, )

    if isinstance(func, ast.Attribute):
        if (isinstance(func.value, ast.Name) and func.value.id == "cmath"):
            return infer_cmath_call_type(func.attr, expr, env, context)
        obj_type = infer_type_from_expr(func.value, env, context, )

        arg_types = [infer_type_from_expr(arg, env, context) for arg in expr.args]

        attribute_type = infer_attribute_type(
            obj_type,
            func.attr,
            context.known_classes,
        )

        if isinstance(attribute_type, CallableType):
                return attribute_type.ret

        return resolve_method_call(obj_type, func.attr, arg_types)

    return Unknown()

def infer_named_call_type(
    func_name,
    expr,
    env,
    context,
):
    builtin_type = infer_builtin_call_type(
        func_name,
        expr,
        env,
        context
    )

    if builtin_type is not None:
        return builtin_type

    class_type = infer_class_call_type(
        func_name,
        context.known_classes,
    )

    if class_type is not None:
        return class_type

    return Unknown()

def infer_builtin_call_type(name, expr, env, context):
    if name == "len":
        return IntType()

    if name == "str":
        return StrType()

    if name == "int":
        return IntType()

    if name == "bool":
        return BoolType()

    if name == "float":
        return FloatType()

    if name == "list":
        return ListType(Unknown())

    if name == "dict":
        return DictType(Unknown(), Unknown())

    if name == "tuple":
        return TupleType([])

    if name == "complex":
        return ComplexType()

    if name == "set":
        return SetType(Unknown())

    return None

def infer_lambda_type(expr, env, context):
    param_types: List[Type] =[
        Unknown()
        for _ in expr.args.args
    ]

    ret_type = infer_type_from_expr(expr.body, env, context)

    return CallableType(param_types, ret_type)

def infer_cmath_call_type(
    func_name,
    expr,
    env,
    context,
):
    if func_name in {
        "acos",
        "asin",
        "atan",
        "acosh",
        "asinh",
        "atanh",
        "cos",
        "sin",
        "tan",
        "cosh",
        "sinh",
        "tanh",
        "exp",
        "log",
        "log10",
        "sqrt",
    }:
        return ComplexType()

    return Unknown()

def infer_subscript_type(expr, env, context):
    container_type = infer_type_from_expr(expr.value, env, context)

    index_type = infer_type_from_expr(
        expr.slice,
        env,
        context,
    )

    result = infer_container_subscript_type(container_type, expr.slice, env, context)

    print(
        "[PYTYPE SUBSCRIPT]",
        ast.unparse(expr),
        "container=",
        container_type,
        "index=",
        index_type,
        "result=",
        result,
    )
    return result

def infer_list_attribute_type(obj_type, attr_name):
    if attr_name == "copy":
        return CallableType(
            [],
            ListType(obj_type.elem),
        )

    if attr_name == "pop":
        return CallableType(
            [],
            obj_type.elem,
        )

    if attr_name in {
        "append",
        "extend",
        "insert",
        "remove",
        "clear",
        "reverse",
        "sort",
    }:
        return CallableType(
            [],
            NoneType(),
        )

    if attr_name in {
        "count",
        "index",
    }:
        return CallableType(
            [],
            IntType(),
        )

    return Unknown()

def infer_tuple_attribute_type(obj_type, attr_name):
    if attr_name in {
        "count",
        "index",
    }:
        return CallableType(
            [],
            IntType(),
        )

    return Unknown()

def infer_str_attribute_type(obj_type, attr_name):
    if attr_name in {
        "upper",
        "lower",
        "capitalize",
        "title",
        "casefold",
        "swapcase",
        "strip",
        "lstrip",
        "rstrip",
        "replace",
        "removeprefix",
        "removesuffix",
        "expandtabs",
        "center",
        "ljust",
        "rjust",
        "zfill",
        "join",
        "translate",
    }:
        return CallableType(
            [],
            StrType(),
        )

    if attr_name in {
        "split",
        "rsplit",
        "splitlines",
    }:
        return CallableType(
            [],
            ListType(StrType()),
        )

    if attr_name in {
        "partition",
        "rpartition",
    }:
        return CallableType(
            [],
            TupleType([
                StrType(),
                StrType(),
                StrType(),
            ]),
        )

    if attr_name in {
        "find",
        "rfind",
        "index",
        "rindex",
        "count",
    }:
        return CallableType(
            [],
            IntType(),
        )

    if attr_name in {
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
        return CallableType(
            [],
            BoolType(),
        )

    return Unknown()

def infer_set_attribute_type(obj_type, attr_name):
    if attr_name == "copy":
        return CallableType(
            [],
            SetType(obj_type.elem),
        )

    if attr_name == "pop":
        return CallableType(
            [],
            obj_type.elem,
        )

    if attr_name in {
        "union",
        "intersection",
        "difference",
        "symmetric_difference",
    }:
        return CallableType(
            [],
            SetType(obj_type.elem),
        )

    if attr_name in {
        "add",
        "clear",
        "discard",
        "remove",
        "update",
        "intersection_update",
        "difference_update",
        "symmetric_difference_update",
    }:
        return CallableType(
            [],
            NoneType(),
        )

    if attr_name in {
        "issubset",
        "issuperset",
        "isdisjoint",
    }:
        return CallableType(
            [],
            BoolType(),
        )

    return Unknown()

def infer_tuple_subscript_type(
    tuple_type,
    index_expr,
    env,
):
    if not isinstance(tuple_type, TupleType):
        return Unknown()

    if isinstance(index_expr, ast.Constant):
        index = index_expr.value

        if isinstance(index, int) and not isinstance(index, bool):
            if -len(tuple_type.elems) <= index < len(tuple_type.elems):
                return tuple_type.elems[index]

            return Unknown()

    if tuple_type.elems:
        result = Unknown()

        for elem in tuple_type.elems:
            result = result.join(elem)

        return result

    return Unknown()

def infer_unary_op_type(expr, env, context):
    operand_type = infer_type_from_expr(
        expr.operand,
        env,
        context,
    )

    if isinstance(expr.op, (ast.UAdd, ast.USub)):
        if isinstance(operand_type, IntType):
            return IntType()

        if isinstance(operand_type, FloatType):
            return FloatType()

    if isinstance(expr.op, ast.Not):
        return BoolType()

    if isinstance(expr.op, ast.Not):
        return BoolType()

    if isinstance(expr.op, ast.Invert):
        if isinstance(operand_type, IntType):
            return IntType()

    return Unknown()

def infer_container_subscript_type(container_type, index_expr, env, context):
    if isinstance(container_type, ListType):
        index_type = infer_type_from_expr(index_expr, env, context)

        if isinstance(index_type, IntType):
            return container_type.elem

        if isinstance(index_expr, ast.Slice):
            return ListType(container_type.elem)

        return Unknown()

    if isinstance(container_type, DictType):
        key_type = infer_type_from_expr(index_expr, env, context)
            
        if not is_type_compatible(key_type, container_type.key_t):
            return Unknown()

        return container_type.val_t

    if isinstance(container_type, TupleType):
        return infer_tuple_subscript_type(
            container_type,
            index_expr,
            env,
        )
    if isinstance(container_type, StrType):
        if isinstance(index_expr, ast.Slice):
            return StrType()
        
        return StrType()

    if isinstance(container_type, SetType):
        return Unknown()

    return Unknown()

def is_type_compatible(actual, expected):
    if isinstance(actual, Unknown) or isinstance(expected, Unknown):
        return Type

    if type(actual) is type(expected):
        return True

    if isinstance(actual, IntType) and isinstance(expected, FloatType):
        return True

    if isinstance(actual, BoolType) and isinstance(expected, IntType):
        return True

    return False

def infer_attribute_type(
    obj_type,
    attr_name,
    known_classes,
):
    if isinstance(obj_type, DictType):
        return infer_dict_attribute_type(
            obj_type,
            attr_name,
        )

    if isinstance(obj_type, ListType):
        return infer_list_attribute_type(
            obj_type,
            attr_name,
        )

    if isinstance(obj_type, TupleType):
        return infer_tuple_attribute_type(
            obj_type,
            attr_name,
        )

    if isinstance(obj_type, StrType):
        return infer_str_attribute_type(
            obj_type,
            attr_name,
        )

    if isinstance(obj_type, SetType):
        return infer_set_attribute_type(
            obj_type,
            attr_name,
        )
    if isinstance(obj_type, ComplexType):
        if attr_name in {"real", "imag"}:
            print(
            "COMPLEX ATTRIBUTE:",
            obj_type,
            attr_name,
            "-> FloatType"
            )
            return FloatType()

    if isinstance(obj_type, InstanceType):
        return infer_instance_attribute_type(
            obj_type,
            attr_name,
            known_classes,
        )

    return Unknown()

# def known_classes(tree):
#     classes = {}

#     for node in ast.walk(tree):
#         if isinstance(node, ast.ClassDef):
#             class_name = node.name
#             attributes = {}

#             for stmt in node.body:
#                 if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
#                     attr_name = stmt.target.id
#                     attr_type = mk_type_from_annotation(stmt.annotation)
#                     attributes[attr_name] = attr_type

#             classes[class_name] = ClassInfo(class_name, attributes)

#     return classes
def build_class_info(class_node, known_classes):
    context = InferenceContext(known_classes=known_classes)

    attributes = {}
    bases = []

    for base in class_node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)

    for stmt in class_node.body:

        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name):
                attributes[stmt.target.id] = (infer_annotation_type(stmt.annotation))

        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    attributes[target.id] = (
                        infer_type_from_expr(
                            stmt.value,
                            {},
                            context,
                        )
                    )

        elif isinstance(stmt, ast.FunctionDef):
            collect_instance_attributes(
                stmt,
                attributes,
                context,
            )

    return ClassInfo(
        name=class_node.name,
        bases=bases,
        attributes=attributes,
    )

def collect_known_classes(tree)-> dict[str, Any]:
    known_classes = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            known_classes[node.name] = ClassInfo(name=node.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            known_classes[node.name] = build_class_info(node, known_classes)

    return known_classes

def collect_instance_attributes(
    function_node,
    attributes,
    context,
):
    for node in ast.walk(function_node):

        if isinstance(node, ast.AnnAssign):
            target = node.target

            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                continue

            attributes[target.attr] = infer_annotation_type(
                node.annotation
            )
            continue

        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue

            target = node.targets[0]

            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                continue

            attributes[target.attr] = infer_type_from_expr(
                node.value,
                {},
                context,
            )
        
            

def infer_annotation_type(annotation):
    if isinstance(annotation, ast.Name):
        return mk_type_from_name(annotation.id)

    if isinstance(annotation, ast.Constant):
        if isinstance(annotation.value, str):
            return mk_type_from_name(annotation.value)

        return Unknown()

    if isinstance(annotation, ast.Subscript):
        if not isinstance(annotation.value, ast.Name):
            return Unknown()

        name = annotation.value.id
        slice_node = annotation.slice

        if name == "list":
            return ListType(
                infer_annotation_type(slice_node)
            )

        if name == "set": 
            return SetType(infer_annotation_type(slice_node))

        if name == "dict":
            if isinstance(slice_node, ast.Tuple):
                if len(slice_node.elts) == 2:
                    return DictType(infer_annotation_type(slice_node.elts[0]), infer_annotation_type(slice_node.elts[1]),)

            return DictType(Unknown(), Unknown())

        if name == "tuple":
            if isinstance(slice_node, ast.Tuple):
                return TupleType([infer_annotation_type(element) for element in slice_node.elts])

            return TupleType([infer_annotation_type(slice_node)])
    return Unknown()        
def infer_type_from_expr(expr, env, context):
    if isinstance(expr, ast.Constant):
        return infer_constant_type(expr)

    if isinstance(expr, ast.Name):
        return env.get(expr.id, Unknown())

    if isinstance(expr, ast.Attribute):
        obj_type = infer_type_from_expr(
            expr.value,
            env,
            context
        )
        if isinstance(obj_type, ComplexType):
            if expr.attr == "real":
               return FloatType()

        if expr.attr == "imag":
            return FloatType()
        return infer_attribute_type(
            obj_type,
            expr.attr,
            context.known_classes,
        )

    if isinstance(expr, ast.List):
        return infer_list_literal_type(expr, env, context)

    if isinstance(expr, ast.Tuple):
        return infer_tuple_literal_type(expr, env, context)

    if isinstance(expr, ast.Dict):
        return infer_dict_literal_type(expr, env, context)

    if isinstance(expr, ast.UnaryOp):
        return infer_unary_op_type(expr, env, context)

    if isinstance(expr, ast.BinOp):
        return infer_binop_type(expr, env, context)

    if isinstance(expr, ast.Subscript):
        return infer_subscript_type(expr, env, context)

    if isinstance(expr, ast.Call):
        return infer_call_type(expr, env, context,)

    if isinstance(expr, ast.Lambda):
        return infer_lambda_type(expr, env, context)

    return Unknown()

def repr_dict(d):
    return {k:repr(v) for k,v in d.items()}
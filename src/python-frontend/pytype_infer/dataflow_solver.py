import ast
from typing import Dict, List, Any
from dataclasses import dataclass, field
from .lattice import DictType, ListType, BoolType, StrType, TupleType, Unknown, NoneType, Type
from .lattice import *
from .cfg_builder import *


@dataclass
class InferenceContext:

    known_classes: dict[str, Any]
    function_returns: Dict[str, Type] = field(default_factory=dict)
    function_params: Dict[str, list[Type]] = field(default_factory=dict)
    function_param_observations: dict[
        str,
        list[list[Type]],
    ] = field(default_factory=dict)
    function_keys: Dict[int, str] = field(
        default_factory=dict
    )  #Maps an AST function node to the name used by the interprocedural analysis
    function_owners: Dict[int, str] = field(
        default_factory=dict)  #Maps method nodes to their containg class


def parse_isinstance_condition(cond):
    if not (isinstance(cond, ast.Call) and isinstance(cond.func, ast.Name) and cond.func.id
            == "isinstance" and len(cond.args) >= 2 and isinstance(cond.args[0], ast.Name)):
        return None
    var = cond.args[0].id
    type_expr = cond.args[1]

    if isinstance(type_expr, ast.Name):
        return ("isinstance", var, type_expr.id)

    if (isinstance(type_expr, ast.Subscript) and isinstance(type_expr.value, ast.Name)):

        return ("isinstance", var, type_expr.value.id)

    return None


def parse_none_condition(cond):
    if not isinstance(cond, ast.Compare):
        return None

    if not isinstance(cond.left, ast.Name):
        return None

    comparator = cond.comparators[0]

    if not (isinstance(comparator, ast.Constant) and comparator.value is None):
        return None

    if isinstance(cond.ops[0], ast.Is):
        return ("is_none", cond.left.id, False)

    if isinstance(cond.ops[0], ast.IsNot):
        return ("is_none", cond.left.id, True)

    return None


def function_key(func_node, context) -> str:

    return context.function_keys.get(id(func_node), func_node.name)


def function_owner(
    func_node,
    context,
):
    return context.function_owners.get(id(func_node))


def resolve_method_key(
    class_name,
    method_name,
    context,
):
    pending = [class_name]
    visited = set()

    while pending:
        current_class = pending.pop()

        if current_class in visited:
            continue

        visited.add(current_class)

        candidate = (f"{current_class}."
                     f"{method_name}")

        if candidate in context.function_keys.values():
            return candidate

        class_info = context.known_classes.get(current_class)

        if class_info is None:
            continue

        pending.extend(class_info.bases)

    return None


class CallArgumentCollector(ast.NodeVisitor):

    def __init__(
        self,
        env,
        collected,
        context,
    ):
        self.env = env
        self.collected = collected
        self.context = context

    def visit_Call(self, node):
        arg_types = [infer_type_from_expr(
            arg,
            self.env,
            self.context,
        ) for arg in node.args]

        #observed_params = (self.context.function_param_observations.setdefault(function_name,

        call_key = None
        collected_types = None  #                 [[] for _ in arg_types],))

        if isinstance(
                node.func,
                ast.Name,
        ):
            name = node.func.id

            if name in self.context.known_classes:
                constructor_key = resolve_method_key(
                    name,
                    "__init__",
                    self.context,
                )

                if constructor_key is not None:
                    call_key = constructor_key

                    collected_types = [
                        InstanceType(name),
                        *arg_types,
                    ]

            elif name in (self.context.function_keys.values()):
                call_key = name
                collected_types = arg_types

            elif isinstance(
                    node.func,
                    ast.Attribute,
            ):
                receiver_type = infer_type_from_expr(
                    node.func.value,
                    self.env,
                    self.context,
                )

                if isinstance(
                        receiver_type,
                        InstanceType,
                ):
                    method_key = resolve_method_key(
                        receiver_type.class_name,
                        node.func.attr,
                        self.context,
                    )

                    if method_key is not None:
                        call_key = method_key

                        collected_types = [
                            receiver_type,
                            *arg_types,
                        ]

            if (call_key is not None and collected_types is not None):
                merge_call_argument_types(
                    self.collected,
                    call_key,
                    collected_types,
                )

                observed_params = (self.context.function_param_observations.setdefault(
                    call_key,
                    [[] for _ in collected_types],
                ))

                while (len(observed_params) < len(collected_types)):
                    observed_params.append([])

                for i, arg_type in enumerate(collected_types):
                    if isinstance(
                            arg_type,
                        (
                            Unknown,
                            Bottom,
                        ),
                    ):
                        continue

                    already_observed = any(arg_type == old_type for old_type in observed_params[i])

                    if not already_observed:
                        observed_params[i].append(arg_type)

            self.generic_visit(node)

    def visit_Assign(self, node):
        # First collect calls occurring in the RHS.
        self.visit(node.value)

        value_type = infer_type_from_expr(
            node.value,
            self.env,
            self.context,
        )

        for target in node.targets:
            bind_target_type(
                target,
                value_type,
                self.env,
            )

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit(node.value)

            value_type = infer_type_from_expr(
                node.value,
                self.env,
                self.context,
            )
        else:
            value_type = (infer_annotation_type_with_classes(
                node.annotation,
                self.context.known_classes,
            ))

        bind_target_type(
            node.target,
            value_type,
            self.env,
        )

    def visit_FunctionDef(self, node):
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_ClassDef(self, node):
        return

    def visit_Lambda(self, node):
        return


def parse_condition(cond):
    result = parse_isinstance_condition(cond)
    if result is not None:
        return result

    return parse_none_condition(cond)


def analyze_function(func_node: ast.FunctionDef | ast.AsyncFunctionDef,
                     context: InferenceContext,
                     max_iters: int = 50):
    cfg = build_cfg_for_function(func_node)
    in_envs, out_envs = initialize_environments(
        func_node,
        cfg,
        context,
    )

    return_types = {}

    changed = True
    iteration = 0

    while changed and iteration < max_iters:
        changed = False
        iteration += 1

        for i, block in enumerate(cfg.blocks):
            env, marker, stmts = prepare_block(block, in_envs, i)

            print(f"\nBLOCK {i}")
            print("IN :", in_envs[i])
            print("OUT:", out_envs[i])

            for stmt in stmts:
                print("STMT TYPE:", type(stmt))
                print("STMT:", repr(stmt))

                if isinstance(stmt, tuple):
                    print("TUPLE LENGTH:", len(stmt))
                    for j, item in enumerate(stmt):
                        print(f"  [{j}] TYPE:", type(item))
                        print(f"  [{j}] VALUE:", repr(item))

                if isinstance(stmt, ast.AST):
                    print("AST:", ast.dump(stmt))
                else:
                    print("NOT AST!", repr(stmt))
                if isinstance(stmt, ast.Return):
                    return_types[id(stmt)] = infer_return_type(
                        stmt,
                        env,
                        context,
                    )
                #print(f"  BEFORE {ast.dump(stmt)}")
                assert isinstance(env, dict), (f"BAD ENV BEFORE STATEMENT: "
                                               f"{type(env).__name__}: {env!r}")

                new_env = transfer_statement(stmt, env, context)

                assert isinstance(new_env, dict), (f"BAD ENV AFTER STATEMENT: "
                                                   f"stmt={debug_dump(stmt)} "
                                                   f"result={type(new_env).__name__}: {new_env!r}")

                env = new_env

            merged = merge_environment(out_envs[i], env)

            if merged != out_envs[i]:
                out_envs[i] = merged
                changed = True

            propagate_block(
                block,
                merged,
                marker,
                in_envs,
                context,
            )
    if changed:
        return None

    return cfg, in_envs, out_envs, return_types


def debug_dump(node) -> str:
    if isinstance(node, ast.AST):
        return ast.dump(node)
    return repr(node)


def infer_return_type(stmt, env, context):
    if stmt.value is None:
        current_type = NoneType()
    else:
        current_type = infer_type_from_expr(stmt.value, env, context)

    print(
        "[RETURN DEBUG]",
        ast.dump(stmt.value) if stmt.value is not None else "None",
        "=>",
        current_type,
    )
    return current_type


def propagate_for_marker(
    block,
    env,
    marker,
    in_envs,
    context,
):
    target = marker[1]
    iterable = marker[2]

    if len(block.succ) < 2:
        raise UnsupportedCFGError("for block requires body "
                                  "and exhaustion successors")

    body_env = env.copy()

    element_type = infer_for_element_type(
        iterable,
        env,
        context,
    )

    bind_target_type(
        target,
        element_type,
        body_env,
    )

    merge_successor_environment(
        block.succ[0],
        body_env,
        in_envs,
    )

    merge_successor_environment(
        block.succ[1],
        env,
        in_envs,
    )


def propagate_block(
    block,
    env,
    marker,
    in_envs,
    context,
):
    if marker is None:
        propagate_to_successors(
            block.succ,
            env,
            in_envs,
        )
        return

    if is_condition_marker(marker):
        propagate_condition(
            block,
            env,
            marker[1],
            in_envs,
        )
        return

    if is_for_marker(marker):
        propagate_for_marker(
            block,
            env,
            marker,
            in_envs,
            context,
        )
        return

    if is_with_marker(marker):
        propagate_with_marker(
            block,
            env,
            marker,
            in_envs,
            context,
        )
        return

    if is_async_for_marker(marker):
        propagate_async_for_marker(
            block,
            env,
            marker,
            in_envs,
            context,
        )
        return

    raise UnsupportedCFGError(f"unsupported CFG marker: {marker[0]}")


def propagate_async_for_marker(
    block,
    env,
    marker,
    in_envs,
    context,
):
    target = marker[1]
    iterable = marker[2]

    if len(block.succ) < 2:
        raise UnsupportedCFGError("async for block requires body "
                                  "and exhaustion successors")

    body_env = env.copy()

    element_type = infer_async_for_element_type(
        iterable,
        env,
        context,
    )

    bind_target_type(
        target,
        element_type,
        body_env,
    )

    merge_successor_environment(
        block.succ[0],
        body_env,
        in_envs,
    )

    merge_successor_environment(
        block.succ[1],
        env,
        in_envs,
    )


def infer_async_for_element_type(
    iterable,
    env,
    context,
):
    iterable_type = infer_type_from_expr(
        iterable,
        env,
        context,
    )

    element_type = infer_async_iterable_element_type(
        iterable_type,
        context,
    )

    if not isinstance(
            element_type,
            Unknown,
    ):
        return element_type

    return infer_async_for_element_type_from_calls(
        iterable,
        env,
        context,
    )


def is_range_call(expr):
    return (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
            and expr.func.id == "range")


def infer_iterable_element_type(iterable_type, ):

    if isinstance(iterable_type, ListType):
        return iterable_type.elem

    if isinstance(iterable_type, SetType):
        return iterable_type.elem

    if isinstance(iterable_type, DictType):
        return iterable_type.key_t

    if isinstance(iterable_type, StrType):
        return StrType()

    if isinstance(iterable_type, TupleType):
        return infer_tuple_element_type(iterable_type, )

    return Unknown()


def infer_tuple_element_type(tuple_type):
    if not tuple_type.elems:
        return Unknown()

    element_type = tuple_type.elems[0]

    for current_type in tuple_type.elems[1:]:
        element_type = element_type.join(current_type)

    return element_type


def infer_for_element_type(
    iterable,
    env,
    context,
):
    if is_range_call(iterable):
        return IntType()

    iterable_type = infer_type_from_expr(
        iterable,
        env,
        context,
    )

    element_type = infer_iterable_element_type(iterable_type, )

    if isinstance(element_type, Unknown):
        return TopType()

    return element_type


def infer_async_for_element_type_from_calls(
    iterable,
    env,
    context,
):
    async_iter_call = ast.Call(
        func=ast.Attribute(
            value=iterable,
            attr="__aiter__",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[],
    )

    async_next_call = ast.Call(
        func=ast.Attribute(
            value=async_iter_call,
            attr="__anext__",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[],
    )

    element_type = infer_type_from_expr(
        async_next_call,
        env,
        context,
    )

    if isinstance(
            element_type,
            Unknown,
    ):
        return TopType()

    return element_type


def register_function_context(
    tree,
    context,
):
    for stmt in tree.body:
        if isinstance(
                stmt,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            context.function_keys[id(stmt)] = stmt.name

        elif isinstance(stmt, ast.ClassDef):
            for member in stmt.body:
                if not isinstance(
                        member,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    continue

                key = f"{stmt.name}.{member.name}"

                context.function_keys[id(member)] = key
                context.function_owners[id(member)] = stmt.name


def infer_method_type(
    method_node,
    known_classes,
):
    param_types = []

    args = method_node.args.args

    for arg in args[1:]:
        if arg.annotation is None:
            param_types.append(Unknown())
        else:
            param_types.append(infer_annotation_type_with_classes(
                arg.annotation,
                known_classes,
            ))

    if method_node.returns is None:
        return_type = Unknown()
    else:
        return_type = infer_annotation_type_with_classes(
            method_node.returns,
            known_classes,
        )

    return CallableType(
        param_types,
        return_type,
    )


def infer_annotation_type_with_classes(
    annotation,
    known_classes,
):

    if isinstance(annotation, ast.Name):
        if annotation.id in known_classes:
            return InstanceType(annotation.id)

    if (isinstance(annotation, ast.Constant) and isinstance(annotation, ast.Constant)
            and isinstance(annotation.value, str) and annotation.value in known_classes):
        return InstanceType(annotation.value)

    return infer_annotation_type(annotation)


def infer_async_iterable_element_type(
    iterable_type,
    context,
):
    if not isinstance(
            iterable_type,
            InstanceType,
    ):
        return Unknown()

    class_info = context.known_classes.get(iterable_type.class_name)

    if class_info is None:
        return Unknown()

    async_iter_type = class_info.attributes.get(
        "__aiter__",
        Unknown(),
    )

    if isinstance(
            async_iter_type,
            CallableType,
    ):
        iterator_type = async_iter_type.ret

        if isinstance(
                iterator_type,
                InstanceType,
        ):
            element_type = (infer_async_instance_element_type(
                iterator_type,
                context,
            ))

            if not isinstance(
                    element_type,
                    Unknown,
            ):
                return element_type

    return infer_async_instance_element_type(
        iterable_type,
        context,
    )


def infer_async_instance_element_type(
    iterable_type,
    context,
):
    class_info = context.known_classes.get(iterable_type.class_name)

    if class_info is None:
        return Unknown()

    async_next_type = class_info.attributes.get(
        "__anext__",
        Unknown(),
    )

    if isinstance(
            async_next_type,
            CallableType,
    ):
        return async_next_type.ret

    return Unknown()


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
        true_env[var] = (true_env[var].narrow_with_isinstance(type_name))

    if var in false_env:
        false_env[var] = (false_env[var].remove_isinstance(type_name))


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


def initialize_environments(func_node, cfg, context):
    n = len(cfg.blocks)

    in_envs = [{} for _ in range(n)]
    out_envs = [{} for _ in range(n)]

    entry_env = function_parameter_environment(
        func_node,
        context,
    )

    if n > 0:
        in_envs[0] = entry_env.copy()

    return in_envs, out_envs


def assign_attribute_type(
    target,
    value_type,
    env,
    context,
):

    if not isinstance(target, ast.Attribute):
        return

    if isinstance(target.value, ast.Name):
        env_name = (f"{target.value.id}."
                    f"{target.attr}")

        env[env_name] = value_type

    obj_type = infer_type_from_expr(
        target.value,
        env,
        context,
    )

    if not isinstance(
            obj_type,
            InstanceType,
    ):
        return

    class_info = context.known_classes.get(obj_type.class_name)

    if class_info is None:
        return

    old_type = class_info.attributes.get(target.attr)

    if (old_type is None or isinstance(old_type, Unknown)):
        class_info.attributes[target.attr] = value_type
        return

    if isinstance(value_type, Unknown):
        return

    class_info.attributes[target.attr] = old_type.join(value_type)


def merge_into_environment(target, source):
    for name, typ in source.items():
        if name in target:
            target[name] = target[name].join(typ)
        else:
            target[name] = typ


def is_with_marker(stmt):
    return (isinstance(stmt, tuple) and len(stmt) == 2 and stmt[0] == "with")


def is_for_marker(stmt):
    return (isinstance(stmt, tuple) and len(stmt) >= 3 and stmt[0] == "for")


def is_async_for_marker(stmt):
    return (isinstance(stmt, tuple) and len(stmt) >= 3 and stmt[0] == "async_for")


def is_block_marker(stmt):
    return (is_condition_marker(stmt) or is_for_marker(stmt) or is_with_marker(stmt)
            or is_async_for_marker(stmt))


def prepare_block(
    block,
    in_envs,
    block_index,
):
    env = in_envs[block_index].copy()

    marker = None

    if block.stmts and is_block_marker(block.stmts[0]):
        marker = block.stmts[0]
        stmts = block.stmts[1:]
    else:
        marker = None
        stmts = block.stmts

    return env, marker, stmts


def is_condition_marker(stmt):
    return (isinstance(stmt, tuple) and len(stmt) >= 2 and stmt[0] == "cond")


def propagate_with_marker(
    block,
    env,
    marker,
    in_envs,
    context,
):
    items = marker[1]
    body_env = env.copy()

    transfer_with_items(
        items,
        body_env,
        context,
    )

    propagate_to_successors(
        block.succ,
        body_env,
        in_envs,
    )


def transfer_with_items(
    items,
    env,
    context,
):
    for item in items:
        # Make sure the context expression itself is analysed.
        infer_type_from_expr(
            item.context_expr,
            env,
            context,
        )

        if item.optional_vars is None:
            continue

        enter_type = infer_with_enter_type(
            item.context_expr,
            env,
            context,
        )

        bind_target_type(
            item.optional_vars,
            enter_type,
            env,
        )

    return env


def infer_with_enter_type(
    context_expr,
    env,
    context,
):
    enter_call = ast.Call(
        func=ast.Attribute(
            value=context_expr,
            attr="__enter__",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[],
    )

    result = infer_type_from_expr(
        enter_call,
        env,
        context,
    )

    if isinstance(result, Unknown):
        return TopType()

    return result


def bind_target_type(
    target,
    typ,
    env,
):

    if isinstance(target, ast.Name):
        env[target.id] = typ
        return

    if isinstance(target, (ast.Tuple, ast.List)):
        bind_sequence_target(
            target,
            typ,
            env,
        )
        return


def bind_sequence_target(target, typ, env):
    if (isinstance(typ, TupleType) and len(typ.elems) == len(target.elts)):
        for element, element_type in zip(target.elts, typ.elems):
            bind_target_type(
                element,
                element_type,
                env,
            )
        return

    for element in target.elts:
        bind_target_type(
            element,
            TopType(),
            env,
        )


def transfer_statement(stmt, env, context):
    if isinstance(stmt, ast.Assign):
        return transfer_assign(stmt, env, context)

    if isinstance(stmt, ast.AnnAssign):
        return transfer_annassign(stmt, env, context)

    if isinstance(stmt, ast.AugAssign):
        return transfer_augassign(
            stmt,
            env,
            context,
        )

    if isinstance(stmt, ast.Expr):
        return transfer_expr(stmt, env, context)

    if isinstance(stmt, ast.Delete):
        return transfer_delete(stmt, env)

    #if isinstance(stmt, ast.Return):
    #   return transfer_return(stmt, env, context)

    return env


def transfer_augassign(
    stmt,
    env,
    context,
):
    target = stmt.target

    if isinstance(target, ast.Name):

        left = ast.Name(
            id=target.id,
            ctx=ast.Load(),
        )

        expression = ast.BinOp(
            left=left,
            op=stmt.op,
            right=stmt.value,
        )

        env[target.id] = infer_type_from_expr(
            expression,
            env,
            context,
        )

        return env

    if isinstance(target, ast.Attribute):
        load_target = ast.Attribute(
            value=target.value,
            attr=target.attr,
            ctx=ast.Load(),
        )

        expression = ast.BinOp(
            left=load_target,
            op=stmt.op,
            right=stmt.value,
        )

        result_type = infer_type_from_expr(
            expression,
            env,
            context,
        )

        assign_attribute_type(
            target,
            result_type,
            env,
            context,
        )

        return env

    if isinstance(target, ast.Subscript):
        return transfer_augassign_subscript


def transfer_augassign_subscript(
    stmt,
    env,
    context,
):
    target = stmt.target

    if not isinstance(target.value, ast.Name):
        return env

    container = target.value.id

    left = ast.Subscript(
        value=target.value,
        slice=target.slice,
        ctx=ast.Load(),
    )

    expression = ast.BinOp(
        left=left,
        op=stmt.op,
        right=stmt.value,
    )

    result_type = infer_type_from_expr(
        expression,
        env,
        context,
    )

    key_type = infer_type_from_expr(
        target.slice,
        env,
        context,
    )

    update_subscript_assignment(
        container,
        key_type,
        result_type,
        env,
        is_slice=isinstance(target.slice, ast.Slice),
    )

    return env


def update_method_signature(
    func_node,
    ret_type,
    context,
):

    owner = function_owner(
        func_node,
        context,
    )

    if owner is None:
        return

    class_info = context.known_classes.get(owner)

    if class_info is None:
        return

    env = function_parameter_environment(
        func_node,
        context,
    )

    param_types = []

    for arg in func_node.args.args[1:]:
        param_types.append(env.get(
            arg.arg,
            Unknown(),
        ))

    previous = class_info.attributes.get(func_node.name)

    previous_return = Unknown()

    if isinstance(
            previous,
            CallableType,
    ):
        previous_return = previous.ret

    if func_node.name == "__init__":
        method_return = NoneType()

    elif isinstance(
            ret_type,
        (
            Unknown,
            Bottom,
        ),
    ):
        method_return = previous_return

    else:
        method_return = ret_type

    class_info.attributes[func_node.name] = CallableType(
        param_types,
        method_return,
    )


def transfer_return(stmt, env, context):
    if stmt.value is None:
        return env

    return_type = infer_type_from_expr(
        stmt.value,
        env,
        context,
    )

    return env, return_type


def transfer_assign(
    stmt,
    env,
    context,
):
    for target in stmt.targets:
        if isinstance(target, ast.Name):
            assign_name(
                target,
                stmt.value,
                env,
                context,
            )

        elif isinstance(
                target,
                ast.Attribute,
        ):
            value_type = infer_type_from_expr(
                stmt.value,
                env,
                context,
            )

            assign_attribute_type(
                target,
                value_type,
                env,
                context,
            )

        elif isinstance(
                target,
                ast.Subscript,
        ):
            assign_subscript(
                target,
                stmt.value,
                env,
                context,
            )

        elif isinstance(
                target,
            (
                ast.Tuple,
                ast.List,
            ),
        ):
            value_type = infer_type_from_expr(
                stmt.value,
                env,
                context,
            )

            bind_target_type(
                target,
                value_type,
                env,
            )

    return env


def merge_call_argument_types(
    collected,
    name,
    arg_types,
):
    if name not in collected:
        collected[name] = arg_types
        return

    previous = collected[name]

    for i in range(min(
            len(previous),
            len(arg_types),
    )):
        current = arg_types[i]

        if isinstance(
                previous[i],
                Unknown,
        ):
            continue

        if isinstance(
                current,
                Unknown,
        ):
            previous[i] = Unknown()
            continue

        previous[i] = previous[i].join(current)


def function_parameter_environment(
    func_node,
    context,
):
    env = {}

    key = function_key(
        func_node,
        context,
    )

    owner = function_owner(
        func_node,
        context,
    )

    inferred_params = context.function_params.get(
        key,
        [],
    )

    for i, arg in enumerate(func_node.args.args):

        if owner is not None and i == 0:
            env[arg.arg] = InstanceType(owner)
            continue

        if arg.annotation is not None:
            env[arg.arg] = infer_annotation_type_with_classes(arg.annotation, context.known_classes)

        elif i < len(inferred_params):
            env[arg.arg] = inferred_params[i]

        else:
            env[arg.arg] = Unknown()

    if (owner is not None and func_node.args.args):
        self_name = func_node.args.args[0].arg

        class_info = context.known_classes.get(owner)

        if class_info is not None:
            for attr_name, attr_type in (class_info.attributes.items()):
                if isinstance(
                        attr_type,
                        CallableType,
                ):
                    continue

                env[f"{self_name}.{attr_name}"] = attr_type

    return env


def transfer_delete(stmt, env):
    for target in stmt.targets:
        delete_target(
            target,
            env,
        )

    return env


def delete_target(target, env):
    if isinstance(target, ast.Name):
        env.pop(
            target.id,
            None,
        )
        return

    if isinstance(
            target,
        (ast.Attribute, ast.Subscript),
    ):
        return

    if isinstance(
            target,
        (
            ast.Tuple,
            ast.List,
        ),
    ):
        for element in target.elts:
            delete_target(
                element,
                env,
            )

        return

    raise UnsupportedCFGError("unsupported delete target: "
                              f"{type(target).__name__}")


def assign_name(target, value, env, context):
    name = target.id
    typ = infer_type_from_expr(value, env, context)

    if typ is None:
        typ = Unknown()

    #if name in env:
    #   env[name] = env[name].join(typ)
    #else:
    env[name] = typ


def assign_subscript(target, value, env, context):
    if not isinstance(target.value, ast.Name):
        return

    container = target.value.id
    key_type = infer_type_from_expr(target.slice, env, context)
    value_type = infer_type_from_expr(value, env, context)

    is_slice = isinstance(target.slice, ast.Slice)

    update_subscript_assignment(
        container,
        key_type,
        value_type,
        env,
        is_slice=is_slice,
    )


def update_subscript_assignment(
    container,
    key_type,
    value_type,
    env,
    is_slice=False,
):
    current = env.get(container, Unknown())

    if isinstance(current, DictType):
        env[container] = DictType(
            current.key_t.join(key_type),
            current.val_t.join(value_type),
        )
        return

    if isinstance(current, ListType):
        # env[container] = ListType(current.elem.join(value_type))
        update_list_subscript_assignment(
            container,
            current,
            value_type,
            env,
            is_slice=is_slice,
        )
        return

    if isinstance(current, Unknown):
        # env[container] = infer_subscript_container(key_type, value_type,)
        return

    env[container] = Unknown()


def update_list_subscript_assignment(
    container,
    current,
    value_type,
    env,
    is_slice,
):
    if not is_slice:
        env[container] = ListType(current.elem.join(value_type))
        return

    if isinstance(value_type, ListType):
        env[container] = ListType(current.elem.join(value_type.elem))
        return

    env[container] = TopType()


def infer_subscript_container(key_type, value_type):
    if isinstance(key_type, StrType):
        return DictType(key_type, value_type)

    if isinstance(key_type, IntType):
        return ListType(value_type)

    return Unknown()


def transfer_annassign(
    stmt,
    env,
    context,
):

    declared_type = infer_annotation_type_with_classes(
        stmt.annotation,
        context.known_classes,
    )

    final_type = declared_type

    if isinstance(
            declared_type,
        (
            AnyType,
            Unknown,
        ),
    ):
        if stmt.value is not None:
            inferred_type = infer_type_from_expr(
                stmt.value,
                env,
                context,
            )

            if not isinstance(
                    inferred_type,
                (
                    AnyType,
                    Unknown,
                ),
            ):
                final_type = inferred_type

    if isinstance(
            stmt.target,
            ast.Name,
    ):
        env[stmt.target.id] = final_type
        return env

    if isinstance(
            stmt.target,
            ast.Attribute,
    ):
        assign_attribute_type(
            stmt.target,
            final_type,
            env,
            context,
        )
        return env

    return env


""" def transfer_expr(stmt, env, context):
    expr = stmt.value

    if isinstance(expr, ast.Call):
        handle_call_effect(expr, env, context, )

        if (isinstance(expr.func, ast.Attribute)
            and expr.func.attr == "setdefault"
            and isinstance(expr.func.value, ast. Name)
            ):
            dict_name = expr.func.value.id
            current = env.get(dict_name, Unknown())

            if isinstance(current, DictType):
                if len(expr.args) >=2:
                    default_type = infer_type_from_expr(expr.args[1], env, context,)
                    return current.val_t.join(default_type)

                return current.val_t
            

    return env    """


def collect_function_argument_types(
    tree,
    context,
    max_iters=10,
) -> set[str]:
    for _ in range(max_iters):
        collected = {}

        context.function_param_observations = {}

        module_collector = CallArgumentCollector(
            {},
            collected,
            context,
        )

        for stmt in tree.body:
            if isinstance(
                    stmt,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                continue

            module_collector.visit(stmt)

        for node in ast.walk(tree):
            if not isinstance(
                    node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            env = function_parameter_environment(
                node,
                context,
            )

            collector = CallArgumentCollector(
                env,
                collected,
                context,
            )

            for stmt in node.body:
                collector.visit(stmt)

        if collected == context.function_params:
            break

        context.function_params = {name: list(types) for name, types in collected.items()}

    unsafe = set()

    for function_name, observations in (context.function_param_observations.items()):
        for observed_types in observations:
            if len(observed_types) > 1:
                unsafe.add(function_name)
                break

    changed = True

    while changed:
        changed = False

        for stmt in tree.body:
            if isinstance(
                    stmt,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue

            for call in ast.walk(stmt):
                if not isinstance(
                        call,
                        ast.Call,
                ):
                    continue

                if not isinstance(
                        call.func,
                        ast.Name,
                ):
                    continue

                called_name = call.func.id

                if (called_name not in context.function_keys.values()):
                    continue

                has_unsafe_argument = False

                for arg in call.args:
                    for child in ast.walk(arg):
                        if (isinstance(
                                child,
                                ast.Call,
                        ) and isinstance(
                                child.func,
                                ast.Name,
                        ) and child.func.id in unsafe):
                            has_unsafe_argument = True
                            break

                if (has_unsafe_argument and called_name not in unsafe):
                    unsafe.add(called_name)
                    changed = True

        # Propagation through parameters of an
        # already unsafe function.
        #
        # Example:
        #
        # def cube(x):
        #     return square(x) * x
        #
        # If cube is unsafe, x may be int or float,
        # so square must also remain unannotated.
        for func_node in ast.walk(tree):
            if not isinstance(
                    func_node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            current_key = function_key(
                func_node,
                context,
            )

            if current_key not in unsafe:
                continue

            parameter_names = {arg.arg for arg in func_node.args.args}

            pending: list[ast.AST] = list(func_node.body)

            while pending:
                node = pending.pop()

                if isinstance(
                        node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.Lambda,
                        ast.ClassDef,
                    ),
                ):
                    continue

                if (isinstance(
                        node,
                        ast.Call,
                ) and isinstance(
                        node.func,
                        ast.Name,
                )):
                    called_name = node.func.id

                    if (called_name in context.function_keys.values()):
                        passes_parameter = any(
                            isinstance(
                                arg,
                                ast.Name,
                            ) and arg.id in parameter_names for arg in node.args)

                        if (passes_parameter and called_name not in unsafe):
                            unsafe.add(called_name)
                            changed = True

                pending.extend(ast.iter_child_nodes(node))

    return unsafe

    # context.function_params = collected


def transfer_expr(
    stmt: ast.Expr,
    env: dict[str, Type],
    context: InferenceContext,
) -> dict[str, Type]:

    expr = stmt.value

    if isinstance(expr, ast.Call):
        handle_call_effect(expr, env, context)

    return env


def handle_call_effect(call, env, context):
    if not isinstance(call.func, ast.Attribute):
        return

    if call.func.attr == "append":
        handle_append(call, env, context)

    elif call.func.attr == "setdefault":
        handle_setdefault(call, env, context)


def handle_setdefault(call, env, context):
    if not isinstance(call.func, ast.Attribute):
        return

    if call.func.attr != "setdefault":
        return

    if not isinstance(call.func.value, ast.Name):
        return

    dict_name = call.func.value.id

    key_type = infer_type_from_expr(
        call.args[0],
        env,
        context,
    )

    #default_type = infer_type_from_expr(call.args[1], env, context,)

    current = env.get(
        dict_name,
        Unknown(),
    )

    if not isinstance(current, DictType):
        return

    if len(call.args) >= 2:
        default_type = infer_type_from_expr(
            call.args[1],
            env,
            context,
        )

        new_key_type = current.key_t.join(key_type)
        new_value_type = current.val_t.join(default_type)

        env[dict_name] = DictType(
            new_key_type,
            new_value_type,
        )
        return

    # No explicit default: Python uses None.
    new_key_type = current.key_t.join(key_type)
    new_value_type = current.val_t.join(NoneType())

    env[dict_name] = DictType(
        new_key_type,
        new_value_type,
    )


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
            env[listname] = ListType(current.elem.join(arg_type))
        #elif isinstance(current, Unknown):
        #   env[listname] = ListType(arg_type)

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
            updated_value = ListType(value_type.elem.join(arg_type))
        # elif isinstance(default_type, ListType):
        #     updated_value = ListType(
        #         default_type.elem.join(arg_type)
        #     )
        else:
            updated_value = ListType(default_type.elem.join(arg_type))

        env[dictname] = DictType(
            current.key_t.join(key_type),
            value_type.join(updated_value),
        )
        return

    if isinstance(default_type, ListType):
        env[dictname] = DictType(
            key_type,
            ListType(default_type.elem.join(arg_type)),
        )


def merge_environment(previous, current):
    merged = previous.copy()
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


def infer_class_call_type(func_name, known_classes):
    if func_name in known_classes:
        return InstanceType(func_name)

    return None


def infer_instance_attribute_type(
    obj_type,
    attr_name,
    known_classes,
):
    pending = [obj_type.class_name]
    visited = set()

    while pending:
        class_name = pending.pop()

        if class_name in visited:
            continue

        visited.add(class_name)

        class_info = known_classes.get(class_name)

        if class_info is None:
            continue

        if attr_name in class_info.attributes:
            return class_info.attributes[attr_name]

        pending.extend(class_info.bases)

    return Unknown()


def infer_dict_attribute_type(obj_type, attr_name):
    if attr_name == "copy":
        return CallableType([], DictType(obj_type.key_t, obj_type.val_t))

    if attr_name == "get":
        return CallableType([], obj_type.val_t)

    if attr_name == "pop":
        return CallableType([], obj_type.val_t)

    if attr_name == "popitem":
        return CallableType(
            [],
            TupleType([obj_type.key_t, obj_type.val_t]),
        )

    if attr_name == "keys":
        return CallableType(
            [],
            ListType(obj_type.key_t),
        )

    if attr_name == "values":
        return CallableType([], ListType(obj_type.val_t))

    if attr_name == "items":
        return CallableType(
            [],
            ListType(TupleType([obj_type.key_t, obj_type.val_t])),
        )

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
    elems = [infer_type_from_expr(element, env, context) for element in expr.elts]
    return TupleType(elems)


def infer_tuple_constructor_type(
    expr,
    env,
    context,
):
    if expr.keywords:
        return Unknown()

    if len(expr.args) == 0:
        return TupleType([])

    if len(expr.args) != 1:
        return Unknown()

    arg = expr.args[0]

    if isinstance(arg, ast.Tuple):
        return infer_tuple_literal_type(
            arg,
            env,
            context,
        )

    if isinstance(arg, ast.List):
        return TupleType([infer_type_from_expr(
            element,
            env,
            context,
        ) for element in arg.elts])

    arg_type = infer_type_from_expr(
        arg,
        env,
        context,
    )

    if isinstance(arg_type, TupleType):
        return TupleType(list(arg_type.elems))

    return Unknown()


def infer_dict_literal_type(expr, env, context):
    key_type = Unknown()
    value_type = Unknown()

    for key, value in zip(expr.keys, expr.values):
        if key is None:
            continue

        key_type = key_type.join(infer_type_from_expr(key, env, context))

        value_type = value_type.join(infer_type_from_expr(value, env, context))

    return DictType(
        key_type,
        value_type,
    )


def infer_binop_type(expr, env, context):
    left = infer_type_from_expr(
        expr.left,
        env,
        context,
    )

    right = infer_type_from_expr(
        expr.right,
        env,
        context,
    )

    print(
        "[PYTYPE BINOP]",
        ast.dump(expr),
        "LEFT =",
        left,
        "RIGHT =",
        right,
        "OP =",
        type(expr.op).__name__,
    )

    numeric_types = (
        BoolType,
        IntType,
        FloatType,
        ComplexType,
    )

    real_numeric_types = (
        BoolType,
        IntType,
        FloatType,
    )

    # Special cases for +.
    if isinstance(expr.op, ast.Add):
        result = infer_add_type(
            left,
            right,
        )

        if result is not None:
            return result

    # Tuple repetition is only valid for multiplication.
    if isinstance(expr.op, ast.Mult):
        if (isinstance(left, TupleType) and isinstance(expr.right, ast.Constant)
                and isinstance(expr.right.value, int)):
            count = expr.right.value

            if count >= 0:
                return TupleType(left.elems * count)

        if (isinstance(right, TupleType) and isinstance(expr.left, ast.Constant)
                and isinstance(expr.left.value, int)):
            count = expr.left.value

            if count >= 0:
                return TupleType(right.elems * count)

    # Complex numeric promotion.
    if isinstance(
            expr.op,
        (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
        ),
    ):
        if (isinstance(left, numeric_types) and isinstance(right, numeric_types)
                and (isinstance(left, ComplexType) or isinstance(right, ComplexType))):
            return ComplexType()

    # Normal real division and floor division.
    if isinstance(
            expr.op,
        (
            ast.Div,
            ast.FloorDiv,
        ),
    ):
        if not (isinstance(left, real_numeric_types) and isinstance(right, real_numeric_types)):
            return Unknown()

        if isinstance(expr.op, ast.Div):
            return FloatType()

        if (isinstance(left, FloatType) or isinstance(right, FloatType)):
            return FloatType()

        return IntType()

    # Remaining real arithmetic.
    if (isinstance(left, FloatType) or isinstance(right, FloatType)):
        return FloatType()

    if (isinstance(left, IntType) and isinstance(right, IntType)):
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
        return infer_named_call_type(
            func.id,
            expr,
            env,
            context,
        )

    if isinstance(func, ast.Attribute):
        if (isinstance(func.value, ast.Name) and func.value.id == "cmath"):
            return infer_cmath_call_type(func.attr, expr, env, context)

        obj_type = infer_type_from_expr(
            func.value,
            env,
            context,
        )

        if isinstance(
                obj_type,
                InstanceType,
        ):
            method_key = resolve_method_key(
                obj_type.class_name,
                func.attr,
                context,
            )

            if (method_key is not None and method_key in context.function_returns):
                return context.function_returns[method_key]

        arg_types = []
        for arg in expr.args:
            arg_types.append(infer_type_from_expr(arg, env, context))

        if isinstance(obj_type, DictType) and func.attr == "get":
            if not 1 <= len(arg_types) <= 2 or expr.keywords:
                return Unknown()

            default_type = NoneType()

            if len(arg_types) == 2:
                default_type = arg_types[1]

            return obj_type.val_t.join(default_type)

        if isinstance(obj_type, DictType) and func.attr == "setdefault":
            if not 1 <= len(arg_types) <= 2 or expr.keywords:
                return Unknown()

            default_type = NoneType()
            if len(arg_types) == 2:
                default_type = arg_types[1]

            result = obj_type.val_t.join(default_type)
            handle_setdefault(expr, env, context)
            return result

        attribute_type = infer_attribute_type(
            obj_type,
            func.attr,
            context.known_classes,
        )

        if isinstance(attribute_type, CallableType):
            return attribute_type.ret

        #return resolve_method_call(obj_type, func.attr, arg_types)

    return Unknown()


def infer_named_call_type(
    func_name,
    expr,
    env,
    context,
):
    if (func_name == "nondet_int" and not expr.args and not expr.keywords):
        return IntType()

    builtin_type = infer_builtin_call_type(func_name, expr, env, context)
    if func_name in context.function_returns:
        return context.function_returns[func_name]

    builtin_type = infer_builtin_call_type(func_name, expr, env, context)

    if builtin_type is not None:
        return builtin_type

    class_type = infer_class_call_type(
        func_name,
        context.known_classes,
    )

    if class_type is not None:
        return class_type

    return Unknown()


def snapshot_class_attributes(context, ):
    return {
        class_name: dict(class_info.attributes)
        for class_name, class_info in context.known_classes.items()
    }


def infer_builtin_call_type(name, expr, env, context):
    if name == "nondet_bool" and name not in env:
        if expr.args or expr.keywords:
            return Unknown()

        return BoolType()
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

    if name == "round":
        if len(expr.args) == 1 and not expr.keywords:
            return IntType()

        return Unknown()

    if name == "bin":
        if len(expr.args) == 1 and not expr.keywords:
            return StrType()

        return Unknown()

    if name == "list":
        return ListType(Unknown())

    if name == "dict":
        return DictType(Unknown(), Unknown())

    if name == "tuple":
        return infer_tuple_constructor_type(
            expr,
            env,
            context,
        )

    if name == "complex":
        return ComplexType()

    if name == "set":
        return SetType(Unknown())

    return None


def infer_lambda_type(expr, env, context):
    param_types: List[Type] = [Unknown() for _ in expr.args.args]

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

    if isinstance(index_expr, ast.Slice):
        lower: int | None = None
        upper: int | None = None
        step: int | None = None

        if index_expr.lower is not None:
            if (isinstance(index_expr.lower, ast.Constant)
                    and isinstance(index_expr.lower.value, int)
                    and not isinstance(index_expr.lower.value, bool)):
                lower = index_expr.lower.value
            elif (isinstance(index_expr.lower, ast.UnaryOp)
                  and isinstance(index_expr.lower.op, ast.USub)
                  and isinstance(index_expr.lower.operand, ast.Constant)
                  and isinstance(index_expr.lower.operand.value, int)):
                lower = index_expr.lower.operand.value
            else:
                return Unknown()

        if index_expr.upper is not None:
            if (isinstance(index_expr.upper, ast.Constant)
                    and isinstance(index_expr.upper.value, int)
                    and not isinstance(index_expr.upper.value, bool)):
                upper = index_expr.upper.value

            elif (isinstance(index_expr.upper, ast.UnaryOp)
                  and isinstance(index_expr.upper.op, ast.USub)
                  and isinstance(index_expr.upper.operand, ast.Constant)
                  and isinstance(index_expr.upper.operand.value, int)):
                upper = -index_expr.upper.operand.value

            else:
                return Unknown()

        if index_expr.step is not None:
            if (isinstance(index_expr.step, ast.Constant)
                    and isinstance(index_expr.step.value, int)
                    and not isinstance(index_expr.step.value, bool)):
                step = index_expr.step.value

            elif (isinstance(index_expr.step, ast.UnaryOp)
                  and isinstance(index_expr.step.op, ast.USub)
                  and isinstance(index_expr.step.operand, ast.Constant)
                  and isinstance(index_expr.step.operand.value, int)):

                step = -index_expr.step.operand.value
            else:
                return Unknown()

        if step == 0:
            return Unknown()

        elements = tuple_type.elems[slice(lower, upper, step)]

        return TupleType(elements)

    index_type = infer_type_from_expr(
        index_expr,
        env,
        None,
    )

    if not isinstance(index_type, IntType):
        return Unknown()

    # Normal tuple indexing
    if isinstance(index_expr, ast.Constant):
        index = index_expr.value

        if isinstance(index, int) and not isinstance(index, bool):
            if -len(tuple_type.elems) <= index < len(tuple_type.elems):
                return tuple_type.elems[index]

            return Unknown()

    # Dynamic integer index: could refer to any tuple element.
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
        return True

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
            print("COMPLEX ATTRIBUTE:", obj_type, attr_name, "-> FloatType")
            return FloatType()

    if isinstance(obj_type, InstanceType):
        return infer_instance_attribute_type(
            obj_type,
            attr_name,
            known_classes,
        )

    return Unknown()


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
                    attributes[target.id] = (infer_type_from_expr(
                        stmt.value,
                        {},
                        context,
                    ))

        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            attributes[stmt.name] = infer_method_type(stmt, known_classes)

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


def collect_known_classes(tree) -> dict[str, Any]:
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

            if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                continue

            attributes[target.attr] = infer_annotation_type(node.annotation)
            continue

        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue

            target = node.targets[0]

            if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
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
            return ListType(infer_annotation_type(slice_node))

        if name == "set":
            return SetType(infer_annotation_type(slice_node))

        if name == "dict":
            if isinstance(slice_node, ast.Tuple):
                if len(slice_node.elts) == 2:
                    return DictType(
                        infer_annotation_type(slice_node.elts[0]),
                        infer_annotation_type(slice_node.elts[1]),
                    )

            return DictType(Unknown(), Unknown())

        if name == "tuple":
            if isinstance(slice_node, ast.Tuple):
                return TupleType([infer_annotation_type(element) for element in slice_node.elts])

            return TupleType([infer_annotation_type(slice_node)])
    return Unknown()


def infer_type_from_expr(expr, env, context):
    assert isinstance(env, dict), (f"infer_type_from_expr received non-env: "
                                   f"{type(env).__name__}: {env!r}; expr={ast.dump(expr)}")
    if isinstance(expr, ast.Constant):
        return infer_constant_type(expr)

    if isinstance(expr, ast.Name):
        return env.get(expr.id, Unknown())

    if isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name):
            env_name = (f"{expr.value.id}."
                        f"{expr.attr}")

            if env_name in env:
                return env[env_name]
        obj_type = infer_type_from_expr(expr.value, env, context)
        if isinstance(obj_type, ComplexType):
            if expr.attr in {"real", "imag"}:
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
        return infer_call_type(
            expr,
            env,
            context,
        )

    if isinstance(expr, ast.Lambda):
        return infer_lambda_type(expr, env, context)

    return Unknown()


def repr_dict(d):
    return {k: repr(v) for k, v in d.items()}

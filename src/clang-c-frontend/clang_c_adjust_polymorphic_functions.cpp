#include <clang-c-frontend/clang_c_adjust.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/prefix.h>

exprt clang_c_adjust::is_gcc_polymorphic_builtin(
  const irep_idt &identifier,
  const exprt::operandst &arguments)
{
  if(
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_add") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_sub") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_or") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_and") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_xor") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_nand") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_add_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_sub_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_or_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_and_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_xor_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_nand_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_lock_test_and_set"))
  {
    // These are polymorphic, see
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html
    const exprt &ptr_arg = arguments.front();
    const auto &pointer_type = to_pointer_type(ptr_arg.type());

    code_typet t{
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(pointer_type.subtype())},
      pointer_type.subtype()};
    t.make_ellipsis();
    symbol_exprt result{identifier, std::move(t)};
    return result;
  }
  else if(
    has_prefix(identifier.as_string(), "c:@F@__sync_bool_compare_and_swap") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_val_compare_and_swap"))
  {
    // These are polymorphic, see
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html
    const exprt &ptr_arg = arguments.front();

    const typet &base_type = to_pointer_type(ptr_arg.type()).subtype();
    typet sync_return_type = base_type;
    if(has_prefix(identifier.as_string(), "c:@F@__sync_val_compare_and_swap"))
      sync_return_type = bool_type();

    code_typet t{
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(base_type),
       code_typet::argumentt(base_type)},
      sync_return_type};
    t.make_ellipsis();
    symbol_exprt result{identifier, std::move(t)};
    return result;
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__sync_lock_release"))
  {
    // This is polymorphic, see
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html
    const exprt &ptr_arg = arguments.front();

    code_typet t{{code_typet::argumentt(ptr_arg.type())}, empty_typet()};
    t.make_ellipsis();
    symbol_exprt result{identifier, std::move(t)};
    return result;
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__atomic_load_n"))
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    const code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(int_type())},
      to_pointer_type(ptr_arg.type()).subtype());
    symbol_exprt result(identifier, t);
    return result;
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__atomic_store_n"))
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    const auto &base_type = to_pointer_type(ptr_arg.type()).subtype();

    const code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(base_type),
       code_typet::argumentt(int_type())},
      empty_typet());
    symbol_exprt result(identifier, t);
    return result;
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__atomic_exchange_n"))
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();
    const auto &base_type = to_pointer_type(ptr_arg.type()).subtype();

    const code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(base_type),
       code_typet::argumentt(int_type())},
      base_type);
    symbol_exprt result(identifier, t);
    return result;
  }
  else if(
    has_prefix(identifier.as_string(), "c:@F@__atomic_load") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_store"))
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    const code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(int_type())},
      empty_typet());
    symbol_exprt result(identifier, t);
    return result;
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__atomic_exchange"))
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    const code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(int_type())},
      empty_typet());
    symbol_exprt result(identifier, t);
    return result;
  }
  else if(
    has_prefix(identifier.as_string(), "c:@F@__atomic_compare_exchange_n") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_compare_exchange"))
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    code_typet::argumentst parameters;
    parameters.push_back(code_typet::argumentt(ptr_arg.type()));
    parameters.push_back(code_typet::argumentt(ptr_arg.type()));

    if(has_prefix(identifier.as_string(), "c:@F@__atomic_compare_exchange"))
      parameters.push_back(code_typet::argumentt(ptr_arg.type()));
    else
      parameters.push_back(
        code_typet::argumentt(to_pointer_type(ptr_arg.type()).subtype()));

    parameters.push_back(code_typet::argumentt(bool_type()));
    parameters.push_back(code_typet::argumentt(int_type()));
    parameters.push_back(code_typet::argumentt(int_type()));
    code_typet t(std::move(parameters), bool_type());
    symbol_exprt result(identifier, t);
    return result;
  }
  else if(
    has_prefix(identifier.as_string(), "c:@F@__atomic_add_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_sub_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_xor_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_or_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_nand_fetch"))
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(to_pointer_type(ptr_arg.type()).subtype()),
       code_typet::argumentt(int_type())},
      to_pointer_type(ptr_arg.type()).subtype());
    symbol_exprt result(identifier, std::move(t));
    return result;
  }
  else if(
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_add") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_sub") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_and") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_xor") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_or") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_nand"))
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(to_pointer_type(ptr_arg.type()).subtype()),
       code_typet::argumentt(int_type())},
      to_pointer_type(ptr_arg.type()).subtype());
    symbol_exprt result(identifier, std::move(t));
    return result;
  }

  return nil_exprt();
}

static symbolt
result_symbol(const irep_idt &identifier, const typet &type, contextt &context)
{
  symbolt symbol;
  symbol.id = id2string(identifier) + "::1::result";
  symbol.name = "result";
  symbol.type = type;

  context.add(symbol);

  return symbol;
}

static void convert_expression_to_code(exprt &expr)
{
  if(expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

code_blockt clang_c_adjust::instantiate_gcc_polymorphic_builtin(
  const irep_idt &identifier,
  const symbol_exprt &function_symbol)
{
  const irep_idt &identifier_with_type = function_symbol.get_identifier();
  const code_typet &code_type = to_code_type(function_symbol.type());

  code_blockt block;

  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  block.operands().push_back(label);

  // atomic scope begin
  side_effect_expr_function_callt atomic_begin;
  atomic_begin.function() = symbol_exprt("c:@F@__ESBMC_atomic_begin");
  convert_expression_to_code(atomic_begin);
  block.operands().push_back(atomic_begin);

  // Change the cex to show that these code comes from the atomic/sync
  locationt new_loc = function_symbol.location();
  new_loc.set_function(function_symbol.name());

  if(
    has_prefix(identifier.as_string(), "c:@F@__sync_add_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_sub_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_or_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_and_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_xor_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_nand_and_fetch"))
  {
    __builtin_unreachable();
  }
  else if(
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_add") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_sub") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_or") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_and") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_xor") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_nand"))
  {
    const typet &type = code_type.return_type();

    const exprt &initial =
      symbol_expr(result_symbol(identifier_with_type, type, context));

    code_declt decl(initial);
    block.operands().push_back(decl);

    code_typet::argumentt arg0 = code_type.arguments()[0];
    code_assignt assign(
      initial,
      dereference_exprt(
        symbol_exprt(arg0.cmt_identifier(), arg0.type()), arg0.type()));
    assign.location() = new_loc;
    block.operands().push_back(assign);

    exprt new_expr;
    if(has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_add"))
    {
      if(type.is_floatbv())
        new_expr = exprt("ieee_add", type);
      else
        new_expr = exprt("+", type);
    }
    else if(has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_sub"))
    {
      if(type.is_floatbv())
        new_expr = exprt("ieee_sub", type);
      else
        new_expr = exprt("-", type);
    }
    else if(has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_or"))
    {
      new_expr = exprt("or", type);
    }
    else if(has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_and"))
    {
      new_expr = exprt("and", type);
    }
    else if(has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_xor"))
    {
      new_expr = exprt("bitxor", type);
    }
    else if(has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_nand"))
    {
      new_expr = exprt("bitand", type);
    }

    dereference_exprt arg0_deref(
      symbol_exprt(arg0.cmt_identifier(), arg0.type()), arg0.type());

    code_typet::argumentt arg1 = code_type.arguments()[1];
    new_expr.copy_to_operands(
      arg0_deref, symbol_exprt(arg1.cmt_identifier(), arg1.type()));

    code_assignt assign1(arg0_deref, new_expr);
    assign1.location() = new_loc;
    block.operands().push_back(assign1);

    // atomic scope end
    side_effect_expr_function_callt atomic_end;
    atomic_end.function() = symbol_exprt("c:@F@__ESBMC_atomic_end");
    convert_expression_to_code(atomic_end);
    block.operands().push_back(atomic_end);

    code_returnt ret;
    ret.return_value() = initial;
    ret.location() = new_loc;
    block.operands().push_back(ret);
  }
  else if(has_prefix(
            identifier.as_string(), "c:@F@__sync_bool_compare_and_swap"))
  {
    __builtin_unreachable();
  }
  else if(has_prefix(
            identifier.as_string(), "c:@F@__sync_val_compare_and_swap"))
  {
    __builtin_unreachable();
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__sync_lock_release"))
  {
    __builtin_unreachable();
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__sync_lock_test_and_set"))
  {
    __builtin_unreachable();
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__atomic_load_n"))
  {
    const typet &type = code_type.return_type();

    const exprt &result =
      symbol_expr(result_symbol(identifier_with_type, type, context));

    code_declt decl(result);
    block.operands().push_back(decl);

    code_typet::argumentt arg0 = code_type.arguments()[0];
    code_assignt assign(
      result,
      dereference_exprt(
        symbol_exprt(arg0.cmt_identifier(), arg0.type()), arg0.type()));
    assign.location() = new_loc;
    block.operands().push_back(assign);

    // atomic scope end
    side_effect_expr_function_callt atomic_end;
    atomic_end.function() = symbol_exprt("c:@F@__ESBMC_atomic_end");
    convert_expression_to_code(atomic_end);
    block.operands().push_back(atomic_end);

    code_returnt ret;
    ret.return_value() = result;
    ret.location() = new_loc;
    block.operands().push_back(ret);
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__atomic_store_n"))
  {
    code_typet::argumentt arg0 = code_type.arguments()[0];
    code_typet::argumentt arg1 = code_type.arguments()[1];
    code_assignt assign(
      dereference_exprt(
        symbol_exprt(arg0.cmt_identifier(), arg0.type()), arg0.type()),
      symbol_exprt(arg1.cmt_identifier(), arg1.type()));
    assign.location() = new_loc;
    block.operands().push_back(assign);

    // atomic scope end
    side_effect_expr_function_callt atomic_end;
    atomic_end.function() = symbol_exprt("c:@F@__ESBMC_atomic_end");
    convert_expression_to_code(atomic_end);
    block.operands().push_back(atomic_end);
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__atomic_exchange_n"))
  {
    __builtin_unreachable();
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__atomic_load"))
  {
    code_typet::argumentt arg0 = code_type.arguments()[0];
    code_typet::argumentt arg1 = code_type.arguments()[1];
    code_assignt assign(
      dereference_exprt(
        symbol_exprt(arg1.cmt_identifier(), arg1.type()), arg1.type()),
      dereference_exprt(
        symbol_exprt(arg0.cmt_identifier(), arg0.type()), arg0.type()));
    assign.location() = new_loc;
    block.operands().push_back(assign);

    // atomic scope end
    side_effect_expr_function_callt atomic_end;
    atomic_end.function() = symbol_exprt("c:@F@__ESBMC_atomic_end");
    convert_expression_to_code(atomic_end);
    block.operands().push_back(atomic_end);
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__atomic_store"))
  {
    code_typet::argumentt arg0 = code_type.arguments()[0];
    code_typet::argumentt arg1 = code_type.arguments()[1];
    code_assignt assign(
      dereference_exprt(
        symbol_exprt(arg0.cmt_identifier(), arg0.type()), arg0.type()),
      dereference_exprt(
        symbol_exprt(arg1.cmt_identifier(), arg1.type()), arg1.type()));
    assign.location() = new_loc;
    block.operands().push_back(assign);

    // atomic scope end
    side_effect_expr_function_callt atomic_end;
    atomic_end.function() = symbol_exprt("c:@F@__ESBMC_atomic_end");
    convert_expression_to_code(atomic_end);
    block.operands().push_back(atomic_end);
  }
  else if(has_prefix(identifier.as_string(), "c:@F@__atomic_exchange"))
  {
    __builtin_unreachable();
  }
  else if(has_prefix(
            identifier.as_string(), "c:@F@__atomic_compare_exchange_n"))
  {
    __builtin_unreachable();
  }
  else if(has_prefix(
            identifier.as_string(), "c:@F@__atomic_compare_exchange_n"))
  {
    __builtin_unreachable();
  }
  else if(
    has_prefix(identifier.as_string(), "c:@F@__atomic_add_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_sub_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_xor_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_or_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_nand_fetch"))
  {
    __builtin_unreachable();
  }
  else if(
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_add") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_sub") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_and") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_xor") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_or") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_nand"))
  {
    __builtin_unreachable();
  }

  return block;
}
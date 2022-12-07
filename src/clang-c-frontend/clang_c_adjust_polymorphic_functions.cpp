#include <clang-c-frontend/clang_c_adjust.h>
#include <util/c_types.h>
#include <util/expr_util.h>

exprt clang_c_adjust::is_gcc_polymorphic_builtin(
  const irep_idt &identifier,
  const exprt::operandst &arguments)
{
  if(
    identifier == "c:@F@__sync_fetch_and_add" || identifier == "c:@F@__sync_fetch_and_sub" ||
    identifier == "c:@F@__sync_fetch_and_or" || identifier == "c:@F@__sync_fetch_and_and" ||
    identifier == "c:@F@__sync_fetch_and_xor" || identifier == "c:@F@__sync_fetch_and_nand" ||
    identifier == "c:@F@__sync_add_and_fetch" || identifier == "c:@F@__sync_sub_and_fetch" ||
    identifier == "c:@F@__sync_or_and_fetch" || identifier == "c:@F@__sync_and_and_fetch" ||
    identifier == "c:@F@__sync_xor_and_fetch" || identifier == "c:@F@__sync_nand_and_fetch" ||
    identifier == "c:@F@__sync_lock_test_and_set")
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
    identifier == "c:@F@__sync_bool_compare_and_swap" ||
    identifier == "c:@F@__sync_val_compare_and_swap")
  {
    // These are polymorphic, see
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html
    const exprt &ptr_arg = arguments.front();

    const typet &base_type = to_pointer_type(ptr_arg.type()).subtype();
    typet sync_return_type = base_type;
    if(identifier == "c:@F@__sync_val_compare_and_swap")
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
  else if(identifier == "c:@F@__sync_lock_release")
  {
    // This is polymorphic, see
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html
    const exprt &ptr_arg = arguments.front();

    code_typet t{
      {code_typet::argumentt(ptr_arg.type())}, pointer_typet(empty_typet())};
    t.make_ellipsis();
    symbol_exprt result{identifier, std::move(t)};
    return result;
  }
  else if(identifier == "c:@F@__atomic_load_n")
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
  else if(identifier == "c:@F@__atomic_store_n")
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    const auto &base_type = to_pointer_type(ptr_arg.type()).subtype();

    const code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(base_type),
       code_typet::argumentt(int_type())},
      pointer_typet(empty_typet()));
    symbol_exprt result(identifier, t);
    return result;
  }
  else if(identifier == "c:@F@__atomic_exchange_n")
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
  else if(identifier == "c:@F@__atomic_load" || identifier == "c:@F@__atomic_store")
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    const code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(int_type())},
      pointer_typet(empty_typet()));
    symbol_exprt result(identifier, t);
    return result;
  }
  else if(identifier == "c:@F@__atomic_exchange")
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    const code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(int_type())},
      pointer_typet(empty_typet()));
    symbol_exprt result(identifier, t);
    return result;
  }
  else if(
    identifier == "c:@F@__atomic_compare_exchange_n" ||
    identifier == "c:@F@__atomic_compare_exchange")
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    code_typet::argumentst parameters;
    parameters.push_back(code_typet::argumentt(ptr_arg.type()));
    parameters.push_back(code_typet::argumentt(ptr_arg.type()));

    if(identifier == "c:@F@__atomic_compare_exchange")
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
    identifier == "c:@F@__atomic_add_fetch" || identifier == "c:@F@__atomic_sub_fetch" ||
    identifier == "c:@F@__atomic_and_fetch" || identifier == "c:@F@__atomic_xor_fetch" ||
    identifier == "c:@F@__atomic_or_fetch" || identifier == "c:@F@__atomic_nand_fetch")
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
    identifier == "c:@F@__atomic_fetch_add" || identifier == "c:@F@__atomic_fetch_sub" ||
    identifier == "c:@F@__atomic_fetch_and" || identifier == "c:@F@__atomic_fetch_xor" ||
    identifier == "c:@F@__atomic_fetch_or" || identifier == "c:@F@__atomic_fetch_nand")
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

#include <clang-c-frontend/clang_c_adjust.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/base/prefix.h>
#include <util/expr/type2name.h>
#include <util/arith/arith_tools.h>

#include <algorithm>

static bool is_overflow_builtin(const irep_idt &identifier)
{
  const std::string &id = identifier.as_string();
  return has_prefix(id, "c:@F@__builtin_add_overflow") ||
         has_prefix(id, "c:@F@__builtin_sub_overflow") ||
         has_prefix(id, "c:@F@__builtin_mul_overflow");
}

/* One prefix per family: the suffixes (addcb, addcs, addc, addcl, addcll) all
 * follow. */
static bool is_carry_builtin(const irep_idt &identifier)
{
  const std::string &id = identifier.as_string();
  return has_prefix(id, "c:@F@__builtin_addc") ||
         has_prefix(id, "c:@F@__builtin_subc");
}

/* These two families differ from every other name handled here: they take their
 * result pointer last rather than first, are pure computation rather than
 * shared-memory access, and their parameters do not all share one type. */
static bool is_overflow_or_carry_builtin(const irep_idt &identifier)
{
  return is_overflow_builtin(identifier) || is_carry_builtin(identifier);
}

exprt clang_c_adjust::is_gcc_polymorphic_builtin(
  const irep_idt &identifier,
  const exprt::operandst &arguments)
{
  if (
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
  else if (is_overflow_builtin(identifier))
  {
    /* _Bool __builtin_<op>_overflow(T1 a, T2 b, T3 *res): the operands and
     * the result may all differ in type, and clang leaves these generic --
     * unlike the typed __builtin_sadd_overflow family, which it lowers
     * itself. https://gcc.gnu.org/onlinedocs/gcc/Integer-Overflow-Builtins.html
     *
     * Each operand keeps its own type: the operation is performed as if in
     * infinite precision, so converting them to the result type up front
     * would wrap away the very overflow being reported. */
    const exprt &res_arg = arguments.back();

    code_typet t{
      {code_typet::argumentt(arguments[0].type()),
       code_typet::argumentt(arguments[1].type()),
       code_typet::argumentt(res_arg.type())},
      bool_type()};
    t.make_ellipsis();
    symbol_exprt result{identifier, std::move(t)};
    return result;
  }
  else if (is_carry_builtin(identifier))
  {
    /* T __builtin_addc<suffix>(T a, T b, T carry_in, T *carry_out): all four
     * share one type, fixed by the suffix. Returns the modular sum; stores
     * whether either partial addition wrapped.
     * clang.llvm.org/docs/LanguageExtensions.html
     * #multiprecision-arithmetic-builtins */
    const exprt &carry_arg = arguments.back();
    const typet &value_type = to_pointer_type(carry_arg.type()).subtype();

    code_typet t{
      {code_typet::argumentt(value_type),
       code_typet::argumentt(value_type),
       code_typet::argumentt(value_type),
       code_typet::argumentt(carry_arg.type())},
      value_type};
    t.make_ellipsis();
    symbol_exprt result{identifier, std::move(t)};
    return result;
  }
  else if (
    has_prefix(identifier.as_string(), "c:@F@__sync_bool_compare_and_swap") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_val_compare_and_swap"))
  {
    // These are polymorphic, see
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html
    const exprt &ptr_arg = arguments.front();

    const typet &base_type = to_pointer_type(ptr_arg.type()).subtype();
    typet sync_return_type = base_type;
    if (has_prefix(identifier.as_string(), "c:@F@__sync_val_compare_and_swap"))
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
  else if (has_prefix(identifier.as_string(), "c:@F@__sync_lock_release"))
  {
    // This is polymorphic, see
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html
    const exprt &ptr_arg = arguments.front();

    code_typet t{{code_typet::argumentt(ptr_arg.type())}, empty_typet()};
    t.make_ellipsis();
    symbol_exprt result{identifier, std::move(t)};
    return result;
  }
  else if (has_prefix(identifier.as_string(), "c:@F@__atomic_load_n"))
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
  else if (has_prefix(identifier.as_string(), "c:@F@__atomic_store_n"))
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
  else if (has_prefix(identifier.as_string(), "c:@F@__atomic_exchange_n"))
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
  else if (
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
  else if (has_prefix(identifier.as_string(), "c:@F@__atomic_exchange"))
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
  else if (
    has_prefix(identifier.as_string(), "c:@F@__atomic_compare_exchange_n") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_compare_exchange"))
  {
    // These are polymorphic
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    const exprt &ptr_arg = arguments.front();

    code_typet::argumentst parameters;
    parameters.push_back(code_typet::argumentt(ptr_arg.type()));
    parameters.push_back(code_typet::argumentt(ptr_arg.type()));

    // __atomic_compare_exchange_n takes `desired` by value; the non-_n variant
    // takes a pointer. has_prefix("__atomic_compare_exchange") also matches
    // the _n suffix, so check the more specific name first.
    if (has_prefix(identifier.as_string(), "c:@F@__atomic_compare_exchange_n"))
      parameters.push_back(
        code_typet::argumentt(to_pointer_type(ptr_arg.type()).subtype()));
    else
      parameters.push_back(code_typet::argumentt(ptr_arg.type()));

    parameters.push_back(code_typet::argumentt(bool_type()));
    parameters.push_back(code_typet::argumentt(int_type()));
    parameters.push_back(code_typet::argumentt(int_type()));
    code_typet t(std::move(parameters), bool_type());
    symbol_exprt result(identifier, t);
    return result;
  }
  else if (
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
  else if (
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
  // The C11 <stdatomic.h> builtins are polymorphic in the atomic object's value
  // type, so they need per-type bodies like the GCC family above (issue #2174).
  // C11 7.17.7.4: `expected` is passed by pointer, `desired` by value.
  else if (has_prefix(
             identifier.as_string(), "c:@F@__c11_atomic_compare_exchange"))
  {
    const exprt &ptr_arg = arguments.front();
    const typet &value_type = to_pointer_type(ptr_arg.type()).subtype();

    code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(pointer_typet(value_type)),
       code_typet::argumentt(value_type),
       code_typet::argumentt(int_type()),
       code_typet::argumentt(int_type())},
      bool_type());
    symbol_exprt result(identifier, std::move(t));
    return result;
  }
  else if (has_prefix(identifier.as_string(), "c:@F@__c11_atomic_load"))
  {
    const exprt &ptr_arg = arguments.front();

    code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(int_type())},
      to_pointer_type(ptr_arg.type()).subtype());
    symbol_exprt result(identifier, std::move(t));
    return result;
  }
  else if (has_prefix(identifier.as_string(), "c:@F@__c11_atomic_store"))
  {
    const exprt &ptr_arg = arguments.front();

    code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(to_pointer_type(ptr_arg.type()).subtype()),
       code_typet::argumentt(int_type())},
      empty_typet());
    symbol_exprt result(identifier, std::move(t));
    return result;
  }
  else if (has_prefix(identifier.as_string(), "c:@F@__c11_atomic_init"))
  {
    // C11 7.17.2.2: atomic_init takes no memory-order operand.
    const exprt &ptr_arg = arguments.front();

    code_typet t(
      {code_typet::argumentt(ptr_arg.type()),
       code_typet::argumentt(to_pointer_type(ptr_arg.type()).subtype())},
      empty_typet());
    symbol_exprt result(identifier, std::move(t));
    return result;
  }
  else if (
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_exchange") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_add") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_sub") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_and") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_or") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_xor"))
  {
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
  symbol.set_type(type);

  context.add(symbol);

  return symbol;
}

static void convert_expression_to_code(exprt &expr)
{
  if (expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

code_blockt clang_c_adjust::instantiate_gcc_polymorphic_builtin(
  const irep_idt &identifier,
  const symbol_exprt &function_symbol,
  contextt &context)
{
  const irep_idt &identifier_with_type = function_symbol.get_identifier();
  const code_typet &code_type = to_code_type(function_symbol.type());

  code_blockt block;

  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  block.operands().push_back(label);

  /* The overflow and carry builtins are pure computation on their arguments:
   * nothing is shared, so wrapping them in an atomic scope would only add an
   * unmatched ATOMIC_BEGIN (their arms emit no atomic_end). Every other arm
   * here reads and writes memory another thread can observe. */
  if (!is_overflow_or_carry_builtin(identifier))
  {
    // atomic scope begin
    side_effect_expr_function_callt atomic_begin;
    atomic_begin.function() = symbol_exprt("c:@F@__ESBMC_atomic_begin");
    convert_expression_to_code(atomic_begin);
    block.operands().push_back(atomic_begin);
  }

  // Change the cex to show that these code comes from the atomic/sync
  locationt new_loc = function_symbol.location();
  new_loc.set_function(function_symbol.name());

  if (
    has_prefix(identifier.as_string(), "c:@F@__sync_add_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_sub_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_or_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_and_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_xor_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_nand_and_fetch"))
  {
    // TODO
  }
  else if (
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_add") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_sub") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_or") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_and") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_xor") ||
    has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_nand") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_add") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_sub") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_or") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_and") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_xor") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_nand") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_add") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_sub") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_or") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_and") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_xor"))
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
    if (
      has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_add") ||
      has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_add") ||
      has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_add"))
    {
      if (type.is_floatbv())
        new_expr = exprt("ieee_add", type);
      else
        new_expr = exprt("+", type);
    }
    else if (
      has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_sub") ||
      has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_sub") ||
      has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_sub"))
    {
      if (type.is_floatbv())
        new_expr = exprt("ieee_sub", type);
      else
        new_expr = exprt("-", type);
    }
    else if (
      has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_or") ||
      has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_or") ||
      has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_or"))
    {
      new_expr = exprt("bitor", type);
    }
    else if (
      has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_and") ||
      has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_and") ||
      has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_and"))
    {
      new_expr = exprt("bitand", type);
    }
    else if (
      has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_xor") ||
      has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_xor") ||
      has_prefix(identifier.as_string(), "c:@F@__c11_atomic_fetch_xor"))
    {
      new_expr = exprt("bitxor", type);
    }
    else if (
      has_prefix(identifier.as_string(), "c:@F@__sync_fetch_and_nand") ||
      has_prefix(identifier.as_string(), "c:@F@__atomic_fetch_nand"))
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
  else if (is_overflow_builtin(identifier))
  {
    /* GCC performs the operation "as if" in infinite precision and reports
     * whether the exact result fits the type *res points at; *res always
     * receives that exact result truncated to its own type. So the operation
     * happens in a type wide enough for it to be exact, and only the fit into
     * the result type is the overflow being reported -- not any wrapping of
     * the operands.
     * https://gcc.gnu.org/onlinedocs/gcc/Integer-Overflow-Builtins.html */
    const code_typet::argumentst &args = code_type.arguments();
    const typet &res_type = to_pointer_type(args[2].type()).subtype();

    const exprt a(symbol_exprt(args[0].cmt_identifier(), args[0].type()));
    const exprt b(symbol_exprt(args[1].cmt_identifier(), args[1].type()));
    const exprt res_ptr(symbol_exprt(args[2].cmt_identifier(), args[2].type()));

    std::string op = "+";
    if (has_prefix(identifier.as_string(), "c:@F@__builtin_sub_overflow"))
      op = "-";
    else if (has_prefix(identifier.as_string(), "c:@F@__builtin_mul_overflow"))
      op = "*";

    /* Signed, and wide enough that neither the operands nor the exact result
     * can wrap: a product needs the two operand widths summed, a sum or
     * difference one more than the wider operand. The result type joins the
     * max so that truncating to it is the only narrowing, and the final +1
     * carries an unsigned value's top bit into the signed type. */
    const std::size_t w0 = bv_width(args[0].type());
    const std::size_t w1 = bv_width(args[1].type());
    const std::size_t operand_width =
      op == "*" ? w0 + w1 : std::max(w0, w1) + 1;
    const std::size_t exact_width =
      std::max(operand_width, std::size_t(bv_width(res_type))) + 1;

    const typet exact_type = signedbv_typet(exact_width);

    exprt wide_a("typecast", exact_type);
    wide_a.copy_to_operands(a);
    exprt wide_b("typecast", exact_type);
    wide_b.copy_to_operands(b);

    exprt exact(op, exact_type);
    exact.copy_to_operands(wide_a, wide_b);
    exact.location() = new_loc;

    /* clang accepts a `_Bool *` result, and stores the exact value truncated
     * to one bit: 1 + 1 stores 0, not the 1 a C cast to _Bool would give. Go
     * through a 1-bit unsigned to reproduce that -- casting straight to bool
     * tests against zero instead. */
    const bool res_is_bool = res_type.id() == typet::t_bool;

    exprt value("typecast", res_type);
    if (res_is_bool)
    {
      exprt one_bit("typecast", unsignedbv_typet(1));
      one_bit.copy_to_operands(exact);
      one_bit.location() = new_loc;
      value.copy_to_operands(one_bit);
    }
    else
      value.copy_to_operands(exact);
    value.location() = new_loc;

    code_assignt store(dereference_exprt(res_ptr, args[2].type()), value);
    store.location() = new_loc;
    block.operands().push_back(store);

    /* The reported condition is exactly "the exact result is outside the
     * range of the result type". overflow-typecast- cannot express it: its
     * lowering tests [0, 2^N) regardless of the destination's signedness.
     * bool_typet carries no width, so bv_width would report 0 here and make
     * the range [0, 0] -- every non-zero result an overflow. */
    const std::size_t res_width = res_is_bool ? 1 : bv_width(res_type);
    const bool res_signed = res_type.id() == typet::t_signedbv;
    const BigInt lo = res_signed ? -BigInt::power2(res_width - 1) : BigInt(0);
    const BigInt hi =
      (res_signed ? BigInt::power2(res_width - 1) : BigInt::power2(res_width)) -
      1;

    exprt below("<", bool_type());
    below.copy_to_operands(exact, from_integer(lo, exact_type));
    exprt above(">", bool_type());
    above.copy_to_operands(exact, from_integer(hi, exact_type));

    exprt did_overflow("or", bool_type());
    did_overflow.copy_to_operands(below, above);
    did_overflow.location() = new_loc;

    code_returnt ret;
    ret.return_value() = did_overflow;
    ret.location() = new_loc;
    block.operands().push_back(ret);
  }
  else if (is_carry_builtin(identifier))
  {
    /* sum = (a <op> b) <op> carry_in, wrapping; *carry_out is set when either
     * partial step wrapped. Both steps need their own predicate: a+b may fit
     * and adding the carry then overflow, or the reverse for subtraction. */
    const code_typet::argumentst &args = code_type.arguments();
    const typet &value_type = args[0].type();
    const bool is_add =
      has_prefix(identifier.as_string(), "c:@F@__builtin_addc");
    const std::string op = is_add ? "+" : "-";

    const exprt a(symbol_exprt(args[0].cmt_identifier(), value_type));
    const exprt b(symbol_exprt(args[1].cmt_identifier(), value_type));
    const exprt cin(symbol_exprt(args[2].cmt_identifier(), value_type));
    const exprt cout_ptr(
      symbol_exprt(args[3].cmt_identifier(), args[3].type()));

    exprt partial(op, value_type);
    partial.copy_to_operands(a, b);
    partial.location() = new_loc;

    exprt sum(op, value_type);
    sum.copy_to_operands(partial, cin);
    sum.location() = new_loc;

    exprt ov1("overflow-" + op, bool_type());
    ov1.copy_to_operands(a, b);
    exprt ov2("overflow-" + op, bool_type());
    ov2.copy_to_operands(partial, cin);

    exprt carry("or", bool_type());
    carry.copy_to_operands(ov1, ov2);
    carry.location() = new_loc;

    /* The carry is 1 or 0 in the operand type, not a _Bool. */
    exprt carry_value("typecast", value_type);
    carry_value.copy_to_operands(carry);
    carry_value.location() = new_loc;

    code_assignt store(
      dereference_exprt(cout_ptr, args[3].type()), carry_value);
    store.location() = new_loc;
    block.operands().push_back(store);

    code_returnt ret;
    ret.return_value() = sum;
    ret.location() = new_loc;
    block.operands().push_back(ret);
  }
  else if (has_prefix(
             identifier.as_string(), "c:@F@__sync_bool_compare_and_swap"))
  {
    // TODO
  }
  else if (has_prefix(
             identifier.as_string(), "c:@F@__sync_val_compare_and_swap"))
  {
    // TODO
  }
  else if (has_prefix(identifier.as_string(), "c:@F@__sync_lock_release"))
  {
    // TODO
  }
  else if (has_prefix(identifier.as_string(), "c:@F@__sync_lock_test_and_set"))
  {
    // TODO
  }
  else if (
    has_prefix(identifier.as_string(), "c:@F@__atomic_load_n") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_load"))
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
  else if (
    has_prefix(identifier.as_string(), "c:@F@__atomic_store_n") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_store") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_init"))
  {
    // atomic_init is non-atomic per C11 7.17.2.2, but racing on the object
    // being initialised is UB, so the atomic lock masks no defined behaviour.
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
  else if (
    has_prefix(identifier.as_string(), "c:@F@__atomic_exchange_n") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_exchange"))
  {
    // This atomic builtin follows GCC's __atomic built-in functions
    // specification. See
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html.
    const typet &type = code_type.return_type();

    const exprt &result =
      symbol_expr(result_symbol(identifier_with_type, type, context));

    code_declt decl(result);
    block.operands().push_back(decl);

    code_typet::argumentt arg0 = code_type.arguments()[0];
    code_typet::argumentt arg1 = code_type.arguments()[1];

    dereference_exprt arg0_deref(
      symbol_exprt(arg0.cmt_identifier(), arg0.type()), arg0.type());

    // Store old value in result
    code_assignt assign_old(result, arg0_deref);
    assign_old.location() = new_loc;
    block.operands().push_back(assign_old);

    // Store new value at pointer location
    code_assignt assign_new(
      arg0_deref, symbol_exprt(arg1.cmt_identifier(), arg1.type()));
    assign_new.location() = new_loc;
    block.operands().push_back(assign_new);

    // atomic scope end
    side_effect_expr_function_callt atomic_end;
    atomic_end.function() = symbol_exprt("c:@F@__ESBMC_atomic_end");
    convert_expression_to_code(atomic_end);
    block.operands().push_back(atomic_end);

    // Return old value
    code_returnt ret;
    ret.return_value() = result;
    ret.location() = new_loc;
    block.operands().push_back(ret);
  }
  else if (has_prefix(identifier.as_string(), "c:@F@__atomic_load"))
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
  else if (has_prefix(identifier.as_string(), "c:@F@__atomic_store"))
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
  else if (has_prefix(identifier.as_string(), "c:@F@__atomic_exchange"))
  {
    // TODO
  }
  else if (
    has_prefix(identifier.as_string(), "c:@F@__atomic_compare_exchange_n") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_compare_exchange") ||
    has_prefix(identifier.as_string(), "c:@F@__c11_atomic_compare_exchange"))
  {
    // GCC __atomic_compare_exchange{,_n} - strong CAS modelled atomically.
    // See https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html.
    //
    //   bool __atomic_compare_exchange_n(T *ptr, T *expected, T desired,
    //                                    bool weak, int success_mo,
    //                                    int failure_mo);
    //   bool __atomic_compare_exchange  (T *ptr, T *expected, T *desired,
    //                                    bool weak, int success_mo,
    //                                    int failure_mo);
    //
    // Modelled (under __ESBMC_atomic_begin/_end) as:
    //   result = (*ptr == *expected);
    //   if (result) *ptr = desired_value; else *expected = *ptr;
    //   return result;
    //
    // We treat the CAS as strong regardless of `weak` (a weak CAS that can
    // spuriously fail is a sound under-approximation of strong, but libvsync
    // - the motivating consumer - issues only strong CASes). C11's
    // compare_exchange_weak gets the same treatment, and passes `desired` by
    // value like the GCC _n variant.
    const bool desired_by_value =
      has_prefix(identifier.as_string(), "c:@F@__atomic_compare_exchange_n") ||
      has_prefix(identifier.as_string(), "c:@F@__c11_atomic_compare_exchange");

    const typet &ret_type = code_type.return_type();
    const exprt &result =
      symbol_expr(result_symbol(identifier_with_type, ret_type, context));

    code_declt decl(result);
    block.operands().push_back(decl);

    code_typet::argumentt arg0 = code_type.arguments()[0];
    code_typet::argumentt arg1 = code_type.arguments()[1];
    code_typet::argumentt arg2 = code_type.arguments()[2];

    dereference_exprt ptr_deref(
      symbol_exprt(arg0.cmt_identifier(), arg0.type()), arg0.type());
    dereference_exprt exp_deref(
      symbol_exprt(arg1.cmt_identifier(), arg1.type()), arg1.type());

    exprt desired = symbol_exprt(arg2.cmt_identifier(), arg2.type());
    if (!desired_by_value)
      desired = dereference_exprt(desired, arg2.type());

    exprt eq("=", ret_type);
    eq.copy_to_operands(ptr_deref, exp_deref);
    code_assignt assign_result(result, eq);
    assign_result.location() = new_loc;
    block.operands().push_back(assign_result);

    code_ifthenelset cond;
    cond.cond() = result;

    code_assignt assign_ptr(ptr_deref, desired);
    assign_ptr.location() = new_loc;
    cond.then_case() = assign_ptr;

    code_assignt assign_exp(exp_deref, ptr_deref);
    assign_exp.location() = new_loc;
    cond.else_case() = assign_exp;

    block.operands().push_back(cond);

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
  else if (
    has_prefix(identifier.as_string(), "c:@F@__atomic_add_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_sub_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_and_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_xor_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_or_fetch") ||
    has_prefix(identifier.as_string(), "c:@F@__atomic_nand_fetch"))
  {
    // TODO
  }

  return block;
}

exprt clang_c_adjust::declare_gcc_polymorphic_builtin(
  const symbol_exprt &callee,
  const exprt::operandst &arguments,
  const locationt &call_location,
  contextt &context)
{
  const irep_idt &identifier = callee.identifier();
  // The prefix these names are matched on is not reserved to the builtin, so
  // an ordinary user function reaches here. Every arm binds a pointer argument
  // and dereferences it, which such a call need not supply: with no argument
  // this segfaulted, and with a non-pointer one it tripped to_pointer_type's
  // assertion -- and under NDEBUG cast a non-pointer instead.
  //
  // The atomic/sync arms take that pointer first; the overflow and carry
  // builtins take it last, so require whichever the name implies rather than
  // only the front one.
  if (arguments.empty())
    return nil_exprt();

  // The arms below index their parameters directly, so the arity each name
  // implies is a precondition, not something to discover mid-arm. A user
  // function whose name merely shares one of these prefixes -- which compiles
  // without a diagnostic -- reaches here with whatever arity it was declared
  // with, and master got this floor for free by having no arm to fall into.
  const std::size_t required_arity = is_overflow_builtin(identifier) ? 3
                                     : is_carry_builtin(identifier)  ? 4
                                                                     : 1;
  if (arguments.size() < required_arity)
    return nil_exprt();

  const exprt &pointer_arg = is_overflow_or_carry_builtin(identifier)
                               ? arguments.back()
                               : arguments.front();
  if (!pointer_arg.type().is_pointer())
    return nil_exprt();

  exprt poly = is_gcc_polymorphic_builtin(identifier, arguments);
  if (poly.is_nil())
    return nil_exprt();

  auto &poly_args = to_code_type(poly.type()).arguments();

  // Every parameter type goes into the name. For the atomic/sync built-ins
  // the first one would suffice -- they share a single type -- but the
  // overflow builtins take three independently-typed parameters, and keying
  // on the first alone would hand every call the first instantiation's
  // result type.
  std::string identifier_with_type = id2string(identifier);
  for (const auto &arg : poly_args)
  {
    const typet &t = arg.type();
    identifier_with_type +=
      "_" + type2name(t.is_pointer() ? to_pointer_type(t).subtype() : t);
  }

  poly.identifier(identifier_with_type);
  poly.name(callee.name());
  poly.location() = call_location;

  if (!context.find_symbol(identifier_with_type))
  {
    for (std::size_t i = 0; i < poly_args.size(); ++i)
    {
      const std::string param_name = "p_" + std::to_string(i);

      // TODO: Just like the function parameter symbols in
      // clang_c_convertert::get_function_param, adding this symbol to the
      // context is only necessary for the migrate code.
      symbolt param_symbol;
      param_symbol.id = id2string(identifier_with_type) + "::" + param_name;
      param_symbol.name = param_name;
      param_symbol.location = callee.location();
      param_symbol.set_type(poly_args[i].type());
      param_symbol.lvalue = true;
      param_symbol.is_parameter = true;
      param_symbol.file_local = true;

      poly_args[i].cmt_identifier(param_symbol.id);
      poly_args[i].cmt_base_name(param_symbol.name);

      context.add(param_symbol);
    }

    symbolt new_symbol;
    new_symbol.id = identifier_with_type;
    new_symbol.name = callee.name();
    new_symbol.location = call_location;
    new_symbol.set_type(poly.type());
    new_symbol.set_value(instantiate_gcc_polymorphic_builtin(
      identifier, to_symbol_expr(poly), context));

    context.add(new_symbol);
  }

  return poly;
}

#include <cassert>
#include <goto-symex/goto_symex.h>
#include <string>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <irep2/irep2.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/std_types.h>
#include <algorithm>

void goto_symext::bump_call(
  const code_function_call2t &func_call,
  const std::string &symname)
{
  // We're going to execute a function call, and that's going to mess with
  // the program counter. Set it back *onto* pointing at this intrinsic, so
  // symex_function_call calculates the right return address. Misery.
  cur_state->source.pc--;

  expr2tc newcall = func_call.clone();
  code_function_call2t &mutable_funccall = to_code_function_call2t(newcall);
  mutable_funccall.function = symbol2tc(get_empty_type(), symname);
  // Execute call
  symex_function_call(newcall);
  return;
}

// Copied from https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
static inline bool
ends_with(std::string const &value, std::string const &ending)
{
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

bool goto_symext::run_builtin(
  const code_function_call2t &func_call,
  const std::string &symname)
{
  if (
    has_prefix(symname, "c:@F@__builtin_sadd") ||
    has_prefix(symname, "c:@F@__builtin_uadd") ||
    has_prefix(symname, "c:@F@__builtin_ssub") ||
    has_prefix(symname, "c:@F@__builtin_usub") ||
    has_prefix(symname, "c:@F@__builtin_smul") ||
    has_prefix(symname, "c:@F@__builtin_umul"))
  {
    assert(ends_with(symname, "_overflow"));
    assert(func_call.operands.size() == 3);

    const auto &func_type = to_code_type(func_call.function->type);
    assert(func_type.arguments[0] == func_type.arguments[1]);
    assert(is_pointer_type(func_type.arguments[2]));

    bool is_mult = has_prefix(symname, "c:@F@__builtin_smul") ||
                   has_prefix(symname, "c:@F@__builtin_umul");
    bool is_add = has_prefix(symname, "c:@F@__builtin_sadd") ||
                  has_prefix(symname, "c:@F@__builtin_uadd");
    bool is_sub = has_prefix(symname, "c:@F@__builtin_ssub") ||
                  has_prefix(symname, "c:@F@__builtin_usub");

    expr2tc op;
    if (is_mult)
      op = mul2tc(
        func_type.arguments[0], func_call.operands[0], func_call.operands[1]);
    else if (is_add)
      op = add2tc(
        func_type.arguments[0], func_call.operands[0], func_call.operands[1]);
    else if (is_sub)
      op = sub2tc(
        func_type.arguments[0], func_call.operands[0], func_call.operands[1]);
    else
    {
      log_error("Unknown overflow intrinsics");
      abort();
    }

    // Perform overflow check and assign it to the return object
    if (!is_nil_expr(func_call.ret))
      symex_assign(code_assign2tc(func_call.ret, overflow2tc(op)));

    // Assign result of the two arguments to the dereferenced third argument
    symex_assign(code_assign2tc(
      dereference2tc(
        to_pointer_type(func_call.operands[2]->type).subtype,
        func_call.operands[2]),
      op));

    return true;
  }

  if (has_prefix(symname, "c:@F@__builtin_constant_p"))
  {
    expr2tc op1 = func_call.operands[0];
    cur_state->rename(op1);
    if (!is_nil_expr(func_call.ret))
      symex_assign(code_assign2tc(
        func_call.ret,
        is_constant_int2t(op1) ? gen_one(int_type2()) : gen_zero(int_type2())));
    return true;
  }

  // __builtin_clz / __builtin_clzl / __builtin_clzll: count leading zero bits.
  // One handler covers all widths — the operand type fixes the bit width. The
  // result is undefined for a zero argument (matching GCC), reported as UB.
  // Match the three names exactly: a loose "__builtin_clz" prefix would also
  // capture __builtin_clzs (16-bit) and the two-argument __builtin_clzg, the
  // latter tripping the one-argument assertion below. See #4606.
  if (
    symname == "c:@F@__builtin_clz" || symname == "c:@F@__builtin_clzl" ||
    symname == "c:@F@__builtin_clzll")
  {
    assert(
      func_call.operands.size() == 1 &&
      "__builtin_clz* must have one argument");

    expr2tc arg = func_call.operands[0];
    expr2tc ret = func_call.ret;

    const type2tc &t = arg->type;
    const unsigned width = t->get_width();

    expr2tc zero = constant_int2tc(t, 0);
    expr2tc one = constant_int2tc(t, 1);
    expr2tc upper = constant_int2tc(t, width - 1);

    claim(notequal2tc(arg, zero), "__builtin_clz: UB for x equal to 0");

    // Introduce a nondet symbolic variable clz_sym for the number of leading
    // zeros, constrained to 0 <= clz_sym <= width - 1.
    unsigned int &nondet_count = get_nondet_counter();
    expr2tc clz_sym = symbol2tc(t, "nondet$symex::" + i2string(nondet_count++));

    expr2tc ge = greaterthanequal2tc(clz_sym, zero);
    expr2tc le = lessthanequal2tc(clz_sym, upper);
    assume(and2tc(ge, le));

    // idx = (width - 1) - clz_sym is the position of the most-significant set
    // bit. Force that bit to 1 and every bit above it to 0.
    expr2tc idx = sub2tc(t, upper, clz_sym);

    expr2tc shift = lshr2tc(t, arg, idx);
    expr2tc bit1 = bitand2tc(t, shift, one);
    assume(notequal2tc(bit1, zero));

    expr2tc next = add2tc(t, idx, one);
    expr2tc shift2 = lshr2tc(t, arg, next);
    assume(equality2tc(shift2, zero));

    if (!is_nil_expr(ret))
      symex_assign(code_assign2tc(ret, typecast2tc(ret->type, clz_sym)));

    return true;
  }

  return false;
}

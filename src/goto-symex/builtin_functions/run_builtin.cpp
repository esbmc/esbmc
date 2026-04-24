#include <cassert>
#include <complex>
#include <functional>
#include <goto-symex/execution_state.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/printf_formatter.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <irep2/irep2.h>
#include <util/message.h>
#include <util/message/format.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/std_types.h>
#include <vector>
#include <algorithm>
#include <util/array2string.h>

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

  if (has_prefix(symname, "c:@F@__builtin_clzll"))
  {
    assert(
      func_call.operands.size() == 1 &&
      "__builtin_clzll must have one argument");

    expr2tc arg = func_call.operands[0];
    expr2tc ret = func_call.ret;

    expr2tc zero = constant_int2tc(get_uint64_type(), 0);
    expr2tc one = constant_int2tc(get_uint64_type(), 1);
    expr2tc upper = constant_int2tc(get_uint64_type(), 63);

    claim(notequal2tc(arg, zero), "__builtin_clzll: UB for x equal to 0");

    // Introduce a nondet symbolic variable clz_sym to stand for the number of leading zeros
    unsigned int &nondet_count = get_nondet_counter();
    expr2tc clz_sym =
      symbol2tc(get_uint64_type(), "nondet$symex::" + i2string(nondet_count++));

    // Constrain the range 0 <= clz_sym <= 63
    expr2tc ge = greaterthanequal2tc(clz_sym, zero);
    expr2tc le = lessthanequal2tc(clz_sym, upper);
    expr2tc in_range = and2tc(ge, le);
    assume(in_range);

    // This idx is the bit‐position where the first 1 should occur.
    // 63 - clz_sym
    expr2tc idx = sub2tc(get_uint64_type(), upper, clz_sym);

    // Shifting arg right by idx
    // Masking with & 1 to extract single bit
    // ((x >> idx) & 1) != 0
    expr2tc shift = lshr2tc(get_uint64_type(), arg, idx);
    expr2tc bit1 = bitand2tc(get_uint64_type(), shift, one);
    expr2tc is_one = notequal2tc(bit1, zero);
    assume(is_one);

    // Requiring (x >> (idx + 1)) == 0 forces every bit from idx + 1 up
    // to bit 63 to be zero, All bits above index idx must be 0
    // (x >> (idx+1)) == 0
    expr2tc next = add2tc(get_uint64_type(), idx, one);
    expr2tc shift2 = lshr2tc(get_uint64_type(), arg, next);
    expr2tc above_zero = equality2tc(shift2, zero);
    assume(above_zero);

    if (!is_nil_expr(ret))
      symex_assign(code_assign2tc(ret, typecast2tc(ret->type, clz_sym)));

    return true;
  }

  return false;
}

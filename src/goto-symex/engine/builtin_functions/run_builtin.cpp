#include <cassert>
#include <goto-symex/engine/goto_symex.h>
#include <string>
#include <util/arith/arith_tools.h>
#include <util/lang/c_builtins.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <irep2/irep2.h>
#include <util/message/message.h>
#include <util/irep/migrate.h>
#include <util/base/prefix.h>
#include <util/irep/std_types.h>
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

// Copied from
// https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
static inline bool
ends_with(std::string const &value, std::string const &ending)
{
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

/// Value of a __builtin_clz*/ctz*/ffs*/clrsb* call. One encoding covers every
/// spelling: the operand type fixes the bit width, and the directions differ
/// only in which way the smear shifts. Zero is undefined for every form but the
/// two-argument clzg/ctzg, ffs and clrsb; the optional UB assertion is added in
/// goto-check (--clz-zero-check), with the other UB checks. See #4606, #6925,
/// #183.
static expr2tc
build_bit_scan(const code_function_call2t &func_call, bit_scan_endt end)
{
  const expr2tc &arg = func_call.operands[0];
  const type2tc &t = arg->type;
  const unsigned width = t->get_width();
  const bool leading =
    end == bit_scan_endt::leading || end == bit_scan_endt::redundant_sign;

  // clz(x) = width - popcount(x with every bit below the most-significant set
  // bit smeared down); ctz mirrors it, smearing up from the least-significant
  // set bit. Reusing the popcount irep means a constant argument folds to a
  // constant (popcount has a simplifier), while a symbolic argument is handled
  // exactly by the backend's popcount encoding.
  // clrsb(x) is clz(x ^ (x >> (width - 1))) - 1: the xor clears every
  // leading copy of the sign bit, and a zero result gives width - 1.
  expr2tc smeared = arg;
  if (end == bit_scan_endt::redundant_sign)
    smeared =
      bitxor2tc(t, arg, ashr2tc(t, arg, constant_int2tc(t, BigInt(width - 1))));
  for (unsigned shift = 1; shift < width; shift <<= 1)
  {
    expr2tc offset = constant_int2tc(t, shift);
    smeared = bitor2tc(
      t,
      smeared,
      leading ? lshr2tc(t, smeared, offset) : shl2tc(t, smeared, offset));
  }

  expr2tc count = sub2tc(
    get_int32_type(),
    constant_int2tc(get_int32_type(), width),
    popcount2tc(smeared));

  if (end == bit_scan_endt::redundant_sign)
    count = sub2tc(
      get_int32_type(), count, constant_int2tc(get_int32_type(), BigInt(1)));

  // ffs counts the same trailing zeros but reports a one-based index, and is
  // defined at zero as 0 rather than left undefined there (POSIX).
  if (end == bit_scan_endt::first_set)
    count = if2tc(
      get_int32_type(),
      equality2tc(arg, gen_zero(t)),
      gen_zero(get_int32_type()),
      add2tc(
        get_int32_type(), count, constant_int2tc(get_int32_type(), BigInt(1))));

  // The second argument of clzg/ctzg is the result at zero.
  if (func_call.operands.size() == 2)
    count = if2tc(
      get_int32_type(),
      equality2tc(arg, gen_zero(t)),
      typecast2tc(get_int32_type(), func_call.operands[1]),
      count);

  return count;
}

/// Value of a __builtin_rotateleft{8,16,32,64}/__builtin_rotateright* call.
/// The distance is taken modulo the width, and so is its complement, so a zero
/// distance gives (x << 0) | (x >> 0) == x rather than a full-width shift.
static expr2tc build_rotate(const code_function_call2t &func_call, bool left)
{
  const expr2tc &x = func_call.operands[0];
  const type2tc &t = x->type;
  const unsigned width = t->get_width();
  const expr2tc mask = constant_int2tc(t, width - 1);
  const expr2tc d = bitand2tc(t, typecast2tc(t, func_call.operands[1]), mask);
  const expr2tc co =
    bitand2tc(t, sub2tc(t, constant_int2tc(t, width), d), mask);
  return left ? bitor2tc(t, shl2tc(t, x, d), lshr2tc(t, x, co))
              : bitor2tc(t, lshr2tc(t, x, d), shl2tc(t, x, co));
}

/// Value of a bit-counting or rotating builtin call, or nil if \p symname is
/// not one.
static expr2tc build_bit_builtin(
  const code_function_call2t &func_call,
  const std::string &symname)
{
  if (
    const bit_scan_endt end = bit_scan_builtin(symname);
    end != bit_scan_endt::none)
  {
    assert(
      !func_call.operands.empty() && func_call.operands.size() <= 2 &&
      "__builtin_clz*/__builtin_ctz* take one or two arguments");
    return build_bit_scan(func_call, end);
  }

  for (const bool left : {true, false})
    for (const char *width : {"8", "16", "32", "64"})
      if (
        symname == std::string("c:@F@__builtin_rotate") +
                     (left ? "left" : "right") + width)
        return build_rotate(func_call, left);

  return expr2tc();
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

  if (
    const expr2tc value = build_bit_builtin(func_call, symname);
    !is_nil_expr(value))
  {
    const expr2tc &ret = func_call.ret;
    if (!is_nil_expr(ret))
      symex_assign(code_assign2tc(ret, typecast2tc(ret->type, value)));

    return true;
  }

  // va_start/va_copy are kept in the GOTO program purely so that symex can
  // track which va_lists have been initialised, and where each reads; a
  // va_arg on an unstarted va_list is then flagged in symex_va_arg.
  if (symname == "c:@F@__builtin_va_start" && !func_call.operands.empty())
  {
    va_list_mark_started(
      func_call.operands[0], true, cur_state->top().va_index);
    return true;
  }

  if (symname == "c:@F@__builtin_va_copy" && func_call.operands.size() == 2)
  {
    va_list_copy(func_call.operands[0], func_call.operands[1]);
    return true;
  }

  return false;
}

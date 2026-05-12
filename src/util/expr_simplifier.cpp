#include <limits>

#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/expr_reassociate.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <irep2/irep2_utils.h>
#include <util/type_byte_size.h>

expr2tc expr2t::do_simplify() const
{
  return expr2tc();
}

expr2tc expr2t::simplify() const
{
  return simplify(/*suppress_reassoc=*/false);
}

expr2tc expr2t::simplify(bool suppress_reassoc) const
{
  // suppress_reassoc here is the cascading umbrella set by the public
  // API caller (e.g. simplify_no_reassoc): true means peephole-only
  // walk over the whole subtree, false means normal simplification.
  // It propagates unchanged into recursive simplify() calls below.
  try
  {
    // Corner case! Don't even try to simplify address of's operands, might end up
    // taking the address of some /completely/ arbitary pice of data, by
    // simplifiying an index to its data, discarding the symbol.
    if (expr_id == address_of_id) // unlikely
      return expr2tc();

    // And overflows too. We don't wish an add to distribute itself, for example,
    // when we're trying to work out whether or not it's going to overflow.
    if (expr_id == overflow_id)
      return expr2tc();

    // Short-circuit pre-pass for and/or/if. do_simplify() runs only the
    // node-local peepholes — it never recurses into operands — so calling
    // it before the operand walk is cheap. Two wins:
    //   - decisive-constant folds (false && X, true || X, if(true,x,_),
    //     if(false,_,y)) discharge without walking the dead operand;
    //   - if the dead arm contains a dyn_sized_array_excp, the catch
    //     below would otherwise return nil for the whole simplify.
    // Algebraic rules that fire here (e.g. if(c,x,x), x&&x, absorption)
    // can return a result whose sub-expressions still need
    // canonicalization. The recursive simplify() below handles that.
    // For arithmetic/bitwise nodes the early peepholes need canonical
    // operands first, so we keep this restricted to and/or/if.
    if (expr_id == and_id || expr_id == or_id || expr_id == if_id)
    {
      expr2tc shortcut = do_simplify();
      if (!is_nil_expr(shortcut))
      {
        expr2tc resimp = shortcut->simplify(suppress_reassoc);
        return is_nil_expr(resimp) ? shortcut : resimp;
      }
    }

    // Step 1: simplify all sub-operands first. This way do_simplify()
    // always sees canonical operands and its peepholes can match nested
    // patterns without a second-shot retry.
    //
    // Pass our own suppress_reassoc unchanged into operand simplify().
    // An earlier version OR'd in a "same-chain kind" bit that would tell
    // a child of the same chain (e.g. add inside add) to skip its own
    // chain-root reassoc step, on the theory that our step 4 would fold
    // its chain into ours anyway. That was a perf-only optimization but
    // had a soundness leak: the same-chain bit propagated through that
    // child's recursive simplify() into its grandchildren, so an
    // unrelated nested chain kind (e.g. a mul under our add child) saw
    // suppress_reassoc=true and never canonicalized. Per Codex review
    // 26: drop the same-chain skip; the redundant reassoc inside a
    // same-chain child is locally bounded and produces a canonicalized
    // sub-chain that our step 4 then merges. Correctness over a small
    // amount of duplicated work.
    bool changed = false;
    std::list<expr2tc> newoperands;

    for (size_t idx = 0; idx < get_num_sub_exprs(); idx++)
    {
      expr2tc tmp;
      const expr2tc *e = get_sub_expr(idx);

      if (expr_id == with_id && idx == 0 && is_with2t(*e))
      {
        // Don't simplify the first operand of a with-of-with: it's already
        // been simplified at construction time.
        newoperands.push_back(tmp);
        continue;
      }

      if (!is_nil_expr(*e))
      {
        tmp = (*e)->simplify(suppress_reassoc);
        if (!is_nil_expr(tmp))
          changed = true;
      }

      newoperands.push_back(tmp);
    }

    // Step 2: build the "current" form of the expression — either the
    // original (if no operand changed) or a clone with rewritten operands.
    // do_simplify() runs on whichever of those we end up with.
    expr2tc current;
    if (changed)
    {
      current = clone();
      std::list<expr2tc>::iterator it = newoperands.begin();
      current->Foreach_operand([&it](expr2tc &e) {
        if (*it)
          e = *it;
        ++it;
      });
    }

    // Step 3: top-level peephole. If do_simplify() returned something
    // structurally different, recurse once on the result so any new
    // sub-expressions it introduced get simplified too.
    expr2tc result;
    expr2tc top = changed ? current->do_simplify() : do_simplify();
    if (!is_nil_expr(top))
    {
      // Pass our own suppress_reassoc through: if a peephole rewrote an
      // add into a sub (or vice versa), that result is still subject to
      // whatever suppression we were given.
      expr2tc top2 = top->simplify(suppress_reassoc);
      result = is_nil_expr(top2) ? top : top2;
    }
    else if (changed)
    {
      // No top-level rewrite, but operands changed.
      result = current;
    }

    // Step 4: chain-root reassociation. Fires only when the caller didn't
    // suppress reassoc and the result is still a chain root (add/sub/neg
    // for arith). The recursive resimplify above means peephole-introduced
    // chain roots are also caught here.
    //
    // Gate on expr_id BEFORE cloning. Most expressions in the codebase
    // aren't chain roots, and the clone(): result fallback was paying for
    // that path on every simplify() call.
    if (!suppress_reassoc)
    {
      // Use the post-step-3 result if it exists; otherwise the original
      // expr_id is unchanged. Avoid clone() until we know we'll mutate.
      const expr2t::expr_ids canonical_id =
        is_nil_expr(result) ? expr_id : result->expr_id;
      const bool is_arith_chain = canonical_id == add_id ||
                                  canonical_id == sub_id ||
                                  canonical_id == neg_id;
      const bool is_other_chain =
        canonical_id == mul_id || canonical_id == bitand_id ||
        canonical_id == bitor_id || canonical_id == bitxor_id;
      if (is_arith_chain || is_other_chain)
      {
        expr2tc canonical = is_nil_expr(result) ? clone() : result;
        bool rewrote = false;
        if (is_arith_chain)
          rewrote = reassociate_arith(canonical);
        else if (canonical_id == mul_id)
          rewrote = reassociate_mul(canonical);
        else if (canonical_id == bitand_id)
          rewrote = reassociate_bitand(canonical);
        else if (canonical_id == bitor_id)
          rewrote = reassociate_bitor(canonical);
        else if (canonical_id == bitxor_id)
          rewrote = reassociate_bitxor(canonical);
        if (rewrote)
        {
          // Run peepholes on the rebuilt tree so add(x, neg(y)) -> sub(x, y)
          // and friends collapse. simplify_no_reassoc forces
          // suppress_reassoc=true throughout to avoid re-entering the
          // chain-root path.
          simplify_no_reassoc(canonical);
          return canonical;
        }
      }
    }

    return result;
  }
  catch (const array_type2t::dyn_sized_array_excp &e)
  {
    // Pretty much anything in any expression could be fouled up by there
    // being a dynamically sized array somewhere in there. In this circumstance,
    // don't even attempt partial simpilfication. We'd probably have to double
    // the size of simplification code in that case.
    return expr2tc();
  }
}

static expr2tc try_simplification(const expr2tc &expr)
{
  expr2tc to_simplify = expr->do_simplify();
  if (is_nil_expr(to_simplify))
    to_simplify = expr;
  return to_simplify;
}

static expr2tc typecast_check_return(const type2tc &type, const expr2tc &expr)
{
  // If the expr is already nil, do nothing
  if (is_nil_expr(expr))
    return expr2tc();

  // Don't type cast from constant to pointer
  // TODO: check if this is right
  if (is_pointer_type(type) && is_number_type(expr))
    return try_simplification(expr);

  // No need to typecast
  if (expr->type == type)
    return expr;

  // Create a typecast of the result
  return try_simplification(typecast2tc(type, expr));
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_arith_2ops(
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2)
{
  if (!is_number_type(type) && !is_pointer_type(type) && !is_vector_type(type))
    return expr2tc();

  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so the operands are already simplified. Local references kept for the
  // rest of the driver to use without renaming.
  const expr2tc &simplified_side_1 = side_1;
  const expr2tc &simplified_side_2 = side_2;

  if (
    !is_constant_expr(simplified_side_1) &&
    !is_constant_expr(simplified_side_2))
    return expr2tc();

  // This should be handled by ieee_*
  assert(!is_floatbv_type(type));

  expr2tc simpl_res;
  if (is_vector_type(type))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_int2t;

    std::function<BigInt &(expr2tc &)> get_value = [](expr2tc &c) -> BigInt & {
      return to_constant_int2t(c).value;
    };

    simpl_res = TFunctor<BigInt>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else if (is_bv_type(simplified_side_1) || is_bv_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_int2t;

    std::function<BigInt &(expr2tc &)> get_value = [](expr2tc &c) -> BigInt & {
      return to_constant_int2t(c).value;
    };

    simpl_res = TFunctor<BigInt>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);

    // Fix rounding when an overflow occurs
    if (!is_nil_expr(simpl_res) && is_constant_int2t(simpl_res))
      simpl_res =
        from_integer(to_constant_int2t(simpl_res).value, simpl_res->type);
  }
  else if (
    is_fixedbv_type(simplified_side_1) || is_fixedbv_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_fixedbv2t;

    std::function<fixedbvt &(expr2tc &)> get_value =
      [](expr2tc &c) -> fixedbvt & { return to_constant_fixedbv2t(c).value; };

    simpl_res = TFunctor<fixedbvt>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else if (is_bool_type(simplified_side_1) || is_bool_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_bool2t;

    std::function<bool &(expr2tc &)> get_value = [](expr2tc &c) -> bool & {
      return to_constant_bool2t(c).value;
    };

    simpl_res = TFunctor<bool>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else
    return expr2tc();

  return typecast_check_return(type, simpl_res);
}

template <class constant_type>
struct Addtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Is a vector operation ? Apply the op
    if (is_constant_vector2t(op1) || is_constant_vector2t(op2))
    {
      auto op = [](type2tc t, expr2tc e1, expr2tc e2) {
        return add2tc(t, e1, e2);
      };
      return distribute_vector_operation(op, op1, op2);
    }

    if (is_constant(op1))
    {
      // Found a zero? Simplify to op2
      expr2tc c1 = op1;
      if (get_value(c1) == 0)
        return op2;
    }

    if (is_constant(op2))
    {
      // Found a zero? Simplify to op1
      expr2tc c2 = op2;
      if (get_value(c2) == 0)
        return op1;
    }

    // Two constants? Simplify to result of the addition
    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      get_value(c1) += get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

// Forward declarations; definitions live further down in this file.
static bool fits_in_width(const BigInt &value, unsigned width, bool is_signed);
static bool is_all_ones_constant(const expr2tc &e);
static bool coerce_to_common_type(expr2tc &a, expr2tc &b);

expr2tc add2t::do_simplify() const
{
  // x + 0 = x, 0 + x = x. Mirrors Addtor::simplify but short-circuits before
  // simplify_arith_2ops walks the operands.
  if (is_constant_int2t(side_2) && to_constant_int2t(side_2).value.is_zero())
    return side_1;
  if (is_constant_int2t(side_1) && to_constant_int2t(side_1).value.is_zero())
    return side_2;

  // Symex sometimes calls do_simplify() directly after renaming a pointer
  // increment, bypassing the full expr2t::simplify() reassociation pass. Keep
  // this pointer-only fold local so repeated increments still canonicalize:
  //   (p + C1) + C2 -> p + (C1 + C2)
  if (is_pointer_type(type))
  {
    auto split_pointer_add_const =
      [](const expr2tc &expr, expr2tc &base, expr2tc &constant) -> bool {
      if (!is_add2t(expr))
        return false;

      const add2t &add = to_add2t(expr);
      if (is_pointer_type(add.side_1) && is_constant_int2t(add.side_2))
      {
        base = add.side_1;
        constant = add.side_2;
        return true;
      }

      if (is_pointer_type(add.side_2) && is_constant_int2t(add.side_1))
      {
        base = add.side_2;
        constant = add.side_1;
        return true;
      }

      return false;
    };

    // Pointer-add fold: each unfolded `add(pointer, ptr, c)` step
    // sign-extends c to the pointer offset width (index_type2, signed long)
    // before the SMT bv add (see smt_memspace.cpp:155). Cast both constants
    // to the offset type, sum the sign-extended values there, and re-emit
    // the folded constant at that same type. This avoids a per-operand
    // type-match guard that would refuse the fold whenever C produces
    // mixed-width offsets (e.g. `&arr[0] + (int)1 + (long)2`).
    auto fold_offsets = [](const expr2tc &c1, const expr2tc &c2) -> expr2tc {
      const type2tc &offset_t = index_type2();
      auto extend = [&](const expr2tc &c) -> BigInt {
        const BigInt &v = to_constant_int2t(c).value;
        const unsigned w = c->type->get_width();
        const bool is_signed = is_signedbv_type(c->type);
        return binary2integer(integer2binary(v, w), is_signed);
      };
      BigInt folded = extend(c1) + extend(c2);
      return from_integer(folded, offset_t);
    };

    expr2tc base, constant;
    if (
      is_constant_int2t(side_2) &&
      split_pointer_add_const(side_1, base, constant))
    {
      expr2tc folded = fold_offsets(constant, side_2);
      if (to_constant_int2t(folded).value.is_zero())
        return base;
      return add2tc(type, base, folded);
    }

    if (
      is_constant_int2t(side_1) &&
      split_pointer_add_const(side_2, base, constant))
    {
      expr2tc folded = fold_offsets(constant, side_1);
      if (to_constant_int2t(folded).value.is_zero())
        return base;
      return add2tc(type, base, folded);
    }
  }

  // x + (-x) = 0
  if (is_neg2t(side_2) && to_neg2t(side_2).value == side_1)
    return gen_zero(type);
  if (is_neg2t(side_1) && to_neg2t(side_1).value == side_2)
    return gen_zero(type);

  // Recognize (base - X) + X = base and X + (base - X) = base. Restricted
  // to non-pointer types: in pointer arithmetic the inner sub may be a
  // pointer-pointer subtraction yielding a ptrdiff integer, and rebuilding
  // the result drops `base`'s pointer provenance — at the SMT layer the
  // returned value loses the object-identity that the original chain had,
  // breaking memory-model proofs (e.g. github_1590).
  if (!is_pointer_type(type))
  {
    if (is_sub2t(side_1))
    {
      const sub2t &sub = to_sub2t(side_1);
      if (sub.side_2 == side_2)
        return sub.side_1;
    }

    if (is_sub2t(side_2))
    {
      const sub2t &sub = to_sub2t(side_2);
      if (sub.side_2 == side_1)
        return sub.side_1;
    }
  }

  // x + ~x -> -1
  if (is_bitnot2t(side_2) && to_bitnot2t(side_2).value == side_1)
    return constant_int2tc(type, BigInt(-1));
  if (is_bitnot2t(side_1) && to_bitnot2t(side_1).value == side_2)
    return constant_int2tc(type, BigInt(-1));

  // (-x) + (-y) -> -(x + y). Signed-bv only: for unsigned bv, neg2t lowers
  // to (modulus - x) % modulus, so the rewrite would fold two cheap structural
  // negs into one wrap, but the wrap re-enters modulus simplification on a
  // larger expression. Stick to signed where neg is structural.
  if (is_signedbv_type(type) && is_neg2t(side_1) && is_neg2t(side_2))
  {
    expr2tc sum = add2tc(type, to_neg2t(side_1).value, to_neg2t(side_2).value);
    return neg2tc(type, sum);
  }

  // x + (-y) -> x - y
  if (is_neg2t(side_2))
    return sub2tc(type, side_1, to_neg2t(side_2).value);

  // (-x) + y -> y - x
  if (is_neg2t(side_1))
    return sub2tc(type, side_2, to_neg2t(side_1).value);

  // x + (negative constant c) -> x - (-c). The reassoc rebuild can produce
  // an `add(x, -c)` once neg2t::do_simplify folds neg(constant) into a
  // negative constant, after which the `is_neg2t` peepholes above no
  // longer match. Catch the value-shaped form too.
  if (
    is_signedbv_type(type) && is_constant_int2t(side_2) &&
    to_constant_int2t(side_2).value.is_negative())
    return sub2tc(
      type, side_1, from_integer(-to_constant_int2t(side_2).value, type));
  if (
    is_signedbv_type(type) && is_constant_int2t(side_1) &&
    to_constant_int2t(side_1).value.is_negative())
    return sub2tc(
      type, side_2, from_integer(-to_constant_int2t(side_1).value, type));

  return simplify_arith_2ops<Addtor, add2t>(type, side_1, side_2);
}

template <class constant_type>
struct Subtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Is a vector operation ? Apply the op
    if (is_constant_vector2t(op1) || is_constant_vector2t(op2))
    {
      auto op = [](type2tc t, expr2tc e1, expr2tc e2) {
        return sub2tc(t, e1, e2);
      };
      return distribute_vector_operation(op, op1, op2);
    }

    if (is_constant(op1))
    {
      // Found a zero? Simplify to -op2
      expr2tc c1 = op1;
      if (get_value(c1) == 0)
      {
        expr2tc c = neg2tc(op2->type, op2);
        ::simplify(c);
        return c;
      }
    }

    if (is_constant(op2))
    {
      // Found a zero? Simplify to op1
      expr2tc c2 = op2;
      if (get_value(c2) == 0)
        return op1;
    }

    // Two constants? Simplify to result of the subtraction
    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      get_value(c1) -= get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

/**
 * Check if value fits in given bit width without overflow
 */
static bool fits_in_width(const BigInt &value, unsigned width, bool is_signed)
{
  if (is_signed)
  {
    BigInt min_val = -(BigInt::power2(width - 1));
    BigInt max_val = BigInt::power2(width - 1) - 1;
    return value >= min_val && value <= max_val;
  }
  else
  {
    if (value < 0)
      return false;
    BigInt max_val = BigInt::power2(width) - 1;
    return value <= max_val;
  }
}

/// True if @p e is a constant_int that holds the all-ones bit pattern for its
/// type. For signed bv that's -1 (BigInt(-1)); for unsigned bv that's
/// 2^width - 1 (UINT_MAX). Useful for bitwise identities like x & all1 = x.
static bool is_all_ones_constant(const expr2tc &e)
{
  if (!is_constant_int2t(e))
    return false;
  const BigInt &v = to_constant_int2t(e).value;
  if (is_signedbv_type(e->type))
    return v == BigInt(-1);
  if (is_unsignedbv_type(e->type))
    return v == BigInt::power2(e->type->get_width()) - 1;
  return false;
}

/// Coerce two operands of a binary fold to a common type so a freshly
/// rebuilt node is well-formed even when @p a and @p b carry different
/// concrete bv widths. Picks the wider of the two operand types; for
/// non-constant operands a type mismatch is unsoundable, so returns false.
/// Constants are reinterpreted at the common type via integer2binary
/// round-trip so the bit pattern under sign-extension matches what the SMT
/// layer would see.
///
/// Use whenever a peephole cancellation produces a fresh node from two
/// surviving operands whose types may diverge — pointer-arith chains commonly
/// mix `(int)1` with `(long)2` after C-frontend promotion, so a strict
/// type-match guard would refuse the fold.
static bool coerce_to_common_type(expr2tc &a, expr2tc &b)
{
  if (a->type == b->type)
    return true;
  if (!is_constant_int2t(a) || !is_constant_int2t(b))
    return false;
  const type2tc &common =
    a->type->get_width() >= b->type->get_width() ? a->type : b->type;
  auto recast = [&](const expr2tc &v) -> expr2tc {
    const BigInt &raw = to_constant_int2t(v).value;
    const unsigned w = v->type->get_width();
    const bool is_signed = is_signedbv_type(v->type);
    BigInt extended = binary2integer(integer2binary(raw, w), is_signed);
    return from_integer(extended, common);
  };
  if (a->type != common)
    a = recast(a);
  if (b->type != common)
    b = recast(b);
  return true;
}

expr2tc sub2t::do_simplify() const
{
  // x - 0 = x. Mirrors Subtor::simplify but short-circuits before
  // simplify_arith_2ops walks the operands.
  if (is_constant_int2t(side_2) && to_constant_int2t(side_2).value.is_zero())
    return side_1;

  // 0 - x = -x. Same motivation; Subtor::simplify also allocates a fresh
  // neg + recursive simplify, which is wasted on the trivial case.
  if (is_constant_int2t(side_1) && to_constant_int2t(side_1).value.is_zero())
    return neg2tc(type, side_2);

  // x - x = 0 (self-subtraction). Use the sub2t's own result type, not
  // side_1's type: pointer subtraction has pointer operands but a
  // ptrdiff/integer result, so gen_zero(side_1->type) would synthesize
  // a pointer-typed zero (i.e. NULL) and corrupt downstream encoding.
  if (side_1 == side_2)
    return gen_zero(type);

  if (is_bv_type(type))
  {
    // Recognize (base + X) - X = base pattern. bv-only: for pointer types,
    // returning add.side_2 (an integer offset) when add.side_1 == side_2
    // would yield a value with the wrong type — sub2t of two pointers has
    // ptrdiff type, but the offset's type is whatever the original add used.
    if (is_add2t(side_1))
    {
      const add2t &add = to_add2t(side_1);

      if (add.side_2 == side_2)
        return add.side_1;
      if (add.side_1 == side_2)
        return add.side_2;
    }

    // -1 - x -> ~x
    if (
      is_constant_int2t(side_1) &&
      to_constant_int2t(side_1).value == BigInt(-1))
      return bitnot2tc(type, side_2);

    // (~x) - (~y) -> y - x
    if (is_bitnot2t(side_1) && is_bitnot2t(side_2))
      return sub2tc(type, to_bitnot2t(side_2).value, to_bitnot2t(side_1).value);

    // x - (x - y) -> y
    if (is_sub2t(side_2))
    {
      const sub2t &sub = to_sub2t(side_2);

      if (sub.side_1 == side_1)
        return sub.side_2;
    }

    // x - (x + y) -> -y and x - (y + x) -> -y
    if (is_add2t(side_2))
    {
      const add2t &add = to_add2t(side_2);

      if (add.side_1 == side_1)
        return neg2tc(type, add.side_2);
      if (add.side_2 == side_1)
        return neg2tc(type, add.side_1);
    }

    // (w + x) - (y + z) with one shared addend cancels the common term.
    // Pointer-arith chains can have a common pointer base with mixed-width
    // integer offsets, and the surviving sub's result type is the parent
    // sub's type (ptrdiff for pointer-pointer subtraction, otherwise the
    // arith type) — coerce the operands so the rebuilt sub2tc is valid.
    auto cancel_sub = [&](expr2tc a, expr2tc b) -> expr2tc {
      if (!coerce_to_common_type(a, b))
        return expr2tc();
      // The arith_2ops invariant requires the operand widths match the
      // result type's width when neither side is a pointer. For
      // pointer-pointer cancellation the result type is ptrdiff but the
      // surviving operands are integer offsets — only fold when widths
      // match.
      if (a->type->get_width() != type->get_width())
        return expr2tc();
      return sub2tc(type, a, b);
    };

    if (is_add2t(side_1) && is_add2t(side_2))
    {
      const add2t &add1 = to_add2t(side_1);
      const add2t &add2 = to_add2t(side_2);
      expr2tc r;
      if (add1.side_1 == add2.side_1)
        if (!is_nil_expr(r = cancel_sub(add1.side_2, add2.side_2)))
          return r;
      if (add1.side_1 == add2.side_2)
        if (!is_nil_expr(r = cancel_sub(add1.side_2, add2.side_1)))
          return r;
      if (add1.side_2 == add2.side_1)
        if (!is_nil_expr(r = cancel_sub(add1.side_1, add2.side_2)))
          return r;
      if (add1.side_2 == add2.side_2)
        if (!is_nil_expr(r = cancel_sub(add1.side_1, add2.side_1)))
          return r;
    }
  }

  // x - (-y) -> x + y
  if (is_neg2t(side_2))
    return add2tc(type, side_1, to_neg2t(side_2).value);

  // (-x) - y -> -(x + y)
  if (is_neg2t(side_1))
  {
    expr2tc sum = add2tc(type, to_neg2t(side_1).value, side_2);
    return neg2tc(type, sum);
  }

  return simplify_arith_2ops<Subtor, sub2t>(type, side_1, side_2);
}

template <class constant_type>
struct Multor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Is a vector operation ? Apply the op
    if (is_constant_vector2t(op1) || is_constant_vector2t(op2))
    {
      auto op = [](type2tc t, expr2tc e1, expr2tc e2) {
        return mul2tc(t, e1, e2);
      };
      return distribute_vector_operation(op, op1, op2);
    }

    if (is_constant(op1))
    {
      expr2tc c1 = op1;

      // Found a zero? Simplify to zero
      if (get_value(c1) == 0)
        return op1;

      // Found an one? Simplify to op2
      if (get_value(c1) == 1)
        return op2;
    }

    if (is_constant(op2))
    {
      expr2tc c2 = op2;

      // Found a zero? Simplify to zero
      if (get_value(c2) == 0)
        return op2;

      // Found an one? Simplify to op1
      if (get_value(c2) == 1)
        return op1;
    }

    // Two constants? Simplify to result of the multiplication
    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      if constexpr (std::is_same<constant_type, bool>::value)
        get_value(c1) &= get_value(c2);
      else
        get_value(c1) *= get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

expr2tc mul2t::do_simplify() const
{
  // Scalar identity/absorber shortcuts. Skip when the result type is a
  // vector — returning a scalar `side_1`/`side_2` (or a scalar neg2t) would
  // produce a value of the wrong type. Vector shapes go through
  // simplify_arith_2ops, which calls distribute_vector_operation.
  if (!is_vector_type(type))
  {
    // x * 0 = 0, 0 * x = 0, x * 1 = x, 1 * x = x.
    if (is_constant_int2t(side_2))
    {
      const BigInt &v = to_constant_int2t(side_2).value;
      if (v.is_zero())
        return side_2;
      if (v == BigInt(1))
        return side_1;
    }
    if (is_constant_int2t(side_1))
    {
      const BigInt &v = to_constant_int2t(side_1).value;
      if (v.is_zero())
        return side_1;
      if (v == BigInt(1))
        return side_2;
    }

    // (-x) * (-y) -> x * y. Signed-bv only: for unsigned bv, the result is
    // still equal mod 2^N, but neg2t lowers to (modulus - x) % modulus, so
    // dropping the negs avoids double-wrap simplification on a larger
    // expression. Stick to signed where neg is structural.
    if (is_signedbv_type(type) && is_neg2t(side_1) && is_neg2t(side_2))
      return mul2tc(type, to_neg2t(side_1).value, to_neg2t(side_2).value);

    // (-x) * y -> -(x * y), x * (-y) -> -(x * y). Signed-bv only for the
    // same reason as above. Pulls neg outside so reassoc / constant-folding
    // can see the inner mul.
    if (is_signedbv_type(type) && is_neg2t(side_1))
      return neg2tc(type, mul2tc(type, to_neg2t(side_1).value, side_2));
    if (is_signedbv_type(type) && is_neg2t(side_2))
      return neg2tc(type, mul2tc(type, side_1, to_neg2t(side_2).value));

    // x * (-1) = -x
    if (is_constant_int2t(side_2) && to_constant_int2t(side_2).value == -1)
      return neg2tc(type, side_1);
    if (is_constant_int2t(side_1) && to_constant_int2t(side_1).value == -1)
      return neg2tc(type, side_2);
  }

  return simplify_arith_2ops<Multor, mul2t>(type, side_1, side_2);
}

namespace
{
template <bool Div, typename constant_type>
struct DivModtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Is a vector operation ? Apply the op
    if (is_constant_vector2t(op1) || is_constant_vector2t(op2))
    {
      auto op = [](type2tc t, expr2tc e1, expr2tc e2) {
        if constexpr (Div)
          return div2tc(t, e1, e2);
        else
          return modulus2tc(t, e1, e2);
      };
      return distribute_vector_operation(op, op1, op2);
    }

    if (is_constant(op2))
    {
      expr2tc c2 = op2;

      // Denominator is zero? Don't simplify
      if (get_value(c2) == 0)
        return expr2tc();

      // Denominator is one? Simplify to numerator's constant
      if (get_value(c2) == 1)
      {
        if constexpr (Div)
          return op1;
        else
          return gen_zero(op1->type);
      }
    }

    // Two constants? Simplify to result of the division
    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1;

      // Numerator is zero? Simplify to zero, as we know
      // that op2 is not zero
      if (get_value(c1) == 0)
        return op1;

      expr2tc c2 = op2;
      if constexpr (Div)
        get_value(c1) /= get_value(c2);
      else
        get_value(c1) %= get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

template <class constant_type>
struct Divtor : DivModtor<true, constant_type>
{
  using DivModtor<true, constant_type>::simplify;
};

template <class constant_type>
struct Modtor : DivModtor<false, constant_type>
{
  using DivModtor<false, constant_type>::simplify;
};

} // namespace

expr2tc div2t::do_simplify() const
{
  return simplify_arith_2ops<Divtor, div2t>(type, side_1, side_2);
}

expr2tc modulus2t::do_simplify() const
{
  // NOTE: deliberately no `x % x = 0` rule. For x == 0, the original is
  // modulo-by-zero (UB), and folding to 0 here would suppress the
  // div-by-zero check that runs as a separate VCC. Mirror of the
  // intentionally-absent `x / x = 1` rule in div2t::do_simplify.
  return simplify_arith_2ops<Modtor, modulus2t>(type, side_1, side_2);
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_arith_1op(const type2tc &type, const expr2tc &value)
{
  if (!is_number_type(type) && !is_vector_type(type))
    return expr2tc();

  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so `value` is already simplified.
  const expr2tc &to_simplify = value;
  if (!is_constant_expr(to_simplify))
    return expr2tc();

  expr2tc simpl_res;
  if (is_bv_type(value))
  {
    std::function<constant_int2t &(expr2tc &)> to_constant =
      (constant_int2t & (*)(expr2tc &)) to_constant_int2t;

    simpl_res = TFunctor<constant_int2t>::simplify(to_simplify, to_constant);

    // Properly handle truncation when an overflow occurs
    // or when the result gets out of the bounds of the type
    if (!is_nil_expr(simpl_res) && is_constant_int2t(simpl_res))
      simpl_res =
        from_integer(to_constant_int2t(simpl_res).value, simpl_res->type);
  }
  else if (is_fixedbv_type(value))
  {
    std::function<constant_fixedbv2t &(expr2tc &)> to_constant =
      (constant_fixedbv2t & (*)(expr2tc &)) to_constant_fixedbv2t;

    simpl_res =
      TFunctor<constant_fixedbv2t>::simplify(to_simplify, to_constant);
  }
  else if (is_floatbv_type(value))
  {
    std::function<constant_floatbv2t &(expr2tc &)> to_constant =
      (constant_floatbv2t & (*)(expr2tc &)) to_constant_floatbv2t;

    simpl_res =
      TFunctor<constant_floatbv2t>::simplify(to_simplify, to_constant);
  }
  else
    return expr2tc();

  return typecast_check_return(type, simpl_res);
}

template <class constant_type>
struct Negator
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;
    to_constant(c).value = -to_constant(c).value;
    return c;
  }
};

expr2tc neg2t::do_simplify() const
{
  // -(-x) -> x
  if (is_neg2t(value))
    return to_neg2t(value).value;

  if (is_constant_vector2t(value))
  {
    std::vector<expr2tc> members = to_constant_vector2t(value).datatype_members;
    for (size_t i = 0; i < members.size(); i++)
    {
      auto &op = members[i];
      members[i] = neg2tc(op->type, op);
    }
    return constant_vector2tc(value->type, std::move(members));
  }
  if (is_unsignedbv_type(value))
  {
    // Get bit-width of the unsigned type
    const unsigned int width = value->type->get_width();

    // Compute modulus: 2^width
    const BigInt modulus = BigInt(1) << width;
    const expr2tc modulus_expr = constant_int2tc(value->type, modulus);

    // Perform modular negation: (modulus - x) % modulus.
    //
    // simplify_no_reassoc instead of plain ::simplify: ::simplify would
    // re-enter the chain-root reassoc path on the freshly-built sub2tc
    // and, on already-flattened reassoc output, recurse without bound.
    // The wrap is a one-shot canonicalisation, not a chain root.
    const expr2tc negated_value = sub2tc(value->type, modulus_expr, value);
    expr2tc wrap = modulus2tc(value->type, negated_value, modulus_expr);
    simplify_no_reassoc(wrap);

    return wrap;
  }

  return simplify_arith_1op<Negator, neg2t>(type, value);
}

template <class constant_type>
struct abstor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;

    if (!to_constant(c).value.is_negative())
      return number;

    to_constant(c).value = -to_constant(c).value;
    return c;
  }
};

expr2tc abs2t::do_simplify() const
{
  // abs(abs(x)) -> abs(x)
  if (is_abs2t(value))
    return value;

  // abs(-x) -> abs(x)
  if (is_neg2t(value))
    return abs2tc(type, to_neg2t(value).value);

  return simplify_arith_1op<abstor, abs2t>(type, value);
}

expr2tc with2t::do_simplify() const
{
  // with(with(s, f, v_old), f, v_new) -> with(s, f, v_new). Two writes to
  // the same field/index — the older one is dead. Works for any source
  // shape (struct field, array index, vector lane); update_field equality
  // is structural so it covers constant_string field names and constant_int
  // indices uniformly.
  if (is_with2t(source_value))
  {
    const with2t &inner = to_with2t(source_value);
    if (inner.update_field == update_field)
      return with2tc(type, inner.source_value, update_field, update_value);
  }

  if (is_constant_struct2t(source_value))
  {
    const constant_struct2t &c_struct = to_constant_struct2t(source_value);
    const constant_string2t &memb = to_constant_string2t(update_field);
    unsigned no = static_cast<const struct_union_data &>(*type.get())
                    .get_component_number(memb.value);
    assert(no < c_struct.datatype_members.size());

    if (c_struct.datatype_members[no] == update_value)
      return source_value;

    // Clone constant struct, update its field according to this "with".
    constant_struct2t copy = c_struct;
    copy.datatype_members[no] = update_value;
    return constant_struct2tc(std::move(copy));
  }
  else if (is_constant_union2t(source_value))
  {
    const constant_union2t &c_union = to_constant_union2t(source_value);
    const union_type2t &thetype = to_union_type(c_union.type);
    const constant_string2t &memb = to_constant_string2t(update_field);
    unsigned no = static_cast<const struct_union_data &>(*c_union.type.get())
                    .get_component_number(memb.value);
    assert(no < thetype.member_names.size());

    // If the update value type matches the current lump of data's type, we can
    // just replace it with the new value. As far as I can tell, constant unions
    // only ever contain one member, and it's the member most recently written.
    if (thetype.members[no] != update_value->type)
      return expr2tc();

    if (
      c_union.init_field == thetype.member_names[no] &&
      !c_union.datatype_members.empty() &&
      c_union.datatype_members[0] == update_value)
      return source_value;

    std::vector<expr2tc> newmembers = {update_value};
    return constant_union2tc(type, thetype.member_names[no], newmembers);
  }
  else if (is_constant_array2t(source_value) && is_constant_int2t(update_field))
  {
    const constant_array2t &array = to_constant_array2t(source_value);
    const constant_int2t &index = to_constant_int2t(update_field);

    // Index may be out of bounds. That's an error in the program, but not in
    // the model we're generating, so permit it. Can't simplify it though.
    if (index.value.is_negative())
      return expr2tc();

    if (index.value >= array.datatype_members.size())
      return expr2tc();

    if (array.datatype_members[index.as_ulong()] == update_value)
      return source_value;

    constant_array2t arr = array; // copy
    arr.datatype_members[index.as_ulong()] = update_value;
    return constant_array2tc(std::move(arr));
  }
  else if (is_constant_vector2t(source_value))
  {
    const constant_vector2t &vec = to_constant_vector2t(source_value);
    const constant_int2t &index = to_constant_int2t(update_field);

    // Index may be out of bounds. That's an error in the program, but not in
    // the model we're generating, so permit it. Can't simplify it though.
    if (index.value.is_negative())
      return expr2tc();

    if (index.value >= vec.datatype_members.size())
      return expr2tc();

    if (vec.datatype_members[index.as_ulong()] == update_value)
      return source_value;

    constant_vector2t vec2 = vec; // copy
    vec2.datatype_members[index.as_ulong()] = update_value;
    return constant_vector2tc(std::move(vec2));
  }
  else if (is_constant_array_of2t(source_value))
  {
    const constant_array_of2t &array = to_constant_array_of2t(source_value);
    const array_type2t &arr_type = to_array_type(array.type);

    // Don't simplify away these withs if the array_of is infinitely sized:
    // infinite arrays aren't supported by the SMT backend, so a per-element
    // assignment must remain visible rather than being const-propagated
    // back into the initializer.
    if (arr_type.size_is_infinite)
      return expr2tc();

    // Eliminate this operation if the update value matches the initializer.
    if (update_value == array.initializer)
      return source_value;

    return expr2tc();
  }
  return expr2tc();
}

expr2tc member2t::do_simplify() const
{
  if (is_constant_struct2t(source_value) || is_constant_union2t(source_value))
  {
    unsigned no =
      static_cast<const struct_union_data &>(*source_value->type.get())
        .get_component_number(member);

    // Clone constant struct, update its field according to this "with".
    expr2tc s;
    if (is_constant_struct2t(source_value))
    {
      s = to_constant_struct2t(source_value).datatype_members[no];
      // Be defensive: if member extraction type doesn't match, skip
      // simplification instead of aborting in the simplifier.
      if (
        !is_pointer_type(type) &&
        !base_type_eq(type, s->type, *migrate_namespace_lookup))
        return expr2tc();
    }
    else
    {
      // constant_union stores the value at position 0 with init_field indicating
      // which member was initialized.
      const constant_union2t &uni = to_constant_union2t(source_value);

      // Only the active union member can be simplified away.
      if (uni.init_field != member)
        return expr2tc();

      // The value is always stored at position 0
      if (uni.datatype_members.empty())
        return expr2tc();

      s = uni.datatype_members[0];

      /* If the type we just selected isn't compatible, it means that whatever
       * field is in the constant union /isn't/ the field we're selecting from
       * it. So don't simplify it, because we can't. */

      if (
        !is_pointer_type(type) &&
        !base_type_eq(type, s->type, *migrate_namespace_lookup))
        return expr2tc();
    }

    return s;
  }
  else if (is_with2t(source_value))
  {
    const with2t &with = to_with2t(source_value);

    // Only safe to peer through a `with` when we know which field/index it
    // updates. A non-constant_string update_field could in principle alias
    // any member; treat it as opaque.
    if (!is_constant_string2t(with.update_field))
      return expr2tc();

    // LLVM-style extractvalue/insertvalue folding: a matching update wins.
    if (member == to_constant_string2t(with.update_field).value)
      return with.update_value;

    // Unrelated update: transparent only for non-union sources. Union members
    // share the same memory bytes, so a write to one member observably
    // changes another member's value — we cannot step past the `with`.
    if (is_union_type(with.source_value->type))
      return expr2tc();

    return member2tc(type, with.source_value, member);
  }
  // Handle bitcast expressions
  else if (is_bitcast2t(source_value))
  {
    const bitcast2t &bc = to_bitcast2t(source_value);
    // If bitcast wraps byte_update, try to extract from the original before byte_update
    if (is_byte_update2t(bc.from))
    {
      const byte_update2t &bu = to_byte_update2t(bc.from);
      // If bu.source_value is also a bitcast, recurse through it
      if (is_bitcast2t(bu.source_value))
      {
        const bitcast2t &inner_bc = to_bitcast2t(bu.source_value);
        if (is_struct_type(inner_bc.from->type))
        {
          expr2tc member_of_original = member2tc(type, inner_bc.from, member);
          expr2tc simplified = member_of_original->simplify();
          if (!is_nil_expr(simplified))
            return simplified;
        }
      }
    }
  }

  return expr2tc();
}

static expr2tc simplify_object(const expr2tc &expr)
{
  if (is_add2t(expr) || is_sub2t(expr))
  {
    if (is_pointer_type(expr->type))
    {
      expr2tc left_op =
        is_add2t(expr) ? to_add2t(expr).side_1 : to_sub2t(expr).side_1;
      expr2tc right_op =
        is_add2t(expr) ? to_add2t(expr).side_2 : to_sub2t(expr).side_2;

      // Look for pointer operands - prioritize left side, then right side
      if (is_pointer_type(left_op->type))
      {
        expr2tc simplified = simplify_object(left_op);
        return is_nil_expr(simplified) ? left_op : simplified;
      }
      else if (is_pointer_type(right_op->type))
      {
        expr2tc simplified = simplify_object(right_op);
        return is_nil_expr(simplified) ? right_op : simplified;
      }
    }
  }
  else if (is_typecast2t(expr))
  {
    const typecast2t &cast_expr = to_typecast2t(expr);
    if (is_pointer_type(cast_expr.from->type))
    {
      expr2tc simplified = simplify_object(cast_expr.from);
      return is_nil_expr(simplified) ? cast_expr.from : simplified;
    }
  }
  // NOTE: address_of(member/index) look-through gated off — the SMT backend
  // expects pointer_object's argument to keep its original pointer type
  // because the projection is type-driven. Rewriting through the inner
  // member/index produces a pointer of a different subtype and triggers
  // "Projecting from non-tuple based AST" downstream.

  return expr2tc();
}

static bool is_null_pointer_constant(const expr2tc &expr)
{
  if (is_constant_int2t(expr))
    return to_constant_int2t(expr).value.is_zero();

  if (is_typecast2t(expr) && is_pointer_type(expr->type))
    return is_null_pointer_constant(to_typecast2t(expr).from);

  return false;
}

expr2tc pointer_object2t::do_simplify() const
{
  expr2tc simplified_obj = simplify_object(ptr_obj);

  if (!is_nil_expr(simplified_obj))
    return pointer_object2tc(type, simplified_obj);

  return expr2tc();
}

expr2tc pointer_offset2t::do_simplify() const
{
  // XXX - this could be better. But the current implementation catches most
  // cases that ESBMC produces internally.

  if (is_symbol2t(ptr_obj) && to_symbol2t(ptr_obj).thename == "NULL")
  {
    if (is_pointer_type(ptr_obj->type))
    {
      const pointer_type2t &ptr_type = to_pointer_type(ptr_obj->type);
      // Allow NULL simplification for pointer types to primitives
      if (!is_symbol_type(ptr_type.subtype))
        return gen_zero(type);
    }
  }

  if (is_address_of2t(ptr_obj))
  {
    const address_of2t &addrof = to_address_of2t(ptr_obj);
    if (is_symbol2t(addrof.ptr_obj) || is_constant_string2t(addrof.ptr_obj))
      return gen_zero(type);

    if (is_index2t(addrof.ptr_obj))
    {
      const index2t &index = to_index2t(addrof.ptr_obj);

      // check if index is constant, looking through typecasts
      expr2tc index_value = index.index;

      // Look through typecast to find the underlying constant
      if (is_typecast2t(index_value))
        index_value = to_typecast2t(index_value).from;

      if (is_constant_int2t(index_value))
      {
        // Fast path for &symbol[0] / &string_lit[0]: byte offset is 0.
        if (
          to_constant_int2t(index_value).value.is_zero() &&
          (is_symbol2t(index.source_value) ||
           is_constant_string2t(index.source_value)))
          return gen_zero(type);

        // compute_pointer_offset builds a fresh typecast/div/mul/add tree;
        // try_simplification only walks the root, so use the full simplifier
        // to fold member_offset + idx*sizeof down to a constant for nested
        // shapes like &(member.array)[0].
        expr2tc offs = compute_pointer_offset(addrof.ptr_obj);
        expr2tc folded = offs->simplify();
        if (!is_nil_expr(folded))
          offs = folded;
        if (is_constant_int2t(offs))
          return offs;
      }
    }

    if (is_member2t(addrof.ptr_obj))
    {
      const member2t &member = to_member2t(addrof.ptr_obj);

      // First, try to use our existing compute_pointer_offset function
      // This should handle most struct member cases
      expr2tc offs = try_simplification(compute_pointer_offset(addrof.ptr_obj));
      if (
        !is_nil_expr(offs) &&
        (!is_dereference2t(member.source_value) ||
         is_null_pointer_constant(to_dereference2t(member.source_value).value)))
        return offs;

      // For dereference-rooted members (i.e. &p->m), the byte offset is
      // pointer_offset(p) + member_offset. Reserve gen_zero for the
      // concrete-object case where p is known to be NULL or where the
      // source isn't a dereference at all.
      const bool is_deref_source = is_dereference2t(member.source_value);
      const expr2tc base_offset =
        is_deref_source
          ? pointer_offset2tc(type, to_dereference2t(member.source_value).value)
          : gen_zero(type);

      // Union members all have member_offset 0 relative to the union base,
      // so &p->u_member's byte offset is just pointer_offset(p).
      if (is_union_type(member.source_value->type))
        return base_offset;

      if (is_struct_type(member.source_value->type))
      {
        const struct_union_data &struct_data =
          static_cast<const struct_union_data &>(
            *member.source_value->type.get());
        unsigned member_no = struct_data.get_component_number(member.member);
        if (member_no == 0)
          return base_offset;
      }
    }
  }
  else if (is_typecast2t(ptr_obj))
  {
    const typecast2t &cast = to_typecast2t(ptr_obj);
    if (
      is_pointer_type(ptr_obj->type) && is_constant_int2t(cast.from) &&
      to_constant_int2t(cast.from).value.is_zero())
      return gen_zero(type);

    expr2tc new_ptr_offs = pointer_offset2tc(type, cast.from);
    expr2tc reduced = new_ptr_offs->simplify();

    // If we got a good simplification, return it
    if (!is_nil_expr(reduced))
      return reduced;
  }
  else if (is_add2t(ptr_obj))
  {
    const add2t &add = to_add2t(ptr_obj);

    // So, one of these should be a ptr type, or there isn't any point in this
    // being a pointer_offset irep.
    if (!is_pointer_type(add.side_1) && !is_pointer_type(add.side_2))
      return expr2tc();

    // Can't have pointer-on-pointer arith.
    assert(!(is_pointer_type(add.side_1) && is_pointer_type(add.side_2)));

    expr2tc ptr_op = (is_pointer_type(add.side_1)) ? add.side_1 : add.side_2;
    expr2tc non_ptr_op =
      (is_pointer_type(add.side_1)) ? add.side_2 : add.side_1;

    // p + 0 has the same offset as p. Skip the rewrite-and-resimplify loop.
    if (
      is_constant_int2t(non_ptr_op) &&
      to_constant_int2t(non_ptr_op).value.is_zero())
      return pointer_offset2tc(type, ptr_op);

    // Can't do any kind of simplification if the ptr op has a symbolic type.
    // Let the SMT layer handle this. In the future, can we pass around a
    // namespace?
    if (is_symbol_type(to_pointer_type(ptr_op->type).subtype))
      return expr2tc();

    // Turn the pointer one into pointer_offset.
    expr2tc new_ptr_op = pointer_offset2tc(type, ptr_op);
    // And multiply the non pointer one by the type size.
    type2tc ptr_subtype = to_pointer_type(ptr_op->type).subtype;
    expr2tc type_size =
      type_byte_size_expr(ptr_subtype, migrate_namespace_lookup);

    if (non_ptr_op->type != type)
      non_ptr_op = typecast2tc(type, non_ptr_op);
    // type_byte_size_expr returns size_type2 (unsigned 64-bit). The outer
    // type may be signed (signed_long for pointer offsets). Coerce to
    // match so the mul/add chain is homogeneously typed; a mixed s64/u64
    // mul confuses downstream equality folding (see github_1590-style
    // pointer-arith proofs).
    if (type_size->type != type)
      type_size = typecast2tc(type, type_size);

    expr2tc new_non_ptr_op = mul2tc(type, non_ptr_op, type_size);

    expr2tc new_add = add2tc(type, new_ptr_op, new_non_ptr_op);

    // So, this add is a valid simplification. We may be able to simplify
    // further though.
    expr2tc tmp = new_add->simplify();
    if (is_nil_expr(tmp))
      return new_add;

    return tmp;
  }
  else if (is_sub2t(ptr_obj))
  {
    const sub2t &sub = to_sub2t(ptr_obj);

    // Handle ptr - offset
    if (is_pointer_type(sub.side_1) && !is_pointer_type(sub.side_2))
    {
      expr2tc ptr_op = sub.side_1;
      expr2tc offset_op = sub.side_2;

      // p - 0 has the same offset as p. Skip the rewrite-and-resimplify loop.
      if (
        is_constant_int2t(offset_op) &&
        to_constant_int2t(offset_op).value.is_zero())
        return pointer_offset2tc(type, ptr_op);

      if (is_symbol_type(to_pointer_type(ptr_op->type).subtype))
        return expr2tc();

      expr2tc ptr_offset = pointer_offset2tc(type, ptr_op);
      type2tc ptr_subtype = to_pointer_type(ptr_op->type).subtype;
      expr2tc type_size =
        type_byte_size_expr(ptr_subtype, migrate_namespace_lookup);

      if (offset_op->type != type)
        offset_op = typecast2tc(type, offset_op);
      // type_byte_size_expr returns size_type2 (unsigned 64-bit). The outer
      // type may be signed (signed_long). Coerce so the mul is homogeneous.
      if (type_size->type != type)
        type_size = typecast2tc(type, type_size);

      expr2tc scaled_offset = mul2tc(type, offset_op, type_size);
      expr2tc result = sub2tc(type, ptr_offset, scaled_offset);

      expr2tc simplified = result->simplify();
      return is_nil_expr(simplified) ? result : simplified;
    }
  }

  return expr2tc();
}

static bool index_values_equal(const expr2tc &idx1, const expr2tc &idx2)
{
  // Same expression (including symbolic) trivially has equal value.
  // Catches the `with(arr, j, v) [j]` pattern where j is symbolic.
  if (idx1 == idx2)
    return true;

  // For constant integers, compare values regardless of signedness
  if (is_constant_int2t(idx1) && is_constant_int2t(idx2))
  {
    const BigInt &val1 = to_constant_int2t(idx1).value;
    const BigInt &val2 = to_constant_int2t(idx2).value;
    return val1 == val2;
  }

  return false;
}

static expr2tc resolve_with_chain_lookup(
  const type2tc &elem_type,
  const expr2tc &source,
  const expr2tc &lookup_index)
{
  expr2tc current = source;

  // Collect all WITH operations in reverse order (most recent first)
  std::vector<std::pair<expr2tc, expr2tc>> updates; // field, value pairs

  while (is_with2t(current))
  {
    const with2t &with_op = to_with2t(current);
    updates.push_back({with_op.update_field, with_op.update_value});
    current = with_op.source_value;
  }

  // Walk most-recent first. A matching update wins. A provably distinct
  // constant-index update is transparent — we can step over it. A symbolic
  // update we can't reason about stops the walk: any later read could alias.
  bool lookup_is_const = is_constant_int2t(lookup_index);
  for (const auto &update : updates)
  {
    if (index_values_equal(update.first, lookup_index))
      return update.second;

    if (!(lookup_is_const && is_constant_int2t(update.first)))
      return expr2tc(); // can't prove non-aliasing, give up
  }

  // All updates were distinct constant indices and didn't match. The read
  // sees through to the chain's base. Returning index(base, lookup) lets
  // the constant-array / constant-array-of arms below fire on the base.
  if (current != source)
    return index2tc(elem_type, current, lookup_index);

  return expr2tc();
}

expr2tc index2t::do_simplify() const
{
  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so source_value and index are already simplified.
  const expr2tc &src = source_value;
  const expr2tc &idx_e = index;

  if (is_with2t(src))
  {
    expr2tc resolved = resolve_with_chain_lookup(type, src, idx_e);
    if (!is_nil_expr(resolved))
      return resolved;
  }

  if (is_constant_array2t(src) && is_constant_int2t(idx_e))
  {
    // Index might be greater than the constant array size. This means we can't
    // simplify it, and the user might be eaten by an assertion failure in the
    // model. We don't have to think about this now though.
    const constant_int2t &idx = to_constant_int2t(idx_e);
    if (idx.value.is_negative())
      return expr2tc();

    const constant_array2t &arr = to_constant_array2t(src);
    unsigned long the_idx = idx.as_ulong();
    if (the_idx >= arr.datatype_members.size())
      return expr2tc();

    return arr.datatype_members[the_idx];
  }

  if (is_constant_vector2t(src) && is_constant_int2t(idx_e))
  {
    const constant_vector2t &arr = to_constant_vector2t(src);
    const constant_int2t &idx = to_constant_int2t(idx_e);

    // Index might be greater than the constant array size. This means we can't
    // simplify it, and the user might be eaten by an assertion failure in the
    // model. We don't have to think about this now though.
    if (idx.value.is_negative())
      return expr2tc();

    if (idx.value >= arr.datatype_members.size())
      return expr2tc();

    return arr.datatype_members[idx.as_ulong()];
  }

  if (is_constant_string2t(src) && is_constant_int2t(idx_e))
  {
    // Same index situation
    const constant_int2t &idx = to_constant_int2t(idx_e);
    if (idx.value.is_negative())
      return expr2tc();

    const constant_string2t &str = to_constant_string2t(src);
    unsigned long the_idx = idx.as_ulong();
    if (the_idx >= str.array_size()) // allow reading null term.
      return expr2tc();

    // String constants had better be some kind of integer type
    assert(is_bv_type(type));
    expr2tc c = str.at(the_idx);
    assert(c);
    return c;
  }

  // Only thing this index can evaluate to is the default value of this array.
  if (is_constant_array_of2t(src))
    return to_constant_array_of2t(src).initializer;

  return expr2tc();
}

expr2tc not2t::do_simplify() const
{
  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so `value` is already simplified.
  const expr2tc &simp = value;

  // !!x = x (double negation)
  if (is_not2t(simp))
    return to_not2t(simp).value;

  // !true / !false fold. Constant operands are common after the operands-first
  // walk; check before allocating new expressions in the rewrites below.
  if (is_constant_bool2t(simp))
    return constant_bool2tc(!to_constant_bool2t(simp).value);

  // De Morgan's laws for logical operations
  // !(x && y) = !x || !y
  if (is_and2t(simp))
  {
    const and2t &and_expr = to_and2t(simp);
    return or2tc(not2tc(and_expr.side_1), not2tc(and_expr.side_2));
  }

  // !(x || y) = !x && !y
  if (is_or2t(simp))
  {
    const or2t &or_expr = to_or2t(simp);
    return and2tc(not2tc(or_expr.side_1), not2tc(or_expr.side_2));
  }

  // !(x ^ y) = !x == !y. For booleans, xor is just inequality, so the negated
  // form is structural equality. Lets the equality folder see through.
  if (is_xor2t(simp))
  {
    const xor2t &xor_expr = to_xor2t(simp);
    return equality2tc(xor_expr.side_1, xor_expr.side_2);
  }

  // !(x => y) = x && !y. ESBMC keeps a structural `implies2t` rather than
  // expanding to !x || y up front, so handle the negation explicitly.
  if (is_implies2t(simp))
  {
    const implies2t &imp = to_implies2t(simp);
    return and2tc(imp.side_1, not2tc(imp.side_2));
  }

  // Comparison negations - only for non-floating point types
  // !(x == y) = x != y
  if (is_equality2t(simp))
  {
    const equality2t &eq = to_equality2t(simp);
    if (is_floatbv_type(eq.side_1) || is_floatbv_type(eq.side_2))
      return expr2tc();
    else
      return notequal2tc(eq.side_1, eq.side_2);
  }

  // !(x != y) = x == y
  if (is_notequal2t(simp))
  {
    const notequal2t &neq = to_notequal2t(simp);
    if (is_floatbv_type(neq.side_1) || is_floatbv_type(neq.side_2))
      return expr2tc();
    else
      return equality2tc(neq.side_1, neq.side_2);
  }

  // !(x < y) = x >= y
  if (is_lessthan2t(simp))
  {
    const lessthan2t &lt = to_lessthan2t(simp);
    if (is_floatbv_type(lt.side_1) || is_floatbv_type(lt.side_2))
      return expr2tc();
    else
      return greaterthanequal2tc(lt.side_1, lt.side_2);
  }

  // !(x <= y) = x > y
  if (is_lessthanequal2t(simp))
  {
    const lessthanequal2t &lte = to_lessthanequal2t(simp);
    if (is_floatbv_type(lte.side_1) || is_floatbv_type(lte.side_2))
      return expr2tc();
    else
      return greaterthan2tc(lte.side_1, lte.side_2);
  }

  // !(x > y) = x <= y
  if (is_greaterthan2t(simp))
  {
    const greaterthan2t &gt = to_greaterthan2t(simp);
    if (is_floatbv_type(gt.side_1) || is_floatbv_type(gt.side_2))
      return expr2tc();
    else
      return lessthanequal2tc(gt.side_1, gt.side_2);
  }

  // !(x >= y) = x < y
  if (is_greaterthanequal2t(simp))
  {
    const greaterthanequal2t &gte = to_greaterthanequal2t(simp);
    if (is_floatbv_type(gte.side_1) || is_floatbv_type(gte.side_2))
      return expr2tc();
    else
      return lessthan2tc(gte.side_1, gte.side_2);
  }

  return expr2tc();
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_logic_2ops(
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2)
{
  if (!is_number_type(type))
    return expr2tc();

  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so the operands are already simplified.
  const expr2tc &simplified_side_1 = side_1;
  const expr2tc &simplified_side_2 = side_2;

  if (
    !is_constant_expr(simplified_side_1) &&
    !is_constant_expr(simplified_side_2))
    return expr2tc();

  expr2tc simpl_res;

  if (is_bv_type(simplified_side_1) || is_bv_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_int2t;

    std::function<BigInt &(expr2tc &)> get_value = [](expr2tc &c) -> BigInt & {
      return to_constant_int2t(c).value;
    };

    simpl_res = TFunctor<BigInt>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else if (
    is_fixedbv_type(simplified_side_1) || is_fixedbv_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_fixedbv2t;

    std::function<fixedbvt &(expr2tc &)> get_value =
      [](expr2tc &c) -> fixedbvt & { return to_constant_fixedbv2t(c).value; };

    simpl_res = TFunctor<fixedbvt>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else if (
    is_floatbv_type(simplified_side_1) || is_floatbv_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_floatbv2t;

    std::function<ieee_floatt &(expr2tc &)> get_value =
      [](expr2tc &c) -> ieee_floatt & {
      return to_constant_floatbv2t(c).value;
    };

    simpl_res = TFunctor<ieee_floatt>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else if (is_bool_type(simplified_side_1) || is_bool_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_bool2t;

    std::function<bool &(expr2tc &)> get_value = [](expr2tc &c) -> bool & {
      return to_constant_bool2t(c).value;
    };

    simpl_res = TFunctor<bool>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else
    return expr2tc();

  return typecast_check_return(type, simpl_res);
}

template <class constant_type>
struct Andtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Two constants? Simplify to result of the and
    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(!(get_value(c1) == 0) && !(get_value(c2) == 0));
    }

    if (is_constant(op1))
    {
      // False? never true
      expr2tc c1 = op1;
      return (get_value(c1) == 0) ? gen_false_expr() : op2;
    }

    if (is_constant(op2))
    {
      // False? never true
      expr2tc c2 = op2;
      return (get_value(c2) == 0) ? gen_false_expr() : op1;
    }

    return expr2tc();
  }
};

template <typename OpType, typename OpConstructor>
expr2tc simplify_associative_binary_op(
  const type2tc &result_type,
  const expr2tc &side_1,
  const expr2tc &side_2,
  bool (*is_op_type)(const expr2tc &),
  const OpType &(*to_op_type)(const expr2tc &),
  OpConstructor op_constructor)
{
  if (is_op_type(side_1) && is_op_type(side_2))
  {
    const OpType &op1 = to_op_type(side_1);
    const OpType &op2 = to_op_type(side_2);

    // Check all four possible combinations for common factors
    if (op1.side_1 == op2.side_1)
    {
      expr2tc combined = op_constructor(op1.side_2, op2.side_2);
      return typecast_check_return(
        result_type, op_constructor(op1.side_1, combined));
    }
    if (op1.side_1 == op2.side_2)
    {
      expr2tc combined = op_constructor(op1.side_2, op2.side_1);
      return typecast_check_return(
        result_type, op_constructor(op1.side_1, combined));
    }
    if (op1.side_2 == op2.side_1)
    {
      expr2tc combined = op_constructor(op1.side_1, op2.side_2);
      return typecast_check_return(
        result_type, op_constructor(op1.side_2, combined));
    }
    if (op1.side_2 == op2.side_2)
    {
      expr2tc combined = op_constructor(op1.side_1, op2.side_1);
      return typecast_check_return(
        result_type, op_constructor(op1.side_2, combined));
    }
  }
  return expr2tc(); // Return empty expr2tc to indicate no simplification was performed
}

expr2tc and2t::do_simplify() const
{
  // Constant short-circuits. After operands-first simplify these are common,
  // and Andtor's full dispatch is wasted effort if we can answer cheaply.
  if (is_constant_bool2t(side_1))
    return to_constant_bool2t(side_1).value ? side_2 : gen_false_expr();
  if (is_constant_bool2t(side_2))
    return to_constant_bool2t(side_2).value ? side_1 : gen_false_expr();

  if (side_1 == side_2)
    return side_1; // x && x = x

  // x && !x = false
  if (is_not2t(side_1) && to_not2t(side_1).value == side_2)
    return gen_false_expr();
  if (is_not2t(side_2) && to_not2t(side_2).value == side_1)
    return gen_false_expr();

  // Absorption: x && (x || y) = x
  if (is_or2t(side_2))
  {
    const or2t &or_expr = to_or2t(side_2);
    if (or_expr.side_1 == side_1 || or_expr.side_2 == side_1)
      return side_1;
  }
  if (is_or2t(side_1))
  {
    const or2t &or_expr = to_or2t(side_1);
    if (or_expr.side_1 == side_2 || or_expr.side_2 == side_2)
      return side_2;
  }

  // Complementary absorption: x && (!x || y) = x && y. The (!x || y) factor
  // contributes no extra information beyond y when x already holds.
  auto match_complement =
    [](const expr2tc &needle, const expr2tc &candidate) -> bool {
    return is_not2t(candidate) && to_not2t(candidate).value == needle;
  };
  if (is_or2t(side_2))
  {
    const or2t &o = to_or2t(side_2);
    if (match_complement(side_1, o.side_1))
      return and2tc(side_1, o.side_2);
    if (match_complement(side_1, o.side_2))
      return and2tc(side_1, o.side_1);
  }
  if (is_or2t(side_1))
  {
    const or2t &o = to_or2t(side_1);
    if (match_complement(side_2, o.side_1))
      return and2tc(side_2, o.side_2);
    if (match_complement(side_2, o.side_2))
      return and2tc(side_2, o.side_1);
  }

  // (x && y) && y = x && y, (x && y) && x = x && y, and the symmetric forms
  // with the outer and's operands swapped. Idempotence after associativity.
  if (is_and2t(side_1))
  {
    const and2t &inner = to_and2t(side_1);
    if (inner.side_1 == side_2 || inner.side_2 == side_2)
      return side_1;
  }
  if (is_and2t(side_2))
  {
    const and2t &inner = to_and2t(side_2);
    if (inner.side_1 == side_1 || inner.side_2 == side_1)
      return side_2;
  }

  // Try associative simplification: (x && a) && (x && b) = x && (a && b)
  expr2tc simplified = simplify_associative_binary_op<and2t>(
    type,
    side_1,
    side_2,
    is_and2t,
    to_and2t,
    [](const expr2tc &left, const expr2tc &right) {
      return and2tc(left, right);
    });

  if (!is_nil_expr(simplified))
    return simplified;

  return simplify_logic_2ops<Andtor, and2t>(type, side_1, side_2);
}

template <class constant_type>
struct Ortor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Two constants? Simplify to result of the or
    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(!(get_value(c1) == 0) || !(get_value(c2) == 0));
    }

    if (is_constant(op1))
    {
      // True? return true
      expr2tc c1 = op1;
      return (!(get_value(c1) == 0)) ? gen_true_expr() : op2;
    }

    if (is_constant(op2))
    {
      // True? return true
      expr2tc c2 = op2;
      return (!(get_value(c2) == 0)) ? gen_true_expr() : op1;
    }

    return expr2tc();
  }
};

expr2tc or2t::do_simplify() const
{
  // Constant short-circuits. After operands-first simplify these are common,
  // and Ortor's full dispatch is wasted effort if we can answer cheaply.
  if (is_constant_bool2t(side_1))
    return to_constant_bool2t(side_1).value ? gen_true_expr() : side_2;
  if (is_constant_bool2t(side_2))
    return to_constant_bool2t(side_2).value ? gen_true_expr() : side_1;

  if (side_1 == side_2)
    return side_1; // x || x = x

  // Special case: if one side is a not of the other, and they're otherwise
  // identical, simplify to true
  if (is_not2t(side_1))
  {
    const not2t &ref = to_not2t(side_1);
    if (ref.value == side_2)
      return gen_true_expr();
  }
  else if (is_not2t(side_2))
  {
    const not2t &ref = to_not2t(side_2);
    if (ref.value == side_1)
      return gen_true_expr();
  }

  // Absorption: x || (x && y) = x
  if (is_and2t(side_2))
  {
    const and2t &and_expr = to_and2t(side_2);
    if (and_expr.side_1 == side_1 || and_expr.side_2 == side_1)
      return side_1;
  }
  if (is_and2t(side_1))
  {
    const and2t &and_expr = to_and2t(side_1);
    if (and_expr.side_1 == side_2 || and_expr.side_2 == side_2)
      return side_2;
  }

  // Complementary absorption: x || (!x && y) = x || y. The (!x && y) factor
  // contributes no extra information beyond y when x already fails.
  auto match_complement =
    [](const expr2tc &needle, const expr2tc &candidate) -> bool {
    return is_not2t(candidate) && to_not2t(candidate).value == needle;
  };
  if (is_and2t(side_2))
  {
    const and2t &a = to_and2t(side_2);
    if (match_complement(side_1, a.side_1))
      return or2tc(side_1, a.side_2);
    if (match_complement(side_1, a.side_2))
      return or2tc(side_1, a.side_1);
  }
  if (is_and2t(side_1))
  {
    const and2t &a = to_and2t(side_1);
    if (match_complement(side_2, a.side_1))
      return or2tc(side_2, a.side_2);
    if (match_complement(side_2, a.side_2))
      return or2tc(side_2, a.side_1);
  }

  // (x || y) || y = x || y, (x || y) || x = x || y, and the symmetric forms
  // with the outer or's operands swapped. Idempotence after associativity.
  if (is_or2t(side_1))
  {
    const or2t &inner = to_or2t(side_1);
    if (inner.side_1 == side_2 || inner.side_2 == side_2)
      return side_1;
  }
  if (is_or2t(side_2))
  {
    const or2t &inner = to_or2t(side_2);
    if (inner.side_1 == side_1 || inner.side_2 == side_1)
      return side_2;
  }

  // Try associative simplification: (x || a) || (x || b) = x || (a || b)
  expr2tc simplified = simplify_associative_binary_op<or2t>(
    type,
    side_1,
    side_2,
    is_or2t,
    to_or2t,
    [](const expr2tc &left, const expr2tc &right) {
      return or2tc(left, right);
    });

  if (!is_nil_expr(simplified))
    return simplified;

  // Otherwise, default
  return simplify_logic_2ops<Ortor, or2t>(type, side_1, side_2);
}

template <class constant_type>
struct Xortor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    if (is_constant(op1))
    {
      expr2tc c1 = op1;
      // False? Simplify to op2
      if (get_value(c1) == 0)
        return op2;
    }

    if (is_constant(op2))
    {
      expr2tc c2 = op2;
      // False? Simplify to op1
      if (get_value(c2) == 0)
        return op1;
    }

    // Two constants? Simplify to result of the xor
    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(!(get_value(c1) == 0) ^ !(get_value(c2) == 0));
    }

    return expr2tc();
  }
};

expr2tc xor2t::do_simplify() const
{
  // Constant short-circuits. xor2t result is always bool, so a true constant
  // toggles the other side and a false constant is the identity.
  if (is_constant_bool2t(side_1))
    return to_constant_bool2t(side_1).value ? not2tc(side_2) : side_2;
  if (is_constant_bool2t(side_2))
    return to_constant_bool2t(side_2).value ? not2tc(side_1) : side_1;

  // x ^ x = false (self-xor)
  if (side_1 == side_2)
    return gen_false_expr();

  // x ^ !x = true, !x ^ x = true (complementary)
  if (is_not2t(side_1) && to_not2t(side_1).value == side_2)
    return gen_true_expr();
  if (is_not2t(side_2) && to_not2t(side_2).value == side_1)
    return gen_true_expr();

  // !x ^ !y = x ^ y (double-negation cancels through xor)
  if (is_not2t(side_1) && is_not2t(side_2))
    return xor2tc(to_not2t(side_1).value, to_not2t(side_2).value);

  // (x ^ y) ^ y = x and three symmetric forms (cancellation through assoc)
  if (is_xor2t(side_1))
  {
    const xor2t &inner = to_xor2t(side_1);
    if (inner.side_2 == side_2)
      return inner.side_1;
    if (inner.side_1 == side_2)
      return inner.side_2;
  }
  if (is_xor2t(side_2))
  {
    const xor2t &inner = to_xor2t(side_2);
    if (inner.side_2 == side_1)
      return inner.side_1;
    if (inner.side_1 == side_1)
      return inner.side_2;
  }

  return simplify_logic_2ops<Xortor, xor2t>(type, side_1, side_2);
}

template <class constant_type>
struct Impliestor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // False => * evaluate to true, always
    if (is_constant(op1))
    {
      expr2tc c1 = op1;
      if (get_value(c1) == 0)
        return gen_true_expr();
    }

    // Otherwise, the only other thing that will make this expr always true is
    // if side 2 is true.
    if (is_constant(op2))
    {
      expr2tc c2 = op2;
      if (!(get_value(c2) == 0))
        return gen_true_expr();
    }

    return expr2tc();
  }
};

expr2tc implies2t::do_simplify() const
{
  // Constant short-circuits. False antecedent makes the implication trivially
  // true; true consequent likewise. False consequent reduces to !antecedent;
  // true antecedent reduces to the consequent.
  if (is_constant_bool2t(side_1))
    return to_constant_bool2t(side_1).value ? side_2 : gen_true_expr();
  if (is_constant_bool2t(side_2))
    return to_constant_bool2t(side_2).value ? gen_true_expr() : not2tc(side_1);

  // x => x = true (self-implication)
  if (side_1 == side_2)
    return gen_true_expr();

  // !x => x = x and x => !x = !x. Implication subsumes the antecedent into
  // the consequent's truth value.
  if (is_not2t(side_1) && to_not2t(side_1).value == side_2)
    return side_2;
  if (is_not2t(side_2) && to_not2t(side_2).value == side_1)
    return side_2;

  return simplify_logic_2ops<Impliestor, implies2t>(type, side_1, side_2);
}

template <typename constructor, typename U64Op>
static expr2tc do_bit_munge_operation(
  U64Op opfunc,
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2)
{
  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so the operands are already simplified.
  const expr2tc &simplified_side_1 = side_1;
  const expr2tc &simplified_side_2 = side_2;

  /* Only support constant folding for integer and's. If you're a float,
   * pointer, or whatever, you're on your own. */
  if (
    is_constant_int2t(simplified_side_1) &&
    is_constant_int2t(simplified_side_2) && type->get_width() <= 64)
  {
    // So - we can't make BigInt by itself do the operation. But we can try to
    // map it to the corresponding operation on our native types.
    const constant_int2t &int1 = to_constant_int2t(simplified_side_1);
    const constant_int2t &int2 = to_constant_int2t(simplified_side_2);
    const BigInt &bl = int1.value;
    const BigInt &br = int2.value;

    /* The bit pattern in two's complement. Pick to_uint64()/to_int64()
     * based on each operand's own type so we never cross the int64 sign
     * boundary on an unsignedbv-with-bit-63-set value. The cast back to
     * uint64_t recovers the same bit pattern either way, but using the
     * type-matching accessor is robust under future BigInt changes. Bail
     * if the value can't be represented in the chosen 64-bit form. */
    auto fits_uint64 = [](const expr2tc &e, const BigInt &v) {
      return is_unsignedbv_type(e) ? v.is_uint64() : v.is_int64();
    };
    if (
      !fits_uint64(simplified_side_1, bl) ||
      !fits_uint64(simplified_side_2, br))
      return expr2tc();
    uint64_t l = is_unsignedbv_type(simplified_side_1)
                   ? bl.to_uint64()
                   : (uint64_t)bl.to_int64();
    uint64_t r = is_unsignedbv_type(simplified_side_2)
                   ? br.to_uint64()
                   : (uint64_t)br.to_int64();

    bool is_shift = false;
    bool can_eval = true;

    /* For &, |, ^, ~, << our low bits are determined only by the corresponding
     * low bits of the arguments, however for >> this is not necessarily true!
     */
    if constexpr (
      std::is_same_v<constructor, lshr2t> ||
      std::is_same_v<constructor, ashr2t>)
    {
      /* do we have a small enough LHS to evaluate on uint64_t? */
      can_eval &=
        br == 0 ||
        (is_signedbv_type(simplified_side_1) ? bl.is_int64() : bl.is_uint64());
      is_shift = true;
    }
    else
      is_shift = std::is_same_v<constructor, shl2t>;

    /* TODO fbrausse: In C, a << b is undefined for signed a if the result is
     * not representable. */

    /* Evaluating shifts with the shift amount >= 64 on (u)int64_t is undefined
     * behavior in C++, we should avoid doing that during simplification. */
    can_eval &= !is_shift || br < 64;
    if (can_eval)
    {
      uint64_t res = opfunc(l, r);

      uint64_t trunc_mask = 0;
      if (type->get_width() < 64)
      {
        // truncate the result to the type's width
        trunc_mask = ~(uint64_t)0 << type->get_width();
        res &= ~trunc_mask;
      }

      BigInt z;
      if (is_signedbv_type(type))
      {
        // if res's sign-bit is set, sign-extend it
        if (res >> (type->get_width() - 1))
          res |= trunc_mask;
        z = BigInt((int64_t)res);
      }
      else
        z = BigInt((uint64_t)res);

      return constant_int2tc(type, z);
    }
  }

  return expr2tc();
}

expr2tc bitand2t::do_simplify() const
{
  if (side_1 == side_2)
    return side_1; // x & x = x

  // x & ~x = 0
  if (is_bitnot2t(side_1) && to_bitnot2t(side_1).value == side_2)
    return gen_zero(type);
  if (is_bitnot2t(side_2) && to_bitnot2t(side_2).value == side_1)
    return gen_zero(type);

  // Scalar identity/absorber shortcuts. Skip when the result type is a
  // vector — returning a scalar `side_1`/`side_2` from a vector-typed op
  // would corrupt the type. Vector shapes go through the
  // distribute_vector_operation path below.
  if (!is_vector_type(type))
  {
    // 0 & x = 0, all1 & x = x
    if (is_constant_int2t(side_1))
    {
      if (to_constant_int2t(side_1).value.is_zero())
        return side_1; // 0 & x = 0
    }
    if (is_all_ones_constant(side_1))
      return side_2; // all1 & x = x

    // x & 0 = 0, x & all1 = x
    if (is_constant_int2t(side_2))
    {
      if (to_constant_int2t(side_2).value.is_zero())
        return side_2; // x & 0 = 0
    }
    if (is_all_ones_constant(side_2))
      return side_1; // x & all1 = x
  }

  // (x & y) & y = x & y, (x & y) & x = x & y, and the symmetric forms with
  // the outer and's operands swapped. Idempotence after associativity.
  if (is_bitand2t(side_1))
  {
    const bitand2t &inner = to_bitand2t(side_1);
    if (inner.side_1 == side_2 || inner.side_2 == side_2)
      return side_1;
  }
  if (is_bitand2t(side_2))
  {
    const bitand2t &inner = to_bitand2t(side_2);
    if (inner.side_1 == side_1 || inner.side_2 == side_1)
      return side_2;
  }

  // Absorption: x & (x | y) = x
  if (is_bitor2t(side_2))
  {
    const bitor2t &bor = to_bitor2t(side_2);
    if (bor.side_1 == side_1 || bor.side_2 == side_1)
      return side_1;
  }
  if (is_bitor2t(side_1))
  {
    const bitor2t &bor = to_bitor2t(side_1);
    if (bor.side_1 == side_2 || bor.side_2 == side_2)
      return side_2;
  }

  // (a | ~b) & (a | b) -> a
  if (is_bitor2t(side_1) && is_bitor2t(side_2))
  {
    const bitor2t &or1 = to_bitor2t(side_1);
    const bitor2t &or2 = to_bitor2t(side_2);

    // Check if one expr is the bitwise not of another
    auto is_complement = [](const expr2tc &a, const expr2tc &b) -> bool {
      return (is_bitnot2t(a) && to_bitnot2t(a).value == b) ||
             (is_bitnot2t(b) && to_bitnot2t(b).value == a);
    };

    // Check all four combinations: we're looking for a common operand 'a'
    // and complementary operands 'b' and '~b'

    // Case 1: or1.side_1 is the common operand
    if (or1.side_1 == or2.side_1 && is_complement(or1.side_2, or2.side_2))
      return or1.side_1;
    if (or1.side_1 == or2.side_2 && is_complement(or1.side_2, or2.side_1))
      return or1.side_1;

    // Case 2: or1.side_2 is the common operand
    if (or1.side_2 == or2.side_1 && is_complement(or1.side_1, or2.side_2))
      return or1.side_2;
    if (or1.side_2 == or2.side_2 && is_complement(or1.side_1, or2.side_1))
      return or1.side_2;
  }

  auto op = [](uint64_t op1, uint64_t op2) { return (op1 & op2); };

  // Is a vector operation ? Apply the op
  if (is_constant_vector2t(side_1) || is_constant_vector2t(side_2))
  {
    auto op = [](type2tc t, expr2tc e1, expr2tc e2) {
      return bitand2tc(t, e1, e2);
    };
    return distribute_vector_operation(op, side_1, side_2);
  }

  return do_bit_munge_operation<bitand2t>(op, type, side_1, side_2);
}

expr2tc bitor2t::do_simplify() const
{
  if (side_1 == side_2)
    return side_1; // x | x = x

  // Scalar identity/absorber shortcuts. Skip when the result type is a
  // vector — returning scalar all-ones / `side_1`/`side_2` from a
  // vector-typed op would corrupt the type. Vector shapes go through the
  // distribute_vector_operation path below.
  if (!is_vector_type(type))
  {
    // x | ~x = all1 (all bits set). Use the type's all-ones value so the
    // result preserves type-correct width for unsigned types.
    auto make_all_ones = [&](const type2tc &t) -> expr2tc {
      if (is_unsignedbv_type(t))
        return constant_int2tc(t, BigInt::power2(t->get_width()) - 1);
      return constant_int2tc(t, BigInt(-1));
    };
    if (is_bitnot2t(side_1) && to_bitnot2t(side_1).value == side_2)
      return make_all_ones(type);
    if (is_bitnot2t(side_2) && to_bitnot2t(side_2).value == side_1)
      return make_all_ones(type);

    // 0 | x = x, all1 | x = all1
    if (is_constant_int2t(side_1))
    {
      if (to_constant_int2t(side_1).value.is_zero())
        return side_2; // 0 | x = x
    }
    if (is_all_ones_constant(side_1))
      return side_1; // all1 | x = all1

    // x | 0 = x, x | all1 = all1
    if (is_constant_int2t(side_2))
    {
      if (to_constant_int2t(side_2).value.is_zero())
        return side_1; // x | 0 = x
    }
    if (is_all_ones_constant(side_2))
      return side_2; // x | all1 = all1
  }

  // Absorption: x | (x & y) = x
  if (is_bitand2t(side_2))
  {
    const bitand2t &band = to_bitand2t(side_2);
    if (band.side_1 == side_1 || band.side_2 == side_1)
      return side_1;
  }
  if (is_bitand2t(side_1))
  {
    const bitand2t &band = to_bitand2t(side_1);
    if (band.side_1 == side_2 || band.side_2 == side_2)
      return side_2;
  }

  // (x | y) | y = x | y, (x | y) | x = x | y, and the symmetric forms with
  // the outer or's operands swapped. Idempotence after associativity.
  if (is_bitor2t(side_1))
  {
    const bitor2t &inner = to_bitor2t(side_1);
    if (inner.side_1 == side_2 || inner.side_2 == side_2)
      return side_1;
  }
  if (is_bitor2t(side_2))
  {
    const bitor2t &inner = to_bitor2t(side_2);
    if (inner.side_1 == side_1 || inner.side_2 == side_1)
      return side_2;
  }

  // Helper lambdas used by multiple simplifications
  auto is_complement = [](const expr2tc &a, const expr2tc &b) -> bool {
    return (is_bitnot2t(a) && to_bitnot2t(a).value == b) ||
           (is_bitnot2t(b) && to_bitnot2t(b).value == a);
  };

  auto unwrap_if_not = [](const expr2tc &e) -> expr2tc {
    if (is_bitnot2t(e))
      return to_bitnot2t(e).value;
    return expr2tc();
  };

  // (a & ~b) | (a ^ b) --> a ^ b
  // (a ^ b) | (a & ~b) --> a ^ b (symmetric case handled together)
  if (
    (is_bitand2t(side_1) && is_bitxor2t(side_2)) ||
    (is_bitxor2t(side_1) && is_bitand2t(side_2)))
  {
    const expr2tc &band_expr = is_bitand2t(side_1) ? side_1 : side_2;
    const expr2tc &bxor_expr = is_bitxor2t(side_1) ? side_1 : side_2;

    const bitand2t &band = to_bitand2t(band_expr);
    const bitxor2t &bxor = to_bitxor2t(bxor_expr);

    // Check all combinations where one AND operand matches one XOR operand
    // and the other AND operand is the complement of the other XOR operand
    if (band.side_1 == bxor.side_1 && is_complement(band.side_2, bxor.side_2))
      return bxor_expr;

    if (band.side_1 == bxor.side_2 && is_complement(band.side_2, bxor.side_1))
      return bxor_expr;

    if (band.side_2 == bxor.side_1 && is_complement(band.side_1, bxor.side_2))
      return bxor_expr;

    if (band.side_2 == bxor.side_2 && is_complement(band.side_1, bxor.side_1))
      return bxor_expr;
  }

  // (~a & b) | ~(a | b) --> ~a
  // ~(a | b) | (~a & b) --> ~a (symmetric case handled together)
  if (
    (is_bitand2t(side_1) && is_bitnot2t(side_2)) ||
    (is_bitnot2t(side_1) && is_bitand2t(side_2)))
  {
    const expr2tc &band_expr = is_bitand2t(side_1) ? side_1 : side_2;
    const expr2tc &bnot_expr = is_bitnot2t(side_1) ? side_1 : side_2;

    const bitand2t &band = to_bitand2t(band_expr);
    const bitnot2t &bnot = to_bitnot2t(bnot_expr);

    // Check if the NOT operand is an OR
    if (is_bitor2t(bnot.value))
    {
      const bitor2t &bor = to_bitor2t(bnot.value);

      // Case 1: band.side_1 = ~a, band.side_2 = b
      expr2tc unwrapped1 = unwrap_if_not(band.side_1);
      if (!is_nil_expr(unwrapped1))
      {
        // Check if OR contains both unwrapped1 (a) and band.side_2 (b)
        if (
          (bor.side_1 == unwrapped1 && bor.side_2 == band.side_2) ||
          (bor.side_2 == unwrapped1 && bor.side_1 == band.side_2))
          return band.side_1; // return ~a
      }

      // Case 2: band.side_2 = ~a, band.side_1 = b
      expr2tc unwrapped2 = unwrap_if_not(band.side_2);
      if (!is_nil_expr(unwrapped2))
      {
        // Check if OR contains both unwrapped2 (a) and band.side_1 (b)
        if (
          (bor.side_1 == unwrapped2 && bor.side_2 == band.side_1) ||
          (bor.side_2 == unwrapped2 && bor.side_1 == band.side_1))
          return band.side_2; // return ~a
      }
    }
  }

  // (~a ^ b) | (a & b) --> ~a ^ b
  // (a & b) | (~a ^ b) --> ~a ^ b (symmetric case handled together)
  if (
    (is_bitxor2t(side_1) && is_bitand2t(side_2)) ||
    (is_bitand2t(side_1) && is_bitxor2t(side_2)))
  {
    const expr2tc &bxor_expr = is_bitxor2t(side_1) ? side_1 : side_2;
    const expr2tc &band_expr = is_bitand2t(side_1) ? side_1 : side_2;

    const bitxor2t &bxor = to_bitxor2t(bxor_expr);
    const bitand2t &band = to_bitand2t(band_expr);

    // Case 1: bxor.side_1 = ~a, bxor.side_2 = b
    expr2tc unwrapped1 = unwrap_if_not(bxor.side_1);
    if (!is_nil_expr(unwrapped1))
    {
      // Check if AND contains both unwrapped1 (a) and bxor.side_2 (b)
      if (
        (band.side_1 == unwrapped1 && band.side_2 == bxor.side_2) ||
        (band.side_2 == unwrapped1 && band.side_1 == bxor.side_2))
        return bxor_expr;
    }

    // Case 2: bxor.side_2 = ~a, bxor.side_1 = b
    expr2tc unwrapped2 = unwrap_if_not(bxor.side_2);
    if (!is_nil_expr(unwrapped2))
    {
      // Check if AND contains both unwrapped2 (a) and bxor.side_1 (b)
      if (
        (band.side_1 == unwrapped2 && band.side_2 == bxor.side_1) ||
        (band.side_2 == unwrapped2 && band.side_1 == bxor.side_1))
        return bxor_expr;
    }
  }

  // (a ^ b) | (a | b) --> a | b
  // (a | b) | (a ^ b) --> a | b (symmetric case handled together)
  if (
    (is_bitxor2t(side_1) && is_bitor2t(side_2)) ||
    (is_bitor2t(side_1) && is_bitxor2t(side_2)))
  {
    const expr2tc &xor_expr_ptr = is_bitxor2t(side_1) ? side_1 : side_2;
    const expr2tc &or_expr_ptr = is_bitor2t(side_1) ? side_1 : side_2;

    const bitxor2t &xor_expr = to_bitxor2t(xor_expr_ptr);
    const bitor2t &or_expr = to_bitor2t(or_expr_ptr);

    // Check if XOR operands match OR operands (any order)
    bool match =
      (xor_expr.side_1 == or_expr.side_1 &&
       xor_expr.side_2 == or_expr.side_2) ||
      (xor_expr.side_1 == or_expr.side_2 && xor_expr.side_2 == or_expr.side_1);

    if (match)
    {
      // Return the OR subexpression unchanged. An earlier version sorted
      // its operands by raw shared_ptr address to canonicalize the shape,
      // but that depended on allocation order — making the simplified
      // tree, pretty-print, and CRC/hash output non-deterministic across
      // runs. The OR expression already exists in its natural form;
      // re-emitting it here would just churn the IR.
      return or_expr_ptr;
    }
  }

  // ~(a ^ b) | (a & b) --> ~(a ^ b)
  if (
    (is_bitnot2t(side_1) && is_bitand2t(side_2)) ||
    (is_bitand2t(side_1) && is_bitnot2t(side_2)))
  {
    const expr2tc &bnot_expr = is_bitnot2t(side_1) ? side_1 : side_2;
    const expr2tc &band_expr = is_bitand2t(side_1) ? side_1 : side_2;

    const bitnot2t &bnot = to_bitnot2t(bnot_expr);
    const bitand2t &band = to_bitand2t(band_expr);

    // Check if the NOT operand is an XOR
    if (is_bitxor2t(bnot.value))
    {
      const bitxor2t &bxor = to_bitxor2t(bnot.value);

      // Check if AND operands match XOR operands (any order)
      bool match = (bxor.side_1 == band.side_1 && bxor.side_2 == band.side_2) ||
                   (bxor.side_1 == band.side_2 && bxor.side_2 == band.side_1);

      if (match)
        return bnot_expr; // Return ~(a ^ b)
    }
  }

  // ~(a & b) | (a ^ b) --> ~(a & b)
  if (
    (is_bitnot2t(side_1) && is_bitxor2t(side_2)) ||
    (is_bitxor2t(side_1) && is_bitnot2t(side_2)))
  {
    const expr2tc &bnot_expr = is_bitnot2t(side_1) ? side_1 : side_2;
    const expr2tc &bxor_expr = is_bitxor2t(side_1) ? side_1 : side_2;

    const bitnot2t &bnot = to_bitnot2t(bnot_expr);
    const bitxor2t &bxor = to_bitxor2t(bxor_expr);

    // Check if the NOT operand is an AND
    if (is_bitand2t(bnot.value))
    {
      const bitand2t &band = to_bitand2t(bnot.value);

      // Check if XOR operands match AND operands (any order)
      bool match = (bxor.side_1 == band.side_1 && bxor.side_2 == band.side_2) ||
                   (bxor.side_1 == band.side_2 && bxor.side_2 == band.side_1);

      if (match)
        return bnot_expr; // Return ~(a & b)
    }
  }

  auto op = [](uint64_t op1, uint64_t op2) { return (op1 | op2); };

  // Is a vector operation ? Apply the op
  if (is_constant_vector2t(side_1) || is_constant_vector2t(side_2))
  {
    auto op = [](type2tc t, expr2tc e1, expr2tc e2) {
      return bitor2tc(t, e1, e2);
    };
    return distribute_vector_operation(op, side_1, side_2);
  }

  return do_bit_munge_operation<bitor2t>(op, type, side_1, side_2);
}

expr2tc bitxor2t::do_simplify() const
{
  // x ^ x = 0 (gen_zero is type-correct for vectors too)
  if (side_1 == side_2)
    return gen_zero(type);

  // Scalar identity/absorber shortcuts. Skip when type is a vector — a
  // scalar `constant_int2tc(type, -1)` or a scalar operand return would
  // corrupt the vector type. distribute_vector_operation below handles
  // those shapes.
  if (!is_vector_type(type))
  {
    // x ^ ~x = all1, ~x ^ x = all1 (complementary toggle covers all bits)
    auto make_all_ones = [&](const type2tc &t) -> expr2tc {
      if (is_unsignedbv_type(t))
        return constant_int2tc(t, BigInt::power2(t->get_width()) - 1);
      return constant_int2tc(t, BigInt(-1));
    };
    if (is_bitnot2t(side_1) && to_bitnot2t(side_1).value == side_2)
      return make_all_ones(type);
    if (is_bitnot2t(side_2) && to_bitnot2t(side_2).value == side_1)
      return make_all_ones(type);

    // x ^ 0 = x
    if (is_constant_int2t(side_1) && to_constant_int2t(side_1).value.is_zero())
      return side_2;
    // 0 ^ x = x
    if (is_constant_int2t(side_2) && to_constant_int2t(side_2).value.is_zero())
      return side_1;

    // x ^ all1 = ~x, all1 ^ x = ~x (toggle-all-bits)
    if (is_all_ones_constant(side_2))
      return bitnot2tc(type, side_1);
    if (is_all_ones_constant(side_1))
      return bitnot2tc(type, side_2);
  }

  // ~x ^ ~y = x ^ y
  if (is_bitnot2t(side_1) && is_bitnot2t(side_2))
    return bitxor2tc(
      type, to_bitnot2t(side_1).value, to_bitnot2t(side_2).value);

  // (x ^ y) ^ y = x, (x ^ y) ^ x = y, and the symmetric forms with the
  // outer xor's operands swapped.
  if (is_bitxor2t(side_1))
  {
    const bitxor2t &inner = to_bitxor2t(side_1);
    if (inner.side_2 == side_2)
      return inner.side_1;
    if (inner.side_1 == side_2)
      return inner.side_2;
  }
  if (is_bitxor2t(side_2))
  {
    const bitxor2t &inner = to_bitxor2t(side_2);
    if (inner.side_2 == side_1)
      return inner.side_1;
    if (inner.side_1 == side_1)
      return inner.side_2;
  }

  auto op = [](uint64_t op1, uint64_t op2) { return (op1 ^ op2); };

  // Is a vector operation ? Apply the op
  if (is_constant_vector2t(side_1) || is_constant_vector2t(side_2))
  {
    auto op = [](type2tc t, expr2tc e1, expr2tc e2) {
      return bitxor2tc(t, e1, e2);
    };
    return distribute_vector_operation(op, side_1, side_2);
  }

  return do_bit_munge_operation<bitxor2t>(op, type, side_1, side_2);
}

expr2tc bitnot2t::do_simplify() const
{
  // ~(~x) = x (double complement)
  if (is_bitnot2t(value))
    return to_bitnot2t(value).value;

  // De Morgan's law: ~(x & y) = (~x) | (~y)
  if (is_bitand2t(value))
  {
    const bitand2t &band = to_bitand2t(value);
    return bitor2tc(
      type,
      bitnot2tc(band.side_1->type, band.side_1),
      bitnot2tc(band.side_2->type, band.side_2));
  }

  // De Morgan's law: ~(x | y) = (~x) & (~y)
  if (is_bitor2t(value))
  {
    const bitor2t &bor = to_bitor2t(value);
    return bitand2tc(
      type,
      bitnot2tc(bor.side_1->type, bor.side_1),
      bitnot2tc(bor.side_2->type, bor.side_2));
  }

  auto op = [](uint64_t op1, uint64_t) { return ~(op1); };

  if (is_constant_vector2t(value))
  {
    constant_vector2t vector = to_constant_vector2t(value); // copy
    for (size_t i = 0; i < vector.datatype_members.size(); i++)
    {
      auto &op = vector.datatype_members[i];
      vector.datatype_members[i] = bitnot2tc(op->type, op);
    }
    return constant_vector2tc(std::move(vector));
  }

  return do_bit_munge_operation<bitnot2t>(op, type, value, value);
}

/// Try to combine `outer(inner(x, c1), c2)` where both `c1` and `c2` are
/// constant ints into `outer(x, c1 + c2)`. Returns nil unless every operand
/// is well-shaped and the combined amount is strictly less than the result
/// width — beyond that, C semantics make the original UB and we must not
/// mask the runtime overflow check.
template <class ShiftT>
static expr2tc combine_constant_shifts(
  const type2tc &type,
  const expr2tc &outer_lhs,
  const expr2tc &outer_amt,
  bool (*is_inner)(const expr2tc &),
  const ShiftT &(*to_inner)(const expr2tc &))
{
  // Vector shifts: type->get_width() reports the *total* vector width, not
  // per-lane. Combining (v4u8 << c1) << c2 with sum < 32 would still wrap
  // each 8-bit lane past its width and mask shift UB. Skip vectors here
  // and let do_bit_munge_operation handle them via vector distribution.
  if (is_vector_type(type))
    return expr2tc();

  if (!is_constant_int2t(outer_amt) || !is_inner(outer_lhs))
    return expr2tc();

  const ShiftT &inner = to_inner(outer_lhs);
  if (!is_constant_int2t(inner.side_2))
    return expr2tc();

  const BigInt &c1 = to_constant_int2t(inner.side_2).value;
  const BigInt &c2 = to_constant_int2t(outer_amt).value;
  if (c1.is_negative() || c2.is_negative())
    return expr2tc();

  BigInt sum = c1 + c2;
  if (sum >= BigInt(type->get_width()))
    return expr2tc();

  // The combined count must also fit in the shift-count operand's type
  // without truncation. ESBMC's IR allows the shift-count type to be
  // narrower than the lhs (the SMT converter zero-extends it to the lhs
  // width). If the combined sum doesn't fit in outer_amt->type, building
  // from_integer(sum, outer_amt->type) would silently wrap. Example:
  // u128 x << u5(20) << u5(20) — sum = 40, outer_amt is u5, 40 wraps to
  // 8, producing the wrong shift.
  const unsigned amt_width = outer_amt->type->get_width();
  const bool amt_signed = is_signedbv_type(outer_amt->type);
  if (!fits_in_width(sum, amt_width, amt_signed))
    return expr2tc();

  return expr2tc(std::make_shared<ShiftT>(
    type, inner.side_1, from_integer(sum, outer_amt->type)));
}

expr2tc shl2t::do_simplify() const
{
  // x << 0 = x
  if (is_constant_int2t(side_2) && to_constant_int2t(side_2).value.is_zero())
    return side_1;

  // 0 << x = 0
  if (is_constant_int2t(side_1) && to_constant_int2t(side_1).value.is_zero())
    return side_1;

  // (x << c1) << c2 -> x << (c1 + c2) when c1 + c2 < width.
  if (
    expr2tc combined =
      combine_constant_shifts<shl2t>(type, side_1, side_2, is_shl2t, to_shl2t))
    return combined;

  auto op = [](uint64_t op1, uint64_t op2) { return op1 << op2; };

  if (is_constant_vector2t(side_1) || is_constant_vector2t(side_2))
  {
    auto op = [](type2tc t, expr2tc e1, expr2tc e2) {
      return shl2tc(t, e1, e2);
    };
    return distribute_vector_operation(op, side_1, side_2);
  }

  return do_bit_munge_operation<shl2t>(op, type, side_1, side_2);
}

expr2tc lshr2t::do_simplify() const
{
  // x >> 0 = x
  if (is_constant_int2t(side_2) && to_constant_int2t(side_2).value.is_zero())
    return side_1;

  // 0 >> x = 0
  if (is_constant_int2t(side_1) && to_constant_int2t(side_1).value.is_zero())
    return side_1;

  auto op = [](uint64_t op1, uint64_t op2) { return op1 >> op2; };

  if (is_constant_vector2t(side_1) || is_constant_vector2t(side_2))
  {
    auto op = [](type2tc t, expr2tc e1, expr2tc e2) {
      return lshr2tc(t, e1, e2);
    };
    return distribute_vector_operation(op, side_1, side_2);
  }

  return do_bit_munge_operation<lshr2t>(op, type, side_1, side_2);
}

expr2tc ashr2t::do_simplify() const
{
  // x >> 0 = x
  if (is_constant_int2t(side_2) && to_constant_int2t(side_2).value.is_zero())
    return side_1;

  // 0 >>s x = 0 (arithmetic right shift of zero is still zero)
  if (is_constant_int2t(side_1) && to_constant_int2t(side_1).value.is_zero())
    return side_1;

  // (x >>s c1) >>s c2 -> x >>s (c1 + c2) when c1 + c2 < width. Sound for
  // arithmetic right shift: stacking sign-extending shifts is associative
  // up to the width boundary; at or past width the result is all sign-bit
  // copies, but C UB rules apply and we leave that to overflow checking.
  if (
    expr2tc combined = combine_constant_shifts<ashr2t>(
      type, side_1, side_2, is_ashr2t, to_ashr2t))
    return combined;

  auto op = [](uint64_t op1, uint64_t op2) {
    /* simulating the arithmetic right shift in C++ requires LHS to be signed */
    return (int64_t)op1 >> op2;
  };

  // Is a vector operation ? Apply the op
  if (is_constant_vector2t(side_1) || is_constant_vector2t(side_2))
  {
    auto op = [](type2tc t, expr2tc e1, expr2tc e2) {
      return ashr2tc(t, e1, e2);
    };
    return distribute_vector_operation(op, side_1, side_2);
  }

  return do_bit_munge_operation<ashr2t>(op, type, side_1, side_2);
}

expr2tc bitcast2t::do_simplify() const
{
  // Follow approach of old irep, i.e., copy it
  if (type == from->type)
  {
    // Bitcast to same type means this can be eliminated entirely
    return from;
  }

  // bitcast(bitcast(x, T1), T2) chain-collapse intentionally NOT applied:
  // when the SMT pointer-arith pipeline has cast a pointer through ulong
  // and back, the round-trip is load-bearing — collapsing it produces a
  // logically-equivalent but solver-harder formula because the bitcast
  // boundary is what tells smt_memspace.cpp to materialize pointer object
  // and offset components (without it, every pointer-as-pointer use stays
  // an array index, which generates a much larger case-split tree).

  // This should be fine, just use typecast
  if (
    !is_floatbv_type(type) && !is_floatbv_type(from->type) &&
    !is_fixedbv_type(type) && !is_fixedbv_type(from->type))
    return typecast2tc(type, from)->do_simplify();

  return expr2tc();
}

expr2tc typecast2t::do_simplify() const
{
  // Follow approach of old irep, i.e., copy it
  if (type == from->type)
  {
    // Typecast to same type means this can be eliminated entirely
    return from;
  }

  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so `from` is already simplified.
  const expr2tc &simp = from;

  if (is_constant_expr(simp))
  {
    // Casts from constant operands can be done here.
    if (is_bool_type(simp) && is_number_type(type))
    {
      // bool to int
      if (is_bv_type(type))
        return constant_int2tc(type, BigInt(to_constant_bool2t(simp).value));

      if (is_fixedbv_type(type))
      {
        fixedbvt fbv;
        fbv.spec = to_fixedbv_type(migrate_type_back(type));
        fbv.from_integer(to_constant_bool2t(simp).value);
        return constant_fixedbv2tc(fbv);
      }

      if (is_floatbv_type(type))
      {
        if (!is_constant_int2t(rounding_mode))
          return expr2tc();

        ieee_floatt fpbv;

        BigInt rm_value = to_constant_int2t(rounding_mode).value;
        fpbv.rounding_mode = ieee_floatt::rounding_modet(rm_value.to_int64());

        // Bool -> float: convert the bool's integer value (0 or 1) into
        // a float of the destination type.
        fpbv.spec = ieee_float_spect(to_floatbv_type(migrate_type_back(type)));
        fpbv.from_integer(BigInt(to_constant_bool2t(simp).value));

        return constant_floatbv2tc(fpbv);
      }
    }
    else if (is_bv_type(simp) && is_number_type(type))
    {
      // int to int/float/double
      const constant_int2t &theint = to_constant_int2t(simp);

      if (is_bv_type(type))
      {
        // If we are typecasting from integer to a smaller integer,
        // this will return the number with the smaller size
        exprt number = from_integer(theint.value, migrate_type_back(type));

        BigInt new_number;
        if (to_integer(number, new_number))
          return expr2tc();

        return constant_int2tc(type, new_number);
      }

      if (is_fixedbv_type(type))
      {
        fixedbvt fbv;
        fbv.spec = to_fixedbv_type(migrate_type_back(type));
        fbv.from_integer(theint.value);
        return constant_fixedbv2tc(fbv);
      }

      if (is_bool_type(type))
      {
        const constant_int2t &theint = to_constant_int2t(simp);
        return theint.value.is_zero() ? gen_false_expr() : gen_true_expr();
      }

      if (is_floatbv_type(type))
      {
        if (!is_constant_int2t(rounding_mode))
          return expr2tc();

        ieee_floatt fpbv;

        BigInt rm_value = to_constant_int2t(rounding_mode).value;
        fpbv.rounding_mode = ieee_floatt::rounding_modet(rm_value.to_int64());

        fpbv.spec = to_floatbv_type(migrate_type_back(type));
        fpbv.from_integer(to_constant_int2t(simp).value);

        return constant_floatbv2tc(fpbv);
      }
    }
    else if (is_fixedbv_type(simp) && is_number_type(type))
    {
      // float/double to int/float/double
      fixedbvt fbv(to_constant_fixedbv2t(simp).value);

      if (is_bv_type(type))
        return constant_int2tc(type, fbv.to_integer());

      if (is_fixedbv_type(type))
      {
        fbv.round(to_fixedbv_type(migrate_type_back(type)));
        return constant_fixedbv2tc(fbv);
      }

      if (is_bool_type(type))
      {
        const constant_fixedbv2t &fbv = to_constant_fixedbv2t(simp);
        return fbv.value.is_zero() ? gen_false_expr() : gen_true_expr();
      }
    }
    else if (is_floatbv_type(simp) && is_number_type(type))
    {
      // float/double to int/float/double
      if (!is_constant_int2t(rounding_mode))
        return expr2tc();

      ieee_floatt fpbv(to_constant_floatbv2t(simp).value);

      BigInt rm_value = to_constant_int2t(rounding_mode).value;
      fpbv.rounding_mode = ieee_floatt::rounding_modet(rm_value.to_int64());

      if (is_bv_type(type))
        return constant_int2tc(type, fpbv.to_integer());

      if (is_floatbv_type(type))
      {
        fpbv.change_spec(to_floatbv_type(migrate_type_back(type)));
        return constant_floatbv2tc(fpbv);
      }

      if (is_bool_type(type))
        return fpbv.is_zero() ? gen_false_expr() : gen_true_expr();
    }
  }
  else if (is_bool_type(type))
  {
    // Bool type -> turn into inequality with zero. Building notequal directly
    // avoids constructing an intermediate `not(equality(...))` that the not2t
    // simplifier would immediately collapse.
    exprt zero = gen_zero(migrate_type_back(simp->type));

    expr2tc zero2;
    migrate_expr(zero, zero2);

    return notequal2tc(simp, zero2);
  }
  else if (is_pointer_type(type) && is_pointer_type(simp))
  {
    // Casting from one pointer to another is meaningless... except when there's
    // pointer arithmetic about to be applied to it. So, only remove typecasts
    // that don't change the subtype width.
    const pointer_type2t &ptr_to = to_pointer_type(type);
    const pointer_type2t &ptr_from = to_pointer_type(simp->type);

    if (
      is_symbol_type(ptr_to.subtype) || is_symbol_type(ptr_from.subtype) ||
      is_code_type(ptr_to.subtype) || is_code_type(ptr_from.subtype))
      return expr2tc(); // Not worth thinking about

    if (
      is_array_type(ptr_to.subtype) &&
      is_symbol_type(get_array_subtype(ptr_to.subtype)))
      return expr2tc(); // Not worth thinking about

    if (
      is_array_type(ptr_from.subtype) &&
      is_symbol_type(get_array_subtype(ptr_from.subtype)))
      return expr2tc(); // Not worth thinking about

    try
    {
      unsigned int to_width =
        (is_empty_type(ptr_to.subtype)) ? 8 : ptr_to.subtype->get_width();
      unsigned int from_width =
        (is_empty_type(ptr_from.subtype)) ? 8 : ptr_from.subtype->get_width();

      if (to_width == from_width)
        return simp;
    }
    catch (const array_type2t::dyn_sized_array_excp &e)
    {
      // Something crazy, and probably C++ based, occurred. Don't attempt to
      // simplify.
    }
    catch (const type2t::symbolic_type_excp &e)
    {
      /* might happen if there is a symbolic type in a ptr's subtype; these
       * have not been squashed by thrash_type_symbols() */
    }

    return expr2tc();
  }
  else if (is_typecast2t(simp) && type == simp->type)
  {
    // Typecast from a typecast can be eliminated. We'll be simplified even
    // further by the caller.
    return typecast2tc(type, to_typecast2t(simp).from);
  }

  return expr2tc();
}

expr2tc nearbyint2t::do_simplify() const
{
  if (!is_number_type(type))
    return expr2tc();

  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so `from` is already simplified.

  // nearbyint(nearbyint(x)) -> nearbyint(x). Rounding to integral is
  // idempotent: the inner result is already integral, so the outer round
  // returns the same value regardless of rounding mode.
  if (is_nearbyint2t(from))
    return typecast_check_return(type, from);

  if (!is_constant_floatbv2t(from))
    return expr2tc();

  // Constants whose values are unaffected by rounding: NaN propagates,
  // signed zero stays itself, and +/- inf stays itself for every IEEE
  // rounding mode. Other finite constants need actual round-to-integral
  // arithmetic — leave that to the SMT layer for now.
  ieee_floatt n = to_constant_floatbv2t(from).value;
  if (n.is_NaN() || n.is_zero() || n.is_infinity())
    return typecast_check_return(type, from);

  return expr2tc();
}

expr2tc address_of2t::do_simplify() const
{
  // NB: address_of never has its operands simplified below its feet for
  // sanity's sake — expr2t::simplify returns nil immediately for address_of_id
  // (see line ~29). This do_simplify is invoked through try_simplification by
  // other simplifiers, so we can't assume `ptr_obj` is already simplified.

  // &(*p) -> p. The C standard guarantees this round-trip (no actual access
  // happens), and dereference2t's result type is the pointee type, so the
  // outer address_of yields back the original pointer's type. Only fire when
  // the types actually match — frontends may insert typecasts that break the
  // structural identity.
  if (is_dereference2t(ptr_obj))
  {
    const expr2tc &p = to_dereference2t(ptr_obj).value;
    if (p->type == type)
      return p;
  }

  // Only attempt to simplify indexes. Whatever we're taking the address of,
  // we can't simplify away the symbol.
  if (is_index2t(ptr_obj))
  {
    const index2t &idx = to_index2t(ptr_obj);
    const pointer_type2t &ptr_type = to_pointer_type(type);

    // Don't simplify &a[0]
    if (
      is_constant_int2t(idx.index) &&
      to_constant_int2t(idx.index).value.is_zero())
      return expr2tc();

    expr2tc new_index = try_simplification(idx.index);
    expr2tc zero = constant_int2tc(index_type2(), BigInt(0));
    expr2tc new_idx = index2tc(idx.type, idx.source_value, zero);
    expr2tc sub_addr_of = address_of2tc(ptr_type.subtype, new_idx);

    return add2tc(type, sub_addr_of, new_index);
  }

  return expr2tc();
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_relations(
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2)
{
  if (!is_number_type(type))
    return expr2tc();

  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so the operands are already simplified.
  const expr2tc &simplified_side_1 = side_1;
  const expr2tc &simplified_side_2 = side_2;

  if (!is_constant(simplified_side_1) || !is_constant(simplified_side_2))
  {
    // Pointer comparison with a shared base: (&x + 1 == &x + 2) => (1 == 2).
    // address = pointer + offset; when the bases match, comparing addresses
    // reduces to comparing offsets.
    if (
      is_add2t(simplified_side_1) && is_add2t(simplified_side_2) &&
      is_pointer_type(simplified_side_1) && is_pointer_type(simplified_side_2))
    {
      const add2t &lhs = to_add2t(simplified_side_1);
      const add2t &rhs = to_add2t(simplified_side_2);

      // Shared-base pointer cancellation reduces to a relation between the
      // two surviving offsets. Coerce both to a common type so the rebuilt
      // node is well-formed even when the offsets carry different concrete
      // bv widths — `arr + (int)c` vs `arr + (long)c` from a p++ chain.
      auto cancel = [&](expr2tc a, expr2tc b) -> expr2tc {
        if (!coerce_to_common_type(a, b))
          return expr2tc();
        expr2tc rel(std::make_shared<constructor>(a, b));
        return typecast_check_return(type, rel);
      };

      if (
        lhs.side_1 == rhs.side_1 && is_constant(lhs.side_2) &&
        is_constant(rhs.side_2))
      {
        expr2tc r = cancel(lhs.side_2, rhs.side_2);
        if (!is_nil_expr(r))
          return r;
      }
      if (
        lhs.side_2 == rhs.side_2 && is_constant(lhs.side_1) &&
        is_constant(rhs.side_1))
      {
        expr2tc r = cancel(lhs.side_1, rhs.side_1);
        if (!is_nil_expr(r))
          return r;
      }
      if (
        lhs.side_1 == rhs.side_2 && is_constant(lhs.side_2) &&
        is_constant(rhs.side_1))
      {
        expr2tc r = cancel(lhs.side_2, rhs.side_1);
        if (!is_nil_expr(r))
          return r;
      }
      if (
        lhs.side_2 == rhs.side_1 && is_constant(lhs.side_1) &&
        is_constant(rhs.side_2))
      {
        expr2tc r = cancel(lhs.side_1, rhs.side_2);
        if (!is_nil_expr(r))
          return r;
      }
    }

    return expr2tc();
  }

  expr2tc simpl_res;

  if (is_bv_type(simplified_side_1) || is_bv_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_int2t;

    std::function<BigInt &(expr2tc &)> get_value = [](expr2tc &c) -> BigInt & {
      return to_constant_int2t(c).value;
    };

    simpl_res = TFunctor<BigInt &>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else if (
    is_fixedbv_type(simplified_side_1) || is_fixedbv_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_fixedbv2t;

    std::function<fixedbvt &(expr2tc &)> get_value =
      [](expr2tc &c) -> fixedbvt & { return to_constant_fixedbv2t(c).value; };

    simpl_res = TFunctor<fixedbvt &>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else if (
    is_floatbv_type(simplified_side_1) || is_floatbv_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_floatbv2t;

    std::function<ieee_floatt &(expr2tc &)> get_value =
      [](expr2tc &c) -> ieee_floatt & {
      return to_constant_floatbv2t(c).value;
    };

    simpl_res = TFunctor<ieee_floatt &>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else if (
    is_pointer_type(simplified_side_1) || is_pointer_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      [&](const expr2tc &t) -> bool {
      if (is_pointer_type(t) && is_symbol2t(t))
      {
        symbol2t s = to_symbol2t(t);
        if (s.thename == "NULL")
          return true;
      }
      return false;
    };

    std::function<int(expr2tc &)> get_value = [](expr2tc &) -> int {
      return 0xbadbeef;
    };

    simpl_res = TFunctor<int>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else
    return expr2tc();

  return typecast_check_return(type, simpl_res);
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_floatbv_relations(
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2)
{
  if (!is_number_type(type))
    return expr2tc();

  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so the operands are already simplified.
  const expr2tc &simplified_side_1 = side_1;
  const expr2tc &simplified_side_2 = side_2;

  if (
    !is_constant_expr(simplified_side_1) &&
    !is_constant_expr(simplified_side_2) &&
    !(simplified_side_1 == simplified_side_2))
    return expr2tc();

  expr2tc simpl_res;

  if (is_floatbv_type(simplified_side_1) || is_floatbv_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_floatbv2t;

    std::function<ieee_floatt &(expr2tc &)> get_value =
      [](expr2tc &c) -> ieee_floatt & {
      return to_constant_floatbv2t(c).value;
    };

    simpl_res = TFunctor<ieee_floatt>::simplify(
      simplified_side_1, simplified_side_2, is_constant, get_value);
  }
  else
    assert(0);

  return typecast_check_return(type, simpl_res);
}

template <class constant_type>
struct IEEE_equalitytor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type(expr2tc &)> get_value)
  {
    // Two constants? Simplify to result of the comparison
    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) == get_value(c2));
    }

    if (op1 == op2)
    {
      // x == x is the same as saying !isnan(x)
      expr2tc is_nan = isnan2tc(op1);
      expr2tc is_not_nan = not2tc(is_nan);
      return try_simplification(is_not_nan);
    }

    return expr2tc();
  }
};

template <class constant_type>
struct Equalitytor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type(expr2tc &)> get_value)
  {
    // Is a vector operation ? Apply the op
    if (is_constant_vector2t(op1) || is_constant_vector2t(op2))
    {
      auto op = [](type2tc, expr2tc e1, expr2tc e2) {
        return equality2tc(e1, e2);
      };
      return distribute_vector_operation(op, op1, op2);
    }

    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) == get_value(c2));
    }

    return expr2tc();
  }
};

expr2tc equality2t::do_simplify() const
{
  // Self-comparison: x == x is always true (except for floats with NaN)
  if (side_1 == side_2 && !is_floatbv_type(side_1) && !is_floatbv_type(side_2))
    return gen_true_expr();

  // If we're dealing with floatbvs, call IEEE_equalitytor instead
  if (is_floatbv_type(side_1) || is_floatbv_type(side_2))
    return simplify_floatbv_relations<IEEE_equalitytor, equality2t>(
      type, side_1, side_2);

  // (x + c1) == c2 -> x == (c2 - c1). Requires homogeneous types across
  // the entire shape: the add, BOTH its operands, and c2 must share a
  // single arithmetic domain. Mixed widths (e.g. (x_u8 + c_u16) == c2_u16)
  // would silently rewrite into something that confuses modular semantics.
  if (
    is_add2t(side_1) && is_constant_int2t(side_2) &&
    side_1->type == side_2->type &&
    to_add2t(side_1).side_1->type == side_2->type &&
    to_add2t(side_1).side_2->type == side_2->type)
  {
    const add2t &add_expr = to_add2t(side_1);

    if (is_constant_int2t(add_expr.side_2))
    {
      const BigInt &c1 = to_constant_int2t(add_expr.side_2).value;
      const BigInt &c2 = to_constant_int2t(side_2).value;
      BigInt diff = c2 - c1;

      if (fits_in_width(
            diff, side_2->type->get_width(), is_signedbv_type(side_2->type)))
      {
        expr2tc new_const = constant_int2tc(side_2->type, diff);
        return equality2tc(add_expr.side_1, new_const);
      }
    }

    if (is_constant_int2t(add_expr.side_1))
    {
      const BigInt &c1 = to_constant_int2t(add_expr.side_1).value;
      const BigInt &c2 = to_constant_int2t(side_2).value;
      BigInt diff = c2 - c1;

      if (fits_in_width(
            diff, side_2->type->get_width(), is_signedbv_type(side_2->type)))
      {
        expr2tc new_const = constant_int2tc(side_2->type, diff);
        return equality2tc(add_expr.side_2, new_const);
      }
    }
  }

  // (x - c1) == c2 -> x == (c2 + c1). Same homogeneity requirement.
  if (
    is_sub2t(side_1) && is_constant_int2t(side_2) &&
    side_1->type == side_2->type &&
    to_sub2t(side_1).side_1->type == side_2->type &&
    to_sub2t(side_1).side_2->type == side_2->type)
  {
    const sub2t &sub_expr = to_sub2t(side_1);

    if (is_constant_int2t(sub_expr.side_2))
    {
      const BigInt &c1 = to_constant_int2t(sub_expr.side_2).value;
      const BigInt &c2 = to_constant_int2t(side_2).value;
      BigInt sum = c2 + c1;

      if (fits_in_width(
            sum, side_2->type->get_width(), is_signedbv_type(side_2->type)))
      {
        expr2tc new_const = constant_int2tc(side_2->type, sum);
        return equality2tc(sub_expr.side_1, new_const);
      }
    }
  }

  // (x * c) == 0 -> x == 0 when c is odd. Restricted to odd constants
  // because modular bv multiplication is injective only for invertibles
  // mod 2^width — i.e. constants coprime to 2^width, which for power-of-two
  // moduli is exactly the odd constants. With c=2 in 8-bit unsigned, for
  // example, x=128 also satisfies x*2 == 0 (mod 256), so dropping the
  // multiplication would silently strengthen the predicate.
  //
  // Defensive type guard: equality2tc requires both sides to share a type.
  // A well-formed mul2t already has homogeneous operands, but if a future
  // construction path produces a mixed-width mul, the rewritten equality
  // would mix widths too. Skip the rewrite unless mul.side_*->type matches
  // side_2->type.
  if (is_mul2t(side_1) && is_constant_int2t(side_2))
  {
    const mul2t &mul_expr = to_mul2t(side_1);
    const BigInt &c2 = to_constant_int2t(side_2).value;

    if (c2 == 0)
    {
      if (
        is_constant_int2t(mul_expr.side_2) &&
        mul_expr.side_1->type == side_2->type)
      {
        const BigInt &c1 = to_constant_int2t(mul_expr.side_2).value;
        if (c1.is_odd())
          return equality2tc(mul_expr.side_1, side_2);
      }

      if (
        is_constant_int2t(mul_expr.side_1) &&
        mul_expr.side_2->type == side_2->type)
      {
        const BigInt &c1 = to_constant_int2t(mul_expr.side_1).value;
        if (c1.is_odd())
          return equality2tc(mul_expr.side_2, side_2);
      }
    }
  }

  // d + c == d + e -> c == e (cancel common addend). When the surviving
  // operands have differing concrete types (pointer-arith chains often mix
  // `(int)c` with `(long)e`), coerce both to a common type so the rebuilt
  // equality is well-formed.
  auto cancel_eq = [](expr2tc a, expr2tc b) -> expr2tc {
    if (!coerce_to_common_type(a, b))
      return expr2tc();
    return equality2tc(a, b);
  };

  if (is_add2t(side_1) && is_add2t(side_2))
  {
    const add2t &add1 = to_add2t(side_1);
    const add2t &add2 = to_add2t(side_2);
    expr2tc r;
    if (add1.side_1 == add2.side_1)
      if (!is_nil_expr(r = cancel_eq(add1.side_2, add2.side_2)))
        return r;
    if (add1.side_1 == add2.side_2)
      if (!is_nil_expr(r = cancel_eq(add1.side_2, add2.side_1)))
        return r;
    if (add1.side_2 == add2.side_1)
      if (!is_nil_expr(r = cancel_eq(add1.side_1, add2.side_2)))
        return r;
    if (add1.side_2 == add2.side_2)
      if (!is_nil_expr(r = cancel_eq(add1.side_1, add2.side_1)))
        return r;
  }

  // (d - c) == (d - e) -> c == e (cancel common minuend)
  if (is_sub2t(side_1) && is_sub2t(side_2))
  {
    const sub2t &sub1 = to_sub2t(side_1);
    const sub2t &sub2 = to_sub2t(side_2);
    if (sub1.side_1 == sub2.side_1)
    {
      expr2tc r = cancel_eq(sub1.side_2, sub2.side_2);
      if (!is_nil_expr(r))
        return r;
    }
  }

  // (-x) == (-y) -> x == y
  if (is_neg2t(side_1) && is_neg2t(side_2))
  {
    expr2tc r = cancel_eq(to_neg2t(side_1).value, to_neg2t(side_2).value);
    if (!is_nil_expr(r))
      return r;
  }

  // (~x) == (~y) -> x == y
  if (is_bitnot2t(side_1) && is_bitnot2t(side_2))
  {
    expr2tc r = cancel_eq(to_bitnot2t(side_1).value, to_bitnot2t(side_2).value);
    if (!is_nil_expr(r))
      return r;
  }

  return simplify_relations<Equalitytor, equality2t>(type, side_1, side_2);
}

template <class constant_type>
struct IEEE_notequalitytor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type(expr2tc &)> get_value)
  {
    // Two constants? Simplify to result of the comparison
    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) != get_value(c2));
    }

    if (op1 == op2)
    {
      // x != x is the same as saying isnan(x)
      expr2tc is_nan = isnan2tc(op1);
      return try_simplification(is_nan);
    }

    return expr2tc();
  }
};

template <class constant_type>
struct Notequaltor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type(expr2tc &)> get_value)
  {
    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) != get_value(c2));
    }

    return expr2tc();
  }
};

expr2tc notequal2t::do_simplify() const
{
  // Self-comparison: x != x is always false (except for floats with NaN)
  if (side_1 == side_2 && !is_floatbv_type(side_1) && !is_floatbv_type(side_2))
    return gen_false_expr();

  // If we're dealing with floatbvs, call IEEE_notequalitytor instead
  if (is_floatbv_type(side_1) || is_floatbv_type(side_2))
    return simplify_floatbv_relations<IEEE_notequalitytor, equality2t>(
      type, side_1, side_2);

  // The shape-canonicalizations below mirror equality2t::do_simplify. They are
  // the same rewrites: != x y holds iff == x y doesn't, so any rewrite that
  // preserves equality also preserves inequality.

  // (x + c1) != c2 -> x != (c2 - c1), and the (c1 + x) != c2 mirror.
  // Same homogeneity requirement as the equality case: the add, BOTH its
  // operands, and c2 must all share a single arithmetic domain.
  if (
    is_add2t(side_1) && is_constant_int2t(side_2) &&
    side_1->type == side_2->type &&
    to_add2t(side_1).side_1->type == side_2->type &&
    to_add2t(side_1).side_2->type == side_2->type)
  {
    const add2t &add_expr = to_add2t(side_1);
    const BigInt &c2 = to_constant_int2t(side_2).value;

    if (is_constant_int2t(add_expr.side_2))
    {
      const BigInt &c1 = to_constant_int2t(add_expr.side_2).value;
      BigInt diff = c2 - c1;

      if (fits_in_width(
            diff, side_2->type->get_width(), is_signedbv_type(side_2->type)))
      {
        expr2tc new_const = constant_int2tc(side_2->type, diff);
        return notequal2tc(add_expr.side_1, new_const);
      }
    }

    if (is_constant_int2t(add_expr.side_1))
    {
      const BigInt &c1 = to_constant_int2t(add_expr.side_1).value;
      BigInt diff = c2 - c1;

      if (fits_in_width(
            diff, side_2->type->get_width(), is_signedbv_type(side_2->type)))
      {
        expr2tc new_const = constant_int2tc(side_2->type, diff);
        return notequal2tc(add_expr.side_2, new_const);
      }
    }
  }

  // (x - c1) != c2 -> x != (c2 + c1). Same homogeneity requirement.
  if (
    is_sub2t(side_1) && is_constant_int2t(side_2) &&
    side_1->type == side_2->type &&
    to_sub2t(side_1).side_1->type == side_2->type &&
    to_sub2t(side_1).side_2->type == side_2->type)
  {
    const sub2t &sub_expr = to_sub2t(side_1);

    if (is_constant_int2t(sub_expr.side_2))
    {
      const BigInt &c1 = to_constant_int2t(sub_expr.side_2).value;
      const BigInt &c2 = to_constant_int2t(side_2).value;
      BigInt sum = c2 + c1;

      if (fits_in_width(
            sum, side_2->type->get_width(), is_signedbv_type(side_2->type)))
      {
        expr2tc new_const = constant_int2tc(side_2->type, sum);
        return notequal2tc(sub_expr.side_1, new_const);
      }
    }
  }

  // d + c != d + e -> c != e (cancel common addend). Coerce surviving
  // operands to a common type when their concrete types differ.
  auto cancel_neq = [](expr2tc a, expr2tc b) -> expr2tc {
    if (!coerce_to_common_type(a, b))
      return expr2tc();
    return notequal2tc(a, b);
  };

  if (is_add2t(side_1) && is_add2t(side_2))
  {
    const add2t &add1 = to_add2t(side_1);
    const add2t &add2 = to_add2t(side_2);
    expr2tc r;
    if (add1.side_1 == add2.side_1)
      if (!is_nil_expr(r = cancel_neq(add1.side_2, add2.side_2)))
        return r;
    if (add1.side_1 == add2.side_2)
      if (!is_nil_expr(r = cancel_neq(add1.side_2, add2.side_1)))
        return r;
    if (add1.side_2 == add2.side_1)
      if (!is_nil_expr(r = cancel_neq(add1.side_1, add2.side_2)))
        return r;
    if (add1.side_2 == add2.side_2)
      if (!is_nil_expr(r = cancel_neq(add1.side_1, add2.side_1)))
        return r;
  }

  // (d - c) != (d - e) -> c != e (cancel common minuend)
  if (is_sub2t(side_1) && is_sub2t(side_2))
  {
    const sub2t &sub1 = to_sub2t(side_1);
    const sub2t &sub2 = to_sub2t(side_2);
    if (sub1.side_1 == sub2.side_1)
    {
      expr2tc r = cancel_neq(sub1.side_2, sub2.side_2);
      if (!is_nil_expr(r))
        return r;
    }
  }

  // (-x) != (-y) -> x != y
  if (is_neg2t(side_1) && is_neg2t(side_2))
  {
    expr2tc r = cancel_neq(to_neg2t(side_1).value, to_neg2t(side_2).value);
    if (!is_nil_expr(r))
      return r;
  }

  // (~x) != (~y) -> x != y
  if (is_bitnot2t(side_1) && is_bitnot2t(side_2))
  {
    expr2tc r =
      cancel_neq(to_bitnot2t(side_1).value, to_bitnot2t(side_2).value);
    if (!is_nil_expr(r))
      return r;
  }

  return simplify_relations<Notequaltor, notequal2t>(type, side_1, side_2);
}

template <class constant_type>
struct Lessthantor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type(expr2tc &)> get_value)
  {
    // op1 < zero and op2 is unsigned: always true
    if (is_constant(op1))
    {
      expr2tc c1 = op1;
      if ((get_value(c1) < 0) && is_unsignedbv_type(op2))
        return gen_true_expr();
    }

    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) < get_value(c2));
    }

    return expr2tc();
  }
};

/// Type-extreme detector. For an integer constant @p c against bv type @p t,
/// reports whether the constant equals the type's representable max or min.
/// Used to short-circuit ordered comparisons against the bounds of the type.
static bool is_type_max(const BigInt &c, const type2tc &t)
{
  if (!is_bv_type(t))
    return false;
  unsigned w = t->get_width();
  if (is_signedbv_type(t))
    return c == BigInt::power2(w - 1) - 1;
  return c == BigInt::power2(w) - 1;
}

static bool is_type_min(const BigInt &c, const type2tc &t)
{
  if (!is_bv_type(t))
    return false;
  if (is_signedbv_type(t))
    return c == -BigInt::power2(t->get_width() - 1);
  return c.is_zero();
}

expr2tc lessthan2t::do_simplify() const
{
  // Self-comparison: x < x is always false (except for floats with NaN)
  if (side_1 == side_2 && !is_floatbv_type(side_1) && !is_floatbv_type(side_2))
    return gen_false_expr();

  // Type-extreme bounds. x < TYPE_MIN is always false; nothing in the type's
  // range is less than the minimum representable value. Subsumes the existing
  // "unsigned < 0" rule. Require the constant to share the variable side's
  // type so we don't fold based on a TYPE_MIN of the wrong domain.
  if (
    is_constant_int2t(side_2) && side_2->type == side_1->type &&
    is_type_min(to_constant_int2t(side_2).value, side_1->type))
    return gen_false_expr();

  // TYPE_MAX < x is always false; nothing in the type's range exceeds the
  // maximum representable value.
  if (
    is_constant_int2t(side_1) && side_1->type == side_2->type &&
    is_type_max(to_constant_int2t(side_1).value, side_2->type))
    return gen_false_expr();

  return simplify_relations<Lessthantor, lessthan2t>(type, side_1, side_2);
}

template <class constant_type>
struct Greaterthantor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type(expr2tc &)> get_value)
  {
    // op2 < zero and op1 is unsigned: always true
    if (is_constant(op2))
    {
      expr2tc c2 = op2;
      if ((get_value(c2) < 0) && is_unsignedbv_type(op1))
        return gen_true_expr();
    }

    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) > get_value(c2));
    }

    return expr2tc();
  }
};

expr2tc greaterthan2t::do_simplify() const
{
  // Self-comparison: x > x is always false (except for floats with NaN)
  if (side_1 == side_2 && !is_floatbv_type(side_1) && !is_floatbv_type(side_2))
    return gen_false_expr();

  // x > TYPE_MAX is always false; nothing exceeds the max representable.
  // Require the constant to share the variable side's type.
  if (
    is_constant_int2t(side_2) && side_2->type == side_1->type &&
    is_type_max(to_constant_int2t(side_2).value, side_1->type))
    return gen_false_expr();

  // TYPE_MIN > x is always false; nothing is below the min representable.
  if (
    is_constant_int2t(side_1) && side_1->type == side_2->type &&
    is_type_min(to_constant_int2t(side_1).value, side_2->type))
    return gen_false_expr();

  return simplify_relations<Greaterthantor, greaterthan2t>(
    type, side_1, side_2);
}

template <class constant_type>
struct Lessthanequaltor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type(expr2tc &)> get_value)
  {
    // op1 <= zero and op2 is unsigned: always true
    if (is_constant(op1))
    {
      expr2tc c1 = op1;
      if ((get_value(c1) <= 0) && is_unsignedbv_type(op2))
        return gen_true_expr();
    }

    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) <= get_value(c2));
    }

    return expr2tc();
  }
};

expr2tc lessthanequal2t::do_simplify() const
{
  // Self-comparison: x <= x is always true (except for floats with NaN)
  if (side_1 == side_2 && !is_floatbv_type(side_1) && !is_floatbv_type(side_2))
    return gen_true_expr();

  // x <= TYPE_MAX is always true; the max representable bounds the type.
  // Require the constant to share the variable side's type.
  if (
    is_constant_int2t(side_2) && side_2->type == side_1->type &&
    is_type_max(to_constant_int2t(side_2).value, side_1->type))
    return gen_true_expr();

  // TYPE_MIN <= x is always true; the min representable bounds the type.
  if (
    is_constant_int2t(side_1) && side_1->type == side_2->type &&
    is_type_min(to_constant_int2t(side_1).value, side_2->type))
    return gen_true_expr();

  return simplify_relations<Lessthanequaltor, lessthanequal2t>(
    type, side_1, side_2);
}

template <class constant_type>
struct Greaterthanequaltor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type(expr2tc &)> get_value)
  {
    // op2 <= zero and op1 is unsigned: always true
    if (is_constant(op2))
    {
      expr2tc c2 = op2;
      if ((get_value(c2) <= 0) && is_unsignedbv_type(op1))
        return gen_true_expr();
    }

    if (is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) >= get_value(c2));
    }

    return expr2tc();
  }
};

expr2tc greaterthanequal2t::do_simplify() const
{
  // Self-comparison: x >= x is always true (except for floats with NaN)
  if (side_1 == side_2 && !is_floatbv_type(side_1) && !is_floatbv_type(side_2))
    return gen_true_expr();

  // x >= TYPE_MIN is always true; the min representable bounds the type.
  // Subsumes the existing "unsigned >= 0" rule. Require the constant to
  // share the variable side's type so we don't fold based on the wrong
  // domain's TYPE_MIN.
  if (
    is_constant_int2t(side_2) && side_2->type == side_1->type &&
    is_type_min(to_constant_int2t(side_2).value, side_1->type))
    return gen_true_expr();

  // TYPE_MAX >= x is always true; the max representable bounds the type.
  if (
    is_constant_int2t(side_1) && side_1->type == side_2->type &&
    is_type_max(to_constant_int2t(side_1).value, side_2->type))
    return gen_true_expr();

  return simplify_relations<Greaterthanequaltor, greaterthanequal2t>(
    type, side_1, side_2);
}

expr2tc cmp_three_way2t::do_simplify() const
{
  auto cmp_int = [](const BigInt &a, const BigInt &b) {
    return (a < b) ? -1 : (a == b ? 0 : 1);
  };

  // Self-comparison on a non-float type is always equivalent. (Floats need
  // partial_ordering::unordered for NaN; leave that case alone.)
  if (side_1 == side_2 && !is_floatbv_type(side_1) && !is_floatbv_type(side_2))
    return make_cmp_value(type, 0);

  // Both sides constant integers: fold to the appropriate value.
  if (is_constant_int2t(side_1) && is_constant_int2t(side_2))
  {
    const BigInt &a = to_constant_int2t(side_1).value;
    const BigInt &b = to_constant_int2t(side_2).value;
    return make_cmp_value(type, cmp_int(a, b));
  }

  // Same-base pointer ordering. (&arr+c1) <=> (&arr+c2) folds to T{cmp(c1,c2)}
  // — this is well-defined per [expr.spaceship]/4 because both operands point
  // into the same array object (or one past the end). Mirrors the shared-base
  // cancellation in simplify_relations: same four permutations of base/offset.
  // Pointers into different objects are unspecified and intentionally left
  // alone.
  if (
    is_add2t(side_1) && is_add2t(side_2) && is_pointer_type(side_1) &&
    is_pointer_type(side_2))
  {
    const add2t &lhs = to_add2t(side_1);
    const add2t &rhs = to_add2t(side_2);

    auto fold = [&](const expr2tc &a, const expr2tc &b) -> expr2tc {
      if (!is_constant_int2t(a) || !is_constant_int2t(b))
        return expr2tc();
      return make_cmp_value(
        type, cmp_int(to_constant_int2t(a).value, to_constant_int2t(b).value));
    };

    if (lhs.side_1 == rhs.side_1)
      if (expr2tc r = fold(lhs.side_2, rhs.side_2); !is_nil_expr(r))
        return r;
    if (lhs.side_2 == rhs.side_2)
      if (expr2tc r = fold(lhs.side_1, rhs.side_1); !is_nil_expr(r))
        return r;
    if (lhs.side_1 == rhs.side_2)
      if (expr2tc r = fold(lhs.side_2, rhs.side_1); !is_nil_expr(r))
        return r;
    if (lhs.side_2 == rhs.side_1)
      if (expr2tc r = fold(lhs.side_1, rhs.side_2); !is_nil_expr(r))
        return r;
  }

  return expr2tc();
}

// Check if two conditions are equivalent (accounting for casts)
static bool conditions_equivalent(const expr2tc &a, const expr2tc &b)
{
  if (a == b)
    return true;

  // Strip typecast from a
  expr2tc a_stripped = a;
  while (is_typecast2t(a_stripped))
    a_stripped = to_typecast2t(a_stripped).from;

  // Strip typecast from b
  expr2tc b_stripped = b;
  while (is_typecast2t(b_stripped))
    b_stripped = to_typecast2t(b_stripped).from;

  // Also handle (x != 0) <-> x pattern
  if (is_notequal2t(a_stripped))
  {
    const notequal2t &ne = to_notequal2t(a_stripped);
    if (is_constant_int2t(ne.side_2) && to_constant_int2t(ne.side_2).value == 0)
      a_stripped = ne.side_1;
  }

  if (is_notequal2t(b_stripped))
  {
    const notequal2t &ne = to_notequal2t(b_stripped);
    if (is_constant_int2t(ne.side_2) && to_constant_int2t(ne.side_2).value == 0)
      b_stripped = ne.side_1;
  }

  return a_stripped == b_stripped;
}

expr2tc if2t::do_simplify() const
{
  // if(c, x, x) -> x. Sound for any type, including bool — moved here from
  // below the bool-arm so it fires for bool-typed selects whose branches
  // happen to be the same symbolic value (the bool arm previously short-
  // circuited to nil before this rule could be reached).
  if (true_value == false_value)
    return typecast_check_return(type, true_value);

  // Constant-condition fold. expr2t::simplify already simplified `cond`.
  if (is_constant_expr(cond))
  {
    if (is_true(cond))
      return typecast_check_return(type, true_value);
    if (is_false(cond))
      return typecast_check_return(type, false_value);
  }

  // LLVM-style select-around-abs: a sign test that picks between x and -x is
  // exactly abs(x). Restricted to signed bv: floating-point abs has IEEE
  // 754-specific behavior for signed zero and NaN that this rewrite would
  // not preserve (e.g. (-0.0 >= 0.0) is true, but abs(-0.0) is +0.0).
  if (is_signedbv_type(type))
  {
    auto match_zero = [](const expr2tc &e) {
      return is_constant_int2t(e) && to_constant_int2t(e).value.is_zero();
    };
    // Decode @p c as a sign test "x is non-negative" (out_positive=true) or
    // "x is non-positive" (out_positive=false), returning x. Recognizes the
    // four relations and both operand orders. Zero must match the operand's
    // type. Returns nil if @p c isn't a sign test.
    auto decode_sign_test =
      [&](const expr2tc &c, expr2tc &x, bool &out_positive) -> bool {
      auto handle = [&](
                      const expr2tc &s1,
                      const expr2tc &s2,
                      bool x_is_left,
                      bool positive_when_x_left) -> bool {
        const expr2tc &candidate_x = x_is_left ? s1 : s2;
        const expr2tc &candidate_z = x_is_left ? s2 : s1;
        if (!match_zero(candidate_z) || candidate_x->type != type)
          return false;
        x = candidate_x;
        out_positive = x_is_left ? positive_when_x_left : !positive_when_x_left;
        return true;
      };
      // (x >= 0) and (0 >= x) ≡ (x <= 0)
      if (is_greaterthanequal2t(c))
      {
        const greaterthanequal2t &r = to_greaterthanequal2t(c);
        if (match_zero(r.side_2))
          return handle(r.side_1, r.side_2, true, true);
        if (match_zero(r.side_1))
          return handle(r.side_1, r.side_2, false, true);
        return false;
      }
      // (x > 0) and (0 > x) ≡ (x < 0)
      if (is_greaterthan2t(c))
      {
        const greaterthan2t &r = to_greaterthan2t(c);
        if (match_zero(r.side_2))
          return handle(r.side_1, r.side_2, true, true);
        if (match_zero(r.side_1))
          return handle(r.side_1, r.side_2, false, true);
        return false;
      }
      // (x <= 0) and (0 <= x) ≡ (x >= 0)
      if (is_lessthanequal2t(c))
      {
        const lessthanequal2t &r = to_lessthanequal2t(c);
        if (match_zero(r.side_2))
          return handle(r.side_1, r.side_2, true, false);
        if (match_zero(r.side_1))
          return handle(r.side_1, r.side_2, false, false);
        return false;
      }
      // (x < 0) and (0 < x) ≡ (x > 0)
      if (is_lessthan2t(c))
      {
        const lessthan2t &r = to_lessthan2t(c);
        if (match_zero(r.side_2))
          return handle(r.side_1, r.side_2, true, false);
        if (match_zero(r.side_1))
          return handle(r.side_1, r.side_2, false, false);
        return false;
      }
      return false;
    };

    expr2tc x;
    bool positive_test;
    if (decode_sign_test(cond, x, positive_test))
    {
      // positive_test == true means cond is "x non-negative" — true_value
      // should be x, false_value should be -x.
      // positive_test == false means cond is "x non-positive" — true_value
      // should be -x, false_value should be x.
      const expr2tc &expected_t = positive_test ? x : neg2tc(type, x);
      const expr2tc &expected_f = positive_test ? neg2tc(type, x) : x;
      if (true_value == expected_t && false_value == expected_f)
        return abs2tc(type, x);
    }
  }

  if (is_bool_type(type))
  {
    // We can only do these simplification if the expecting results is boolean
    // A bug was introduced in 2835092f that applied these simplifications to
    // integers, which resulted in expressions like:
    //
    // c:@t2&1#4 == (guard?0!0&0#3 ? 2 : 0)
    //
    // to be simplified to:
    //
    // c:@t2&1#4 == (unsigned int)guard?0!0&0#3

    expr2tc simp;
    if (is_true(true_value) && is_false(false_value))
    {
      // a?1:0 <-> a
      simp = cond;
    }
    else if (is_false(true_value) && is_true(false_value))
    {
      // a?0:1 <-> !a
      simp = not2tc(cond);
    }
    else if (is_false(false_value))
    {
      // a?b:0 <-> a AND b
      simp = and2tc(cond, true_value);
    }
    else if (is_true(false_value))
    {
      // a?b:1 <-> !a OR b
      simp = or2tc(not2tc(cond), true_value);
    }
    else if (is_true(true_value))
    {
      // a?1:b <-> a||(!a && b) <-> a OR b
      simp = or2tc(cond, false_value);
    }
    else if (is_false(true_value))
    {
      // a?0:b <-> !a && b
      simp = and2tc(not2tc(cond), false_value);
    }
    else
      simp = expr2tc(); // none matched; fall through to generic patterns below

    if (!is_nil_expr(simp))
    {
      ::simplify(simp);
      return simp;
    }
  }

  if (
    is_constant_number(true_value) && is_true(true_value) &&
    (gen_one(true_value->type) == true_value) && is_false(false_value))
    return typecast_check_return(type, cond);

  if (
    is_constant_number(false_value) && is_false(true_value) &&
    (gen_one(false_value->type) == false_value) && is_true(false_value))
    return typecast_check_return(type, not2tc(cond));

  // Nested-if collapse on the false branch. The outer `c` is true on the way
  // into the inner if (we're in its false_value position when c is false), so
  // the inner if can be reduced when its condition is c or !c.

  // The if-collapse folds below all reuse `type` for the rebuilt node
  // while taking operands from the inner if. The if2t constructor asserts
  // type->type_id == trueval->type->type_id, so when the inner branches
  // carry a different type (e.g. mixed-width operands after upstream
  // simplification) we'd crash. Gate each fold on the inner operand's
  // type matching the outer.

  // (c ? x : (c ? y : z)) == (c ? x : z)
  if (is_if2t(false_value))
  {
    const if2t &inner_if = to_if2t(false_value);
    if (inner_if.cond == cond && inner_if.false_value->type == type)
      return if2tc(type, cond, true_value, inner_if.false_value);
  }

  // (c ? x : (!c ? y : z)) == (c ? x : y)
  if (is_if2t(false_value))
  {
    const if2t &inner_if = to_if2t(false_value);
    if (is_not2t(inner_if.cond))
    {
      const not2t &inner_neg = to_not2t(inner_if.cond);
      if (
        conditions_equivalent(inner_neg.value, cond) &&
        inner_if.true_value->type == type)
        return if2tc(type, cond, true_value, inner_if.true_value);
    }
  }

  // (!c ? x : (c ? y : z)) == (c ? y : x)
  if (is_not2t(cond))
  {
    const not2t &neg = to_not2t(cond);
    if (is_if2t(false_value))
    {
      const if2t &inner_if = to_if2t(false_value);
      if (
        conditions_equivalent(inner_if.cond, neg.value) &&
        inner_if.true_value->type == type)
        return if2tc(type, inner_if.cond, inner_if.true_value, true_value);
    }
  }

  // Symmetric collapses on the true branch. When the outer c is true we're
  // in the inner if's true_value position, so the inner if's condition is
  // also constrained.

  // (c ? (c ? a : b) : x) == (c ? a : x)
  if (is_if2t(true_value))
  {
    const if2t &inner_if = to_if2t(true_value);
    if (inner_if.cond == cond && inner_if.true_value->type == type)
      return if2tc(type, cond, inner_if.true_value, false_value);
  }

  // (c ? (!c ? a : b) : x) == (c ? b : x)
  if (is_if2t(true_value))
  {
    const if2t &inner_if = to_if2t(true_value);
    if (is_not2t(inner_if.cond))
    {
      const not2t &inner_neg = to_not2t(inner_if.cond);
      if (
        conditions_equivalent(inner_neg.value, cond) &&
        inner_if.false_value->type == type)
        return if2tc(type, cond, inner_if.false_value, false_value);
    }
  }

  // (!c ? (c ? a : b) : x) == (!c ? b : x)
  if (is_not2t(cond))
  {
    const not2t &neg = to_not2t(cond);
    if (is_if2t(true_value))
    {
      const if2t &inner_if = to_if2t(true_value);
      if (
        conditions_equivalent(inner_if.cond, neg.value) &&
        inner_if.false_value->type == type)
        return if2tc(type, cond, inner_if.false_value, false_value);
    }
  }

  return expr2tc();
}

expr2tc overflow_cast2t::do_simplify() const
{
  // SMT lowering (smt_overflow.cpp:overflow_cast) defines this as
  // `operand < 0 || operand > 2^bits - 1` — an unsigned narrowing check.
  // Bool operand: always fits (true=1, false=0), so the cast never overflows.
  if (is_bool_type(operand->type))
    return gen_false_expr();

  // Unsigned source whose width fits in the destination: the value is
  // already in [0, 2^src_width - 1] ⊆ [0, 2^bits - 1], so no overflow.
  if (is_unsignedbv_type(operand->type) && bits >= operand->type->get_width())
    return gen_false_expr();

  // Constant operand: directly compute the bound check.
  if (is_constant_int2t(operand))
  {
    const BigInt &v = to_constant_int2t(operand).value;
    if (v.is_negative())
      return gen_true_expr();
    return v > BigInt::power2(bits) - 1 ? gen_true_expr() : gen_false_expr();
  }

  return expr2tc();
}

expr2tc overflow2t::do_simplify() const
{
  // expr2t::simplify gates `overflow_id` and never reaches this method via
  // the operands-first walker (see expr_simplifier.cpp around line 34) — the
  // inner arith op must keep its un-simplified shape so the SMT layer can
  // see whether the operation itself overflows. This do_simplify is therefore
  // only reachable through direct try_simplification calls. No callers do
  // that today; leave as a stub rather than add code that would not run.
  return expr2tc();
}

static expr2tc obj_equals_addr_of(const expr2tc &a, const expr2tc &b);

static expr2tc handle_symmetric_cases(const expr2tc &a, const expr2tc &b)
{
  // Array element vs base symbol
  if (is_index2t(a) && is_symbol2t(b))
    return obj_equals_addr_of(to_index2t(a).source_value, b);
  if (is_symbol2t(a) && is_index2t(b))
    return obj_equals_addr_of(to_index2t(b).source_value, a);

  // Struct member vs base symbol
  if (is_member2t(a) && is_symbol2t(b))
    return obj_equals_addr_of(to_member2t(a).source_value, b);
  if (is_symbol2t(a) && is_member2t(b))
    return obj_equals_addr_of(to_member2t(b).source_value, a);

  // Array element vs struct member
  if (is_index2t(a) && is_member2t(b))
    return obj_equals_addr_of(
      to_index2t(a).source_value, to_member2t(b).source_value);
  if (is_member2t(a) && is_index2t(b))
    return obj_equals_addr_of(
      to_member2t(a).source_value, to_index2t(b).source_value);

  return expr2tc(); // no match
}

static expr2tc obj_equals_addr_of(const expr2tc &a, const expr2tc &b)
{
  if (is_symbol2t(a) && is_symbol2t(b))
  {
    if (a == b)
      return gen_true_expr();
    else
      // In symbolic execution, different symbols could potentially
      // have the same value, so let the SMT solver decide
      return expr2tc();
  }
  else if (is_index2t(a) && is_index2t(b))
  {
    // For array elements, check if they belong to the same base array
    // In ESBMC's semantics, different elements of the same array
    // are considered part of the same object
    return obj_equals_addr_of(
      to_index2t(a).source_value, to_index2t(b).source_value);
  }
  else if (is_member2t(a) && is_member2t(b))
  {
    // For struct members, check if they belong to the same base struct
    // In ESBMC's semantics, different members of the same struct
    // are considered part of the same object
    return obj_equals_addr_of(
      to_member2t(a).source_value, to_member2t(b).source_value);
  }
  else if (is_constant_string2t(a) && is_constant_string2t(b))
  {
    bool val = (to_constant_string2t(a).value == to_constant_string2t(b).value);
    if (val)
      return gen_true_expr();

    return gen_false_expr();
  }

  expr2tc res = handle_symmetric_cases(a, b);
  if (!is_nil_expr(res))
    return res;

  // We can't determine if they are the same object
  return expr2tc();
}

static bool is_null_pointer(const expr2tc &expr)
{
  // Check for explicit NULL symbol
  if (is_symbol2t(expr) && to_symbol2t(expr).get_symbol_name() == "NULL")
    return true;

  return false;
}

expr2tc same_object2t::do_simplify() const
{
  expr2tc op1 = side_1;
  expr2tc op2 = side_2;

  // Look through typecast expressions to find the actual operands.
  // Defensive even after operands-first simplify: typecast2t::do_simplify
  // only strips when types match exactly; pointer-to-pointer different-
  // subtype casts may still be present here.
  while (is_typecast2t(op1))
    op1 = to_typecast2t(op1).from;
  while (is_typecast2t(op2))
    op2 = to_typecast2t(op2).from;

  // Handle NULL pointer comparisons first
  bool op1_is_null = is_null_pointer(op1);
  bool op2_is_null = is_null_pointer(op2);

  if (op1_is_null && op2_is_null)
    return gen_true_expr(); // Both NULL

  // Exactly one side is NULL: can the other side equal NULL? &x is the
  // address of a real object, never NULL, so they're definitely different
  // objects. Symbols (pointer variables) might hold NULL — leave to SMT.
  //
  // Special case: address_of(dereference(p)) is semantically equivalent to
  // p (a &*p round-trip), and p may be NULL. expr2t::simplify deliberately
  // doesn't simplify operands of address_of, so this shape can reach us
  // unchanged. Fold to false only when the address_of target isn't a
  // dereference of a pointer that could itself be NULL.
  auto address_of_is_provably_non_null = [](const expr2tc &e) -> bool {
    if (!is_address_of2t(e))
      return false;
    const expr2tc &target = to_address_of2t(e).ptr_obj;
    // &*p reduces to p, which may be NULL — refuse the fold.
    if (is_dereference2t(target))
      return false;
    return true;
  };
  if (op1_is_null && address_of_is_provably_non_null(op2))
    return gen_false_expr();
  if (op2_is_null && address_of_is_provably_non_null(op1))
    return gen_false_expr();

  // Handle address-of expressions
  if (is_address_of2t(op1) && is_address_of2t(op2))
    return obj_equals_addr_of(
      to_address_of2t(op1).ptr_obj, to_address_of2t(op2).ptr_obj);

  // Handle direct pointer comparisons (symbols that represent pointers)
  if (is_symbol2t(op1) && is_symbol2t(op2))
  {
    if (op1 == op2)
      return gen_true_expr();
    else
      // In symbolic execution, different symbols could potentially
      // have the same value, so let the SMT solver decide
      return expr2tc();
  }

  // If we can't simplify, return empty expression
  return expr2tc();
}

expr2tc concat2t::do_simplify() const
{
  // Two-constant fold: concat(c1, c2) places c1 in the high bits, c2 in the
  // low bits, so the combined value is c1 * 2^width(c2) + c2. Operands-first
  // ordering means a chain like concat(c1, concat(c2, c3)) folds bottom-up
  // — the inner concat folds first, then the outer two-constant fold fires.
  if (is_constant_int2t(side_1) && is_constant_int2t(side_2))
  {
    const BigInt &value1 = to_constant_int2t(side_1).value;
    const BigInt &value2 = to_constant_int2t(side_2).value;

    assert(!value1.is_negative());
    assert(!value2.is_negative());

    BigInt accuml = value1;
    accuml *= BigInt::power2(side_2->type->get_width());
    accuml += value2;

    return constant_int2tc(type, accuml);
  }

  // concat(0, x) -> zext(x). When the high bits are constant zero, the
  // result is x with its width zero-extended. ESBMC's typecast widens by
  // sign-extending signed sources, so emitting a plain typecast2tc is
  // unsound for signed x: e.g. signed 8-bit 0xff would become 0xffff
  // instead of the bit-preserving 0x00ff that concat semantics require.
  // Restrict to unsigned/bool sources, where typecast widening is already
  // a zero-extension. For signed, route through the unsigned-of-same-width
  // intermediate so the widening cast becomes zero-extension.
  if (is_constant_int2t(side_1) && to_constant_int2t(side_1).value.is_zero())
  {
    if (is_unsignedbv_type(side_2) || is_bool_type(side_2))
      return typecast2tc(type, side_2);
    if (is_signedbv_type(side_2))
    {
      const unsigned w = side_2->type->get_width();
      expr2tc unsigned_x = typecast2tc(unsignedbv_type2tc(w), side_2);
      return typecast2tc(type, unsigned_x);
    }
  }

  // Detect pattern: nested CONCATs of byte_extracts
  std::function<void(const expr2tc &, std::vector<expr2tc> &)> collect_leaves;
  collect_leaves = [&](const expr2tc &e, std::vector<expr2tc> &leaves) {
    if (is_concat2t(e))
    {
      const concat2t &c = to_concat2t(e);
      collect_leaves(c.side_1, leaves);
      collect_leaves(c.side_2, leaves);
    }
    else
    {
      leaves.push_back(e);
    }
  };

  // Collect concat leaves directly from side_1/side_2 instead of cloning
  // the whole subtree. The lambda dispatches on the operand's expr_id, so
  // it doesn't need a wrapping concat2tc to traverse our own children.
  std::vector<expr2tc> leaves;
  collect_leaves(side_1, leaves);
  collect_leaves(side_2, leaves);

  // Quick exit if not enough leaves or not all byte extracts
  if (leaves.size() < 2)
    return expr2tc();

  for (const auto &leaf : leaves)
  {
    if (!is_byte_extract2t(leaf) || leaf->type->get_width() != 8)
      return expr2tc();
  }

  // Verify all from same source with contiguous offsets
  const byte_extract2t &first_be = to_byte_extract2t(leaves[0]);
  expr2tc source = first_be.source_value;
  bool big_endian = first_be.big_endian;

  // Only optimize complete reconstructions where widths match
  if (
    leaves.size() * 8 != type->get_width() ||
    source->type->get_width() != type->get_width() || big_endian)
    return expr2tc();

  std::vector<BigInt> offsets;
  for (const auto &leaf : leaves)
  {
    const byte_extract2t &be = to_byte_extract2t(leaf);

    if (
      be.source_value != source || be.big_endian != big_endian ||
      !is_constant_int2t(be.source_offset))
      return expr2tc();

    offsets.push_back(to_constant_int2t(be.source_offset).value);
  }

  // Check for contiguous sequence: [n-1, n-2, ..., 1, 0] (little-endian only;
  // big-endian was rejected above pending careful endianness verification).
  for (size_t i = 0; i < offsets.size(); i++)
  {
    BigInt expected = BigInt(offsets.size() - 1 - i);
    if (offsets[i] != expected)
      return expr2tc();
  }

  if (source->type == type)
  {
    // Types match exactly - return as-is
    return source;
  }
  else
  {
    // Same width, different types - use bitcast for bit reinterpretation
    expr2tc result = bitcast2tc(type, source);
    return result;
  }
}

expr2tc extract2t::do_simplify() const
{
  assert(is_bv_type(type));

  // Full-width extract is a no-op when the types match. extract(x, w-1, 0)
  // selects every bit of x; the rewrite avoids emitting a redundant op.
  if (lower == 0 && from->type == type && upper + 1 == from->type->get_width())
    return from;

  // extract(concat(a, b), upper, lower) where the extract lies entirely in
  // one side of the concat collapses to that side. Symex emits this shape
  // routinely as the inverse of the byte-level concat reconstruction.
  if (is_concat2t(from))
  {
    const concat2t &c = to_concat2t(from);
    unsigned low_w = c.side_2->type->get_width();

    // Extract entirely in the high side: shift bit positions down by low_w.
    if (lower >= low_w)
      return extract2tc(type, c.side_1, upper - low_w, lower - low_w);

    // Extract entirely in the low side: bit positions are unchanged.
    if (upper < low_w)
      return extract2tc(type, c.side_2, upper, lower);
  }

  if (!is_constant_int2t(from))
    return expr2tc();

  // If you're hitting this, a non-bitfield related piece of code is now
  // generating extracts, and you have to consider performing extracts on
  // negative numbers.
  assert(is_unsignedbv_type(from->type));
  const BigInt &theint = to_constant_int2t(from).value;
  assert(!theint.is_negative());
  if (!theint.is_uint64())
    return expr2tc();

  // Take the value, mask and shift.
  uint64_t theval = theint.to_uint64();
  theval >>= lower;
  if (upper + 1 < 64)
    theval &= ~(~0ULL << (upper + 1));
  bool isneg = (theval >> upper) & 1;

  if (is_signedbv_type(type) && isneg)
  {
    // Type punning.
    union
    {
      int64_t sign;
      uint64_t nosign;
    } totallytmp;

    theval |= 0xFFFFFFFFFFFFFFFFULL << upper;
    totallytmp.nosign = theval;
    return constant_int2tc(type, BigInt(totallytmp.sign));
  }

  return constant_int2tc(type, BigInt(theval));
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_floatbv_1op(const type2tc &type, const expr2tc &value)
{
  if (!is_number_type(type))
    return expr2tc();

  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so `value` is already simplified.
  const expr2tc &to_simplify = value;
  if (!is_constant_expr(to_simplify))
    return expr2tc();

  expr2tc simpl_res;

  if (is_fixedbv_type(value))
  {
    std::function<constant_fixedbv2t &(expr2tc &)> to_constant =
      (constant_fixedbv2t & (*)(expr2tc &)) to_constant_fixedbv2t;

    simpl_res =
      TFunctor<constant_fixedbv2t>::simplify(to_simplify, to_constant);
  }
  else if (is_floatbv_type(value))
  {
    std::function<constant_floatbv2t &(expr2tc &)> to_constant =
      (constant_floatbv2t & (*)(expr2tc &)) to_constant_floatbv2t;

    simpl_res =
      TFunctor<constant_floatbv2t>::simplify(to_simplify, to_constant);
  }
  else
    return expr2tc();

  return typecast_check_return(type, simpl_res);
}

template <class constant_type>
struct Isnantor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;
    return constant_bool2tc(to_constant(c).value.is_NaN());
  }
};

expr2tc isnan2t::do_simplify() const
{
  // Negation and absolute value preserve NaN classification (IEEE 754
  // operations only flip / clear the sign bit, no payload change).
  if (is_neg2t(value))
    return isnan2tc(to_neg2t(value).value);
  if (is_abs2t(value))
    return isnan2tc(to_abs2t(value).value);

  return simplify_floatbv_1op<Isnantor, isnan2t>(type, value);
}

template <class constant_type>
struct Isinftor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;
    return constant_bool2tc(to_constant(c).value.is_infinity());
  }
};

expr2tc isinf2t::do_simplify() const
{
  // Negation and absolute value preserve the infinity classification.
  if (is_neg2t(value))
    return isinf2tc(to_neg2t(value).value);
  if (is_abs2t(value))
    return isinf2tc(to_abs2t(value).value);

  return simplify_floatbv_1op<Isinftor, isinf2t>(type, value);
}

template <class constant_type>
struct Isnormaltor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;
    return constant_bool2tc(to_constant(c).value.is_normal());
  }
};

expr2tc isnormal2t::do_simplify() const
{
  // Negation and absolute value preserve the normal/subnormal/zero/inf/NaN
  // classification — only the sign bit changes.
  if (is_neg2t(value))
    return isnormal2tc(to_neg2t(value).value);
  if (is_abs2t(value))
    return isnormal2tc(to_abs2t(value).value);

  return simplify_floatbv_1op<Isnormaltor, isnormal2t>(type, value);
}

template <class constant_type>
struct Isfinitetor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;
    return constant_bool2tc(to_constant(c).value.is_finite());
  }
};

expr2tc isfinite2t::do_simplify() const
{
  // Negation and absolute value preserve the finite classification.
  if (is_neg2t(value))
    return isfinite2tc(to_neg2t(value).value);
  if (is_abs2t(value))
    return isfinite2tc(to_abs2t(value).value);

  return simplify_floatbv_1op<Isfinitetor, isfinite2t>(type, value);
}

template <class constant_type>
struct Signbittor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    auto c = number;
    return constant_bool2tc(to_constant(c).value.get_sign());
  }
};

expr2tc signbit2t::do_simplify() const
{
  // signbit(abs(x)) = false. IEEE 754 absoluteValue clears the sign bit
  // unconditionally, including for NaN payloads.
  if (is_abs2t(operand))
    return gen_false_expr();

  // signbit(neg(x)) = !signbit(x). IEEE 754 negate flips the sign bit
  // unconditionally — same caveat for NaN, but the bit flip is
  // architecture-independent at the IR level.
  if (is_neg2t(operand))
    return not2tc(signbit2tc(to_neg2t(operand).value));

  return simplify_floatbv_1op<Signbittor, signbit2t>(type, operand);
}

expr2tc popcount2t::do_simplify() const
{
  // popcount(bswap(x)) = popcount(x). Byte-swap permutes bits but doesn't
  // add or remove any. Lets a constant-folded inner popcount fire if x is
  // itself constant, and shrinks the AST otherwise.
  if (is_bswap2t(operand))
    return popcount2tc(to_bswap2t(operand).value);

  if (!is_constant_int2t(operand))
    return expr2tc();

  // Use integer2binary at the operand's fixed BV width: integer2string
  // emits a magnitude/sign textual form (e.g. "-101" for -5), so counting
  // '1' chars on a negative signed BV value would miss the high one-bits
  // of two's-complement. integer2binary returns the exact two's-
  // complement bit pattern.
  const BigInt &v = to_constant_int2t(operand).value;
  std::string bin = integer2binary(v, operand->type->get_width());
  return constant_int2tc(type, count(bin.begin(), bin.end(), '1'));
}

expr2tc bswap2t::do_simplify() const
{
  // bswap(bswap(x)) = x. Reversing byte order twice is identity. Catches
  // defensive byte-swaps inserted by frontends and the symmetric
  // user-code "swap to network order, swap back" pattern.
  if (is_bswap2t(value) && to_bswap2t(value).value->type == type)
    return to_bswap2t(value).value;

  // Single-byte bswap is a no-op for any operand. Constant-fold loop below
  // already returns the same value for constant 8-bit operands, but
  // symbolic 8-bit operands would emit a useless bswap2t otherwise.
  if (type->get_width() <= 8 && value->type == type)
    return value;

  if (!is_constant_int2t(value))
    return expr2tc();

  const std::size_t bits_per_byte = 8;
  const std::size_t width = type->get_width();
  // Normalize to the unsigned two's-complement bit pattern. BigInt
  // arithmetic with shifts/modulo on a negative value produces signed
  // results that don't match byte-level 2c; round-trip through
  // integer2binary at the fixed width to get the exact bits, then
  // re-interpret as unsigned.
  const BigInt &raw = to_constant_int2t(value).value;
  BigInt v = binary2integer(integer2binary(raw, width), false);

  std::vector<BigInt> bytes;
  // take apart
  for (std::size_t bit = 0; bit < width; bit += bits_per_byte)
    bytes.push_back((v >> bit) % power(2, bits_per_byte));

  // put back together, but backwards
  BigInt new_value = 0;
  for (std::size_t bit = 0; bit < width; bit += bits_per_byte)
  {
    assert(!bytes.empty());
    new_value += bytes.back() << bit;
    bytes.pop_back();
  }

  // For a signedbv result type, from_integer will reinterpret the
  // unsigned bit pattern as signed via the same binary round-trip.
  return from_integer(new_value, type);
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_floatbv_2ops(
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2,
  const expr2tc &rounding_mode)
{
  assert(
    is_vector_type(type) ? is_floatbv_type(to_vector_type(type).subtype)
                         : is_floatbv_type(type));

  if (!is_number_type(type) && !is_pointer_type(type) && !is_vector_type(type))
    return expr2tc();

  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so the operands are already simplified. Local copies are kept because
  // the int->float coercion below mutates them in place.
  expr2tc simplified_side_1 = side_1;
  expr2tc simplified_side_2 = side_2;

  // Robustness: some frontends may build floatbv ops with integer/bool
  // constants as operands. Coerce them to float constants before trying
  // float simplification to avoid malformed FP nodes reaching the solver.
  if (is_floatbv_type(type) && is_constant_int2t(rounding_mode))
  {
    const BigInt rm_value = to_constant_int2t(rounding_mode).value;
    const auto rm = ieee_floatt::rounding_modet(rm_value.to_int64());
    const auto target_spec = to_floatbv_type(migrate_type_back(type));

    auto coerce_to_float_constant = [&](expr2tc &side) {
      if (is_constant_floatbv2t(side))
        return;

      if (is_constant_int2t(side))
      {
        ieee_floatt fp;
        fp.rounding_mode = rm;
        fp.spec = target_spec;
        fp.from_integer(to_constant_int2t(side).value);
        side = constant_floatbv2tc(fp);
        return;
      }

      if (is_constant_bool2t(side))
      {
        ieee_floatt fp;
        fp.rounding_mode = rm;
        fp.spec = target_spec;
        fp.from_integer(BigInt(to_constant_bool2t(side).value ? 1 : 0));
        side = constant_floatbv2tc(fp);
      }
    };

    coerce_to_float_constant(simplified_side_1);
    coerce_to_float_constant(simplified_side_2);
  }

  // Try to handle NaN
  if (is_constant_floatbv2t(simplified_side_1))
    if (to_constant_floatbv2t(simplified_side_1).value.is_NaN())
      return simplified_side_1;

  if (is_constant_floatbv2t(simplified_side_2))
    if (to_constant_floatbv2t(simplified_side_2).value.is_NaN())
      return simplified_side_2;

  if (
    !is_constant_expr(simplified_side_1) ||
    !is_constant_expr(simplified_side_2) || !is_constant_int2t(rounding_mode))
  {
    // Were we able to simplify the sides?
    if ((side_1 != simplified_side_1) || (side_2 != simplified_side_2))
    {
      expr2tc new_op(std::make_shared<constructor>(
        type, simplified_side_1, simplified_side_2, rounding_mode));

      return typecast_check_return(type, new_op);
    }

    return expr2tc();
  }

  expr2tc simpl_res = expr2tc();

  if (is_vector_type(type))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_floatbv2t;

    std::function<ieee_floatt &(expr2tc &)> get_value =
      [](expr2tc &c) -> ieee_floatt & {
      return to_constant_floatbv2t(c).value;
    };

    simpl_res = TFunctor<ieee_floatt>::simplify(
      simplified_side_1,
      simplified_side_2,
      rounding_mode,
      is_constant,
      get_value);
  }
  else if (
    is_floatbv_type(simplified_side_1) || is_floatbv_type(simplified_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_floatbv2t;

    std::function<ieee_floatt &(expr2tc &)> get_value =
      [](expr2tc &c) -> ieee_floatt & {
      return to_constant_floatbv2t(c).value;
    };

    simpl_res = TFunctor<ieee_floatt>::simplify(
      simplified_side_1,
      simplified_side_2,
      rounding_mode,
      is_constant,
      get_value);
  }
  else
  {
    // Do not abort on unexpected constant operand kinds; just skip
    // simplification for this node and keep the original expression.
    return expr2tc();
  }

  return typecast_check_return(type, simpl_res);
}

template <class constant_type>
struct IEEE_addtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const expr2tc &rm,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Is a vector operation ? Apply the op
    if (is_constant_vector2t(op1) || is_constant_vector2t(op2))
    {
      auto op = [rm](type2tc t, expr2tc e1, expr2tc e2) {
        return ieee_add2tc(t, e1, e2, rm);
      };
      return distribute_vector_operation(op, op1, op2);
    }

    // Two constants? Simplify to result of the addition
    if (is_constant(op1) && is_constant(op2))
    {
      ieee_floatt::rounding_modet mode =
        static_cast<ieee_floatt::rounding_modet>(
          to_constant_int2t(rm).value.to_int64());

      expr2tc c1 = op1;
      get_value(c1).rounding_mode = mode;

      expr2tc c2 = op2;
      get_value(c2).rounding_mode = mode;

      get_value(c1) += get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

expr2tc ieee_add2t::do_simplify() const
{
  return simplify_floatbv_2ops<IEEE_addtor, ieee_add2t>(
    type, side_1, side_2, rounding_mode);
}

template <class constant_type>
struct IEEE_subtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const expr2tc &rm,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Is a vector operation ? Apply the op
    if (is_constant_vector2t(op1) || is_constant_vector2t(op2))
    {
      auto op = [rm](type2tc t, expr2tc e1, expr2tc e2) {
        return ieee_sub2tc(t, e1, e2, rm);
      };
      return distribute_vector_operation(op, op1, op2);
    }
    // Two constants? Simplify to result of the subtraction
    if (is_constant(op1) && is_constant(op2))
    {
      ieee_floatt::rounding_modet mode =
        static_cast<ieee_floatt::rounding_modet>(
          to_constant_int2t(rm).value.to_int64());

      expr2tc c1 = op1;
      get_value(c1).rounding_mode = mode;

      expr2tc c2 = op2;
      get_value(c2).rounding_mode = mode;

      get_value(c1) -= get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

expr2tc ieee_sub2t::do_simplify() const
{
  return simplify_floatbv_2ops<IEEE_subtor, ieee_sub2t>(
    type, side_1, side_2, rounding_mode);
}

template <class constant_type>
struct IEEE_multor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const expr2tc &rm,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Is a vector operation ? Apply the op
    if (is_constant_vector2t(op1) || is_constant_vector2t(op2))
    {
      auto op = [rm](type2tc t, expr2tc e1, expr2tc e2) {
        return ieee_mul2tc(t, e1, e2, rm);
      };
      return distribute_vector_operation(op, op1, op2);
    }

    // Two constants? Simplify to result of the multiplication
    if (is_constant(op1) && is_constant(op2))
    {
      ieee_floatt::rounding_modet mode =
        static_cast<ieee_floatt::rounding_modet>(
          to_constant_int2t(rm).value.to_int64());

      expr2tc c1 = op1;
      get_value(c1).rounding_mode = mode;

      expr2tc c2 = op2;
      get_value(c2).rounding_mode = mode;

      get_value(c1) *= get_value(c2);
      return c1;
    }

    // x * 1 = x and 1 * x = x. Sound under IEEE 754: NaN propagates (NaN*1=
    // NaN, returning op preserves the NaN), zero stays zero with its sign,
    // and infinity stays infinity with its sign. Mirrors the existing
    // x / 1 = x rule in IEEE_divtor.
    if (is_constant(op2))
    {
      expr2tc c2 = op2;
      if (get_value(c2) == 1)
        return op1;
    }
    if (is_constant(op1))
    {
      expr2tc c1 = op1;
      if (get_value(c1) == 1)
        return op2;
    }

    return expr2tc();
  }
};

expr2tc ieee_mul2t::do_simplify() const
{
  return simplify_floatbv_2ops<IEEE_multor, ieee_mul2t>(
    type, side_1, side_2, rounding_mode);
}

template <class constant_type>
struct IEEE_divtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const expr2tc &rm,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    if (is_constant_vector2t(op1) || is_constant_vector2t(op2))
    {
      auto op = [rm](type2tc t, expr2tc e1, expr2tc e2) {
        return ieee_div2tc(t, e1, e2, rm);
      };
      return distribute_vector_operation(op, op1, op2);
    }
    // Two constants? Simplify to result of the division
    if (is_constant(op1) && is_constant(op2))
    {
      ieee_floatt::rounding_modet mode =
        static_cast<ieee_floatt::rounding_modet>(
          to_constant_int2t(rm).value.to_int64());

      expr2tc c1 = op1;
      get_value(c1).rounding_mode = mode;

      expr2tc c2 = op2;
      get_value(c2).rounding_mode = mode;

      get_value(c1) /= get_value(c2);
      return c1;
    }

    if (is_constant(op2))
    {
      ieee_floatt::rounding_modet mode =
        static_cast<ieee_floatt::rounding_modet>(
          to_constant_int2t(rm).value.to_int64());

      expr2tc c2 = op2;
      get_value(c2).rounding_mode = mode;

      // Denominator is one? Exact for all rounding modes.
      if (get_value(c2) == 1)
        return op1;
    }

    return expr2tc();
  }
};

expr2tc ieee_div2t::do_simplify() const
{
  return simplify_floatbv_2ops<IEEE_divtor, ieee_div2t>(
    type, side_1, side_2, rounding_mode);
}

expr2tc ieee_fma2t::do_simplify() const
{
  assert(is_floatbv_type(type));

  if (!is_number_type(type) && !is_pointer_type(type))
    return expr2tc();

  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so the three values are already simplified.
  if (
    !is_constant_expr(value_1) || !is_constant_expr(value_2) ||
    !is_constant_expr(value_3) || !is_constant_int2t(rounding_mode))
    return expr2tc();

  ieee_floatt n1 = to_constant_floatbv2t(value_1).value;
  ieee_floatt n2 = to_constant_floatbv2t(value_2).value;
  ieee_floatt n3 = to_constant_floatbv2t(value_3).value;

  // If x or y are NaN, NaN is returned
  if (n1.is_NaN() || n2.is_NaN())
  {
    n1.make_NaN();
    return constant_floatbv2tc(n1);
  }

  // If x is zero and y is infinite or if x is infinite and y is zero,
  // and z is not a NaN, a domain error shall occur, and either a NaN,
  // or an implementation-defined value shall be returned.

  // If x is zero and y is infinite or if x is infinite and y is zero,
  // and z is a NaN, then NaN is returned and FE_INVALID may be raised
  if ((n1.is_zero() && n2.is_infinity()) || (n2.is_zero() && n1.is_infinity()))
  {
    n1.make_NaN();
    return constant_floatbv2tc(n1);
  }

  // If z is NaN, and x*y aren't 0*Inf or Inf*0, then NaN is returned
  // (without FE_INVALID)
  if (n3.is_NaN())
  {
    n1.make_NaN();
    return constant_floatbv2tc(n1);
  }

  // If x*y is an exact infinity and z is an infinity with the opposite sign,
  // NaN is returned and FE_INVALID is raised
  n1 *= n2;
  if (
    (n1.is_infinity() && n3.is_infinity()) && (n1.get_sign() != n3.get_sign()))
  {
    n1.make_NaN();
    return constant_floatbv2tc(n1);
  }

  return expr2tc();
}

expr2tc ieee_sqrt2t::do_simplify() const
{
  if (!is_number_type(type))
    return expr2tc();

  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so `value` is already simplified.
  if (!is_constant_floatbv2t(value))
    return expr2tc();

  ieee_floatt n = to_constant_floatbv2t(value).value;
  if (n < 0)
  {
    n.make_NaN();
    return constant_floatbv2tc(n);
  }

  // sqrt(x) = x for x in {NaN, +/-0, +inf}. NaN and infinity propagate, and
  // sqrt(-0) is -0 / sqrt(+0) is +0 — both equal to the input.
  if (n.is_NaN() || n.is_zero() || n.is_infinity())
    return typecast_check_return(type, value);

  return expr2tc();
}

expr2tc constant_struct2t::do_simplify() const
{
  return expr2tc();
}

expr2tc constant_array2t::do_simplify() const
{
  // Don't fold uniform constant_array N to constant_array_of. The SMT
  // encoding for array_of (default_convert_array_of in smt_conv.cpp) writes
  // the initializer at *every* domain index, including indices past the
  // array's declared size — making out-of-bounds reads of the resulting
  // SMT array return the initializer instead of an unconstrained value.
  // For finite arrays under --no-bounds-check, that hides OOB-read
  // assertion violations. Leave constant_array as-is and let array_create
  // (smt_conv.cpp) populate only the actual range.
  return expr2tc();
}

expr2tc byte_extract2t::do_simplify() const
{
  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so source_value and source_offset are already simplified. Operand-
  // change rebuilding is handled by the framework.
  const expr2tc &src = source_value;
  const expr2tc &off = source_offset;

  // Read-after-write on the same source: byte_extract(byte_update(s, off1,
  // v), off2) folds when both offsets are constant AND the update is a
  // single byte. ESBMC's BV byte_update lowering writes exactly one byte
  // (asserting width(update_value) == 8); a wider update_value here would
  // mean we're inferring multi-byte semantics that don't hold at the
  // solver layer, so reading off+1 etc. could fabricate bytes from the
  // wider value that the SMT model never actually wrote.
  if (
    is_byte_update2t(src) && is_constant_int2t(off) &&
    is_constant_int2t(to_byte_update2t(src).source_offset))
  {
    const byte_update2t &bu = to_byte_update2t(src);
    const BigInt &off1 = to_constant_int2t(bu.source_offset).value;
    const BigInt &off2 = to_constant_int2t(off).value;
    // Three constraints all required:
    //   1. The extract result must be a single byte. A wider extract that
    //      overlaps the update would need to splice the update byte with
    //      bytes from the original source, which we don't model here.
    //   2. The update value must be a single byte. ESBMC's BV byte_update
    //      lowering writes exactly one byte; a wider update_value would
    //      mean we're inferring multi-byte semantics that don't hold at
    //      the solver layer.
    //   3. Endianness flags must match.
    if (
      off1.is_uint64() && off2.is_uint64() && type->get_width() == 8 &&
      bu.update_value->type->get_width() == 8 && bu.big_endian == big_endian)
    {
      uint64_t lo = off1.to_uint64();
      uint64_t r = off2.to_uint64();
      // Bounds check: the SMT byte_update lowering treats an out-of-bounds
      // src_offset as a no-op (smt_byteops.cpp:499-500), and byte_extract
      // at an OOB offset returns an out_of_bounds_byte_extract sentinel
      // (smt_byteops.cpp:187-191). Folding the read-after-write to the
      // update value would replace those backend semantics with the byte
      // we'd have written, hiding any OOB-read effect. Require both
      // offsets to be in range; if either is out of bounds, leave the
      // expression unsimplified.
      const BigInt source_bytes =
        type_byte_size(bu.source_value->type, migrate_namespace_lookup);
      if (
        !source_bytes.is_uint64() || lo >= source_bytes.to_uint64() ||
        r >= source_bytes.to_uint64())
      {
        return expr2tc();
      }
      if (r != lo)
      {
        // Update doesn't touch this byte; see through to the original source.
        return byte_extract2tc(type, bu.source_value, off, big_endian);
      }
      // Update writes the single byte at off2; the byte_extract result is
      // exactly the update_value (with the byte type).
      if (bu.update_value->type == type)
        return bu.update_value;
      return typecast2tc(type, bu.update_value);
    }
  }

  if (is_array_type(src))
  {
    const array_type2t &at = to_array_type(src->type);
    if (is_bv_type(at.subtype) && at.subtype->get_width() == type->get_width())
      return bitcast2tc(type, index2tc(at.subtype, src, off));
  }

  if (is_constant_int2t(off) && type == get_uint8_type())
  {
    const BigInt &off_value = to_constant_int2t(off).value;
    if (src->type == type && off_value.is_zero())
      return src;

    if (off_value.is_uint64() && is_constant_expr(src))
    {
      uint64_t off64 = off_value.to_uint64();
      if (is_constant_int2t(src) && off64 * 8 >= off64)
      {
        off64 *= 8;
        const BigInt &src_value = to_constant_int2t(src).value;
        bool neg = is_signedbv_type(src) && src_value.is_negative();
        unsigned width = src->type->get_width();
        /* width bits in ss...ss|...|ssssssss|xxxxxxxx|xxxxxxxx|...|xxxxxxxx|
         * at most 64 bits x; s = neg ? 1 : 0; off64 is in bits */
        if (
          (neg ? src_value.is_int64() : src_value.is_uint64()) &&
          off64 + 8 <= width)
        {
          /* We assume two's complement, as does do_bit_munge_operation() */

          /* constant repetition of sign bit? */
          if (big_endian ? off64 + 64 + 8 <= width : off64 >= 64)
            return constant_int2tc(type, BigInt(neg ? 0xff : 0x00));
          /* now we know that we are extracting part of |xxxxxxxx|...|xxxxxxxx| */
          uint64_t x = neg ? src_value.to_int64() : src_value.to_uint64();
          if (big_endian)
          {
            /* off64 + 64 + 8 > width and off64 + 8 <= width
             * i.e. width-off64-8 in [0,64) */
            off64 = width - off64 - 8;
          }
          return constant_int2tc(type, BigInt((x >> off64) & 0xff));
        }
        /* XXX how to simplify this? */
      }
    }
  }

  return expr2tc();
}

expr2tc byte_update2t::do_simplify() const
{
  // expr2t::simplify walks operands bottom-up before invoking do_simplify,
  // so source_value, source_offset, and update_value are already simplified.
  if (
    !is_constant_int2t(source_value) || !is_constant_int2t(source_offset) ||
    !is_constant_int2t(update_value))
    return expr2tc();

  std::string value = integer2binary(
    to_constant_int2t(update_value).value, update_value->type->get_width());
  std::string src_value = integer2binary(
    to_constant_int2t(source_value).value, source_value->type->get_width());

  // Reject offsets that don't fit in unsigned 64-bit, are negative, or
  // when src_offset * 8 + value.length() would overflow / fall outside
  // the source. Computing src_offset as a plain `int` and multiplying
  // by 8 used to silently truncate large constant offsets before the
  // bounds check, leaving the simplifier path open to crafted IR.
  const BigInt &off_big = to_constant_int2t(source_offset).value;
  if (!off_big.is_uint64())
    return expr2tc();
  uint64_t src_offset = off_big.to_uint64();
  // src_offset * 8 must fit in uint64_t, and the resulting range must lie
  // within src_value. Use a portable divide-based overflow check (works
  // on MSVC, which doesn't expose __builtin_mul_overflow): if multiplying
  // would overflow, src_offset > UINT64_MAX / 8.
  if (src_offset > std::numeric_limits<uint64_t>::max() / 8)
    return expr2tc();
  uint64_t bit_offset = src_offset * 8;
  // Overflow-safe bounds check: a naive bit_offset + value.length() > N
  // can wrap when bit_offset is near UINT64_MAX. Test bit_offset against
  // the source length first, then test the remaining tail length without
  // any addition. Also ensure bit_offset fits the platform's size_t.
  if (bit_offset > std::numeric_limits<size_t>::max())
    return expr2tc();
  if (bit_offset > src_value.length())
    return expr2tc();
  if (value.length() > src_value.length() - bit_offset)
    return expr2tc();

  // Reverse both the source value and the value that will be updated if we are
  // assuming little endian, because in string the pos 0 is the leftmost element
  // while in bvs, pos 0 is the rightmost bit
  if (!big_endian)
  {
    std::reverse(src_value.begin(), src_value.end());
    std::reverse(value.begin(), value.end());
  }

  src_value.replace(static_cast<size_t>(bit_offset), value.length(), value);

  // Reverse back
  if (!big_endian)
    std::reverse(src_value.begin(), src_value.end());

  return typecast_check_return(
    type,
    constant_int2tc(
      get_uint_type(src_value.length()), string2integer(src_value, 2)));
}

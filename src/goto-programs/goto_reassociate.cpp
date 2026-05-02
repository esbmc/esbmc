#include <goto-programs/goto_reassociate.h>

#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/migrate.h>

#include <vector>

namespace
{
/// One leaf of a flattened add/sub chain: a sub-expression and whether it
/// should be subtracted (negative=true) or added (negative=false).
struct signed_term
{
  bool negative;
  expr2tc term;
};

/// Recursively flatten an add/sub/neg chain into a list of signed terms.
///
/// Stops descending at any non-add/sub/neg node — that node becomes a leaf
/// term. The flatten is type-preserving: every leaf has the same type as the
/// root, since add/sub/neg always preserve the operand type in IRep2.
///
/// Examples:
///   ((2 - x) - 20) - 4*y
///     -> [(+,2), (-,x), (-,20), (-,4*y)]
///   3 - (5 - x)
///     -> [(+,3), (-,5), (+,x)]
///   -((x + y) - z)
///     -> [(-,x), (-,y), (+,z)]
void linearize_add_sub(
  const expr2tc &expr,
  bool negate,
  std::vector<signed_term> &out)
{
  if (is_add2t(expr))
  {
    const add2t &a = to_add2t(expr);
    linearize_add_sub(a.side_1, negate, out);
    linearize_add_sub(a.side_2, negate, out);
    return;
  }

  if (is_sub2t(expr))
  {
    const sub2t &s = to_sub2t(expr);
    linearize_add_sub(s.side_1, negate, out);
    linearize_add_sub(s.side_2, !negate, out); // RHS of sub flips sign
    return;
  }

  if (is_neg2t(expr))
  {
    linearize_add_sub(to_neg2t(expr).value, !negate, out);
    return;
  }

  // Leaf — try a local simplify so embedded constant chunks like 10*2 fold
  // before they're inserted as opaque terms.
  expr2tc leaf = expr;
  ::simplify(leaf);
  out.push_back({negate, leaf});
}

/// True if @p e is a node we will rewrite at the top level (add/sub/neg).
/// Reassociation only fires when at least one such node is present.
bool is_add_sub_root(const expr2tc &e)
{
  return is_add2t(e) || is_sub2t(e) || is_neg2t(e);
}

/// True if @p type is one we know how to fold constants in: bit-vectors and
/// bool. Floating-point is excluded — IEEE add is not associative.
bool reassoc_safe_type(const type2tc &type)
{
  return is_bv_type(type) || is_bool_type(type);
}

/// Build a balanced-ish add/sub tree from a list of signed terms.
/// Result's type is @p type. Returns nil if @p terms is empty.
expr2tc rebuild_chain(
  const type2tc &type,
  const std::vector<signed_term> &terms)
{
  if (terms.empty())
    return gen_zero(type);

  // Following LLVM Reassociate: emit a pure add chain where negative terms
  // are wrapped in neg2t. The local simplifier in expr_simplifier.cpp will
  // collapse `add(x, neg(y))` back to `sub(x, y)` and `add(neg(x), y)` to
  // `sub(y, x)` via its existing peepholes — so we don't need to special-case
  // those shapes here.
  auto materialize = [&](const signed_term &t) -> expr2tc {
    return t.negative ? neg2tc(type, t.term) : t.term;
  };

  expr2tc acc = materialize(terms[0]);
  for (std::size_t i = 1; i < terms.size(); ++i)
    acc = add2tc(type, acc, materialize(terms[i]));

  // One simplify pass over the rebuilt tree lets local peepholes collapse
  // add(x, neg(y)) -> sub(x, y) etc. Calling simplify between each rebuild
  // step would be O(n^2) on long chains and dominates the whole pass on
  // large benchmarks (e.g. github_2513_1).
  ::simplify(acc);
  return acc;
}

/// Optimize a linearized term list:
///  - sum all constants into a single trailing constant
///  - cancel matching X / -X pairs (one positive, one negative occurrence)
/// Returns true if anything was changed.
///
/// Constants are accumulated in @p acc_value as a `BigInt` and re-typed
/// through from_integer() at the end so the rebuilt constant fits the
/// target bit-width (otherwise narrow types like `unsigned char` would
/// see un-truncated sums).
///
/// Constants are never mutated in place: expr2tc shares storage, so writing
/// `to_constant_int2t(c).value +=` would corrupt every other use of that
/// constant in the program.
bool optimize_terms(const type2tc &type, std::vector<signed_term> &terms)
{
  bool changed = false;

  // 1. Sum constants. Track running BigInt + sign separately, then write
  //    once at the end via from_integer() to normalize to @p type's width.
  BigInt acc_value(0);
  bool acc_negative = false;
  bool have_const = false;
  for (std::size_t i = 0; i < terms.size();)
  {
    if (!is_constant_int2t(terms[i].term))
    {
      ++i;
      continue;
    }
    const BigInt &v = to_constant_int2t(terms[i].term).value;
    if (!have_const)
    {
      acc_value = v;
      acc_negative = terms[i].negative;
      have_const = true;
    }
    else if (acc_negative == terms[i].negative)
    {
      acc_value += v;
    }
    else if (acc_value >= v)
    {
      acc_value -= v;
    }
    else
    {
      acc_value = v - acc_value;
      acc_negative = terms[i].negative;
    }
    terms.erase(terms.begin() + i);
    changed = true;
  }

  if (have_const && !acc_value.is_zero())
  {
    // Re-typed via from_integer(exprt overload) + migrate_expr so the value
    // is truncated to the result bit-width (otherwise, e.g., `unsigned char`
    // sums above 255 would carry the full BigInt and confuse the encoder).
    expr2tc folded;
    migrate_expr(from_integer(acc_value, migrate_type_back(type)), folded);
    terms.push_back({acc_negative, folded});
  }

  // 2. Cancel matching X / -X pairs. O(n^2) but n is tiny in practice.
  for (std::size_t i = 0; i < terms.size();)
  {
    bool erased = false;
    for (std::size_t j = i + 1; j < terms.size(); ++j)
    {
      if (terms[i].negative == terms[j].negative)
        continue;
      if (!(terms[i].term == terms[j].term))
        continue;
      // Found X with one sign and X with the other — drop both.
      terms.erase(terms.begin() + j);
      terms.erase(terms.begin() + i);
      erased = true;
      changed = true;
      break;
    }
    if (!erased)
      ++i;
  }

  return changed;
}

/// Walk @p expr and reassociate every add/sub subtree it contains.
/// Mutates in place. Returns true if any rewrite was performed.
bool reassociate_expr(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return false;

  // Don't descend into expressions whose semantics depend on the exact
  // arithmetic shape:
  //   overflow_*: rewriting the operand changes whether overflow is detected
  //   address_of: rewriting an index expression can change which address
  //               is taken (the symex simplifier guards against the same).
  if (
    expr->expr_id == expr2t::overflow_id ||
    expr->expr_id == expr2t::overflow_cast_id ||
    expr->expr_id == expr2t::overflow_neg_id ||
    expr->expr_id == expr2t::address_of_id)
    return false;

  bool changed = false;

  // Bottom-up: rewrite operands first so a parent reassoc sees the
  // already-folded children.
  expr->Foreach_operand([&changed](expr2tc &op) {
    if (reassociate_expr(op))
      changed = true;
  });

  // Only attempt reassoc on integer/bool add/sub/neg roots.
  if (!is_add_sub_root(expr) || !reassoc_safe_type(expr->type))
    return changed;

  // Linearize the whole chain.
  std::vector<signed_term> terms;
  linearize_add_sub(expr, /*negate=*/false, terms);

  // No reassoc opportunity unless the chain has at least 2 terms (which
  // it will if expr is itself an add/sub/neg, by construction).
  if (terms.size() < 2)
    return changed;

  // Save the original size — if optimize doesn't change anything, leave
  // expr as-is to avoid churning structurally-equivalent trees.
  const std::size_t orig_size = terms.size();
  if (!optimize_terms(expr->type, terms) && terms.size() == orig_size)
    return changed;

  expr = rebuild_chain(expr->type, terms);
  return true;
}
} // namespace

void goto_reassociate(goto_functionst &goto_functions)
{
  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available)
      continue;

    Forall_goto_program_instructions (i_it, f_it->second.body)
    {
      reassociate_expr(i_it->code);
      reassociate_expr(i_it->guard);
    }
  }

  goto_functions.update();
}

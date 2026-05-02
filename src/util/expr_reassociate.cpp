#include <util/expr_reassociate.h>

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

/// True if @p operand of an add/sub node is acceptable to descend into
/// while flattening a chain of type @p chain_type. Operand can be:
///   - of type chain_type (genuine chain participant), or
///   - of integer type when chain_type is a pointer (pointer-arith offset).
bool chain_compatible(const expr2tc &operand, const type2tc &chain_type)
{
  if (operand->type == chain_type)
    return true;
  if (is_pointer_type(chain_type) && is_bv_type(operand))
    return true;
  return false;
}

/// Recursively flatten an add/sub/neg chain into a list of signed terms.
///
/// At each add/sub/neg node N we may visit, we descend only if:
///   - N.type == chain_type, AND
///   - every operand of N is chain_compatible with chain_type.
/// Otherwise N (and everything beneath it) becomes a single opaque leaf.
///
/// This keeps the chain homogeneously typed and prevents unsound
/// transitions, e.g. an integer-typed `sub(p, q)` (pointer minus pointer)
/// embedded in an integer chain — it has integer result type but pointer
/// operands, so we treat it as opaque rather than leaking pointer leaves
/// into the integer chain.
///
/// Pointer-typed `neg` is rejected: `-p` is malformed C and we have no
/// sensible interpretation as a chain participant.
///
/// Examples (chain_type = int):
///   ((2 - x) - 20) - 4*y
///     -> [(+,2), (-,x), (-,20), (-,4*y)]
///   3 - (5 - x)
///     -> [(+,3), (-,5), (+,x)]
///   -((x + y) - z)
///     -> [(-,x), (-,y), (+,z)]
///
/// Examples (chain_type = pointer):
///   ((p + 1) + 1) + 1
///     -> [(+,p), (+,1), (+,1), (+,1)]
///   (p + n) - m   (n, m integer)
///     -> [(+,p), (+,n), (-,m)]
void linearize_add_sub(
  const expr2tc &expr,
  const type2tc &chain_type,
  bool negate,
  std::vector<signed_term> &out)
{
  // Helper: would this whole node be a valid chain participant?
  auto can_descend_add_or_sub = [&](const expr2tc &n) -> bool {
    expr2tc s1, s2;
    if (is_add2t(n))
      s1 = to_add2t(n).side_1, s2 = to_add2t(n).side_2;
    else if (is_sub2t(n))
      s1 = to_sub2t(n).side_1, s2 = to_sub2t(n).side_2;
    else
      return false;
    return n->type == chain_type &&
           chain_compatible(s1, chain_type) &&
           chain_compatible(s2, chain_type);
  };

  if (is_add2t(expr) && can_descend_add_or_sub(expr))
  {
    const add2t &a = to_add2t(expr);
    linearize_add_sub(a.side_1, chain_type, negate, out);
    linearize_add_sub(a.side_2, chain_type, negate, out);
    return;
  }

  if (is_sub2t(expr) && can_descend_add_or_sub(expr))
  {
    const sub2t &s = to_sub2t(expr);
    linearize_add_sub(s.side_1, chain_type, negate, out);
    linearize_add_sub(s.side_2, chain_type, !negate, out);
    return;
  }

  // neg2t: descend only when the inner value type matches the chain and
  // the chain itself is non-pointer. `-p` for a pointer is malformed and
  // we don't model "negative pointer term" in the rebuilt chain.
  if (
    is_neg2t(expr) && expr->type == chain_type &&
    !is_pointer_type(chain_type) &&
    to_neg2t(expr).value->type == chain_type)
  {
    linearize_add_sub(to_neg2t(expr).value, chain_type, !negate, out);
    return;
  }

  // Opaque leaf. The caller (expr2t::simplify) walks operands bottom-up
  // before invoking us, so this expression is already canonical — calling
  // ::simplify on it here would just re-walk its subtree and, on deep
  // expressions, blow the stack via the chain-root reassoc -> simplify ->
  // linearize loop.
  out.push_back({negate, expr});
}

/// True if @p e is a node we will rewrite at the top level (add/sub/neg).
/// Reassociation only fires when at least one such node is present.
bool is_add_sub_root(const expr2tc &e)
{
  return is_add2t(e) || is_sub2t(e) || is_neg2t(e);
}

/// True if @p type is one we know how to fold constants in: bit-vectors,
/// bool, and pointer. Floating-point is excluded — IEEE add is not
/// associative. Pointer chains are allowed: their integer offsets fold
/// via the chain_compatible rule in linearize_add_sub, which keeps the
/// pointer leaf as a positive participant and treats integer offsets as
/// signed leaves of integer type.
bool reassoc_safe_type(const type2tc &type)
{
  return is_bv_type(type) || is_bool_type(type) || is_pointer_type(type);
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
  //
  // Use each term's own type for neg2t, not the chain's root type: in a
  // pointer chain the root is pointer but the integer offsets are e.g.
  // `signed long`; we must not synthesize neg2t(pointer_type, ...).
  auto materialize = [](const signed_term &t) -> expr2tc {
    return t.negative ? neg2tc(t.term->type, t.term) : t.term;
  };

  expr2tc acc = materialize(terms[0]);
  for (std::size_t i = 1; i < terms.size(); ++i)
    acc = add2tc(type, acc, materialize(terms[i]));

  // The caller (expr2t::simplify's chain-root step) runs simplify_no_reassoc
  // on the result so add(x, neg(y)) -> sub(x, y) and friends collapse via
  // the existing peepholes. Doing it inside rebuild would force an extra
  // recursion and risks re-entering the reassoc path.
  return acc;
}

/// Optimize a linearized term list:
///  - sum all constants into a single trailing constant
///  - cancel matching X / -X pairs (one positive, one negative occurrence)
/// Returns true if anything was changed.
///
/// Constants are accumulated in a local BigInt and re-typed through
/// from_integer() at the end so the rebuilt constant fits the target
/// bit-width (otherwise narrow types like `unsigned char` would see
/// un-truncated sums).
///
/// The folded constant takes the type of the first integer leaf seen in
/// the term list, not the chain's root type. For pure-integer chains
/// these match. For pointer chains the root is pointer but the integer
/// offsets are e.g. `signed long`, and we must not synthesize a
/// pointer-typed integer constant.
///
/// Constants are never mutated in place: expr2tc shares storage, so writing
/// `to_constant_int2t(c).value +=` would corrupt every other use of that
/// constant in the program.
bool optimize_terms(std::vector<signed_term> &terms)
{
  bool changed = false;

  BigInt acc_value(0);
  bool acc_negative = false;
  bool have_const = false;
  type2tc const_type;
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
      const_type = terms[i].term->type;
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
    // is truncated to the result bit-width.
    expr2tc folded;
    migrate_expr(
      from_integer(acc_value, migrate_type_back(const_type)), folded);
    terms.push_back({acc_negative, folded});
  }

  // Cancel matching X / -X pairs. O(n^2) but n is tiny in practice.
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
} // namespace

bool reassociate_arith(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return false;

  // Only attempt reassoc on integer/bool add/sub/neg roots. We do NOT
  // descend into operands here: the caller is expr2t::simplify(), which
  // already simplifies operands bottom-up and invokes reassociate_arith
  // exactly once at each chain root. Recursing here would duplicate that
  // work and re-enter via rebuild_chain's local resimplify.
  if (!is_add_sub_root(expr) || !reassoc_safe_type(expr->type))
    return false;

  std::vector<signed_term> terms;
  linearize_add_sub(expr, expr->type, /*negate=*/false, terms);

  if (terms.size() < 2)
    return false;

  const std::size_t orig_size = terms.size();
  if (!optimize_terms(terms) && terms.size() == orig_size)
    return false;

  expr = rebuild_chain(expr->type, terms);
  return true;
}

void simplify_no_reassoc(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;
  expr2tc tmp = expr->simplify(/*inside_chain=*/true);
  if (!is_nil_expr(tmp))
    expr = tmp;
}

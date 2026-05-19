#include <util/expr_reassociate.h>

#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>

#include <optional>
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

/// Per-element comparison: same sign and structurally-equal term (using
/// expr2tc's deep-equality `operator==`). Used to gate the rebuild step in
/// reassociate_arith so a "rewrite" that produces a logically identical term
/// list cannot fool the simplifier's "nil iff unchanged" contract.
bool signed_terms_equal(
  const std::vector<signed_term> &a,
  const std::vector<signed_term> &b)
{
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].negative != b[i].negative || !(a[i].term == b[i].term))
      return false;
  return true;
}

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
    return n->type == chain_type && chain_compatible(s1, chain_type) &&
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
    !is_pointer_type(chain_type) && to_neg2t(expr).value->type == chain_type)
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

/// Build a left-leaning add/sub tree from a list of signed terms.
/// Returns gen_zero(type) if @p terms is empty.
///
/// This always constructs fresh expression nodes. Callers must only invoke it
/// after proving the term list is meaningfully different from the input chain;
/// rebuilding an equivalent chain can still affect simplifier change tracking.
expr2tc
rebuild_chain(const type2tc &type, const std::vector<signed_term> &terms)
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

  // Defense-in-depth for pointer chains: every intermediate add we emit
  // is typed as `pointer_type`, so we must seed `acc` with a pointer-
  // typed term. Otherwise a chain like `[i, j, p, ...]` (integer leaves
  // before the pointer base) would synthesize add2tc(pointer_type, i, j)
  // — a pointer-typed add with two integer operands, violating the
  // pointer-arithmetic invariant. The C frontend doesn't currently emit
  // shapes that hit this path, but a future frontend change easily could.
  // If we're rebuilding a pointer-typed chain and no positive pointer
  // base is found, refuse the rebuild rather than seeding from an
  // integer term and producing a malformed pointer expression.
  std::size_t base_idx = 0;
  if (is_pointer_type(type))
  {
    bool found = false;
    for (std::size_t k = 0; k < terms.size(); ++k)
    {
      if (is_pointer_type(terms[k].term) && !terms[k].negative)
      {
        base_idx = k;
        found = true;
        break;
      }
    }
    if (!found)
      return expr2tc();
  }

  // Pointer chains preserve negative offsets via sub2tc directly, not by
  // wrapping in neg2t and emitting add. Reason: neg2t on an unsigned bv
  // is modular negation ((2^width - x) mod 2^width), which yields a large
  // positive value. When that gets sign/zero-extended to the pointer
  // offset width inside SMT pointer-arith conversion, the pointer moves
  // forward by ~UINT_MAX instead of backward by x. Emitting
  // sub2tc(pointer, acc, term) keeps the subtraction at the IR level.
  // For non-pointer chains, modular wrap matches bv semantics, so the
  // existing add+neg lowering stays correct.
  const bool ptr_chain = is_pointer_type(type);
  expr2tc acc = materialize(terms[base_idx]);
  for (std::size_t i = 0; i < terms.size(); ++i)
  {
    if (i == base_idx)
      continue;
    if (ptr_chain && terms[i].negative)
      acc = sub2tc(type, acc, terms[i].term);
    else
      acc = add2tc(type, acc, materialize(terms[i]));
  }

  // The caller (expr2t::simplify's chain-root step) runs simplify_no_reassoc
  // on the result so add(x, neg(y)) -> sub(x, y) and friends collapse via
  // the existing peepholes. Doing it inside rebuild would force an extra
  // recursion and risks re-entering the reassoc path.
  return acc;
}

/// Optimize a linearized term list. Returns a replacement vector if a real
/// rewrite is possible, std::nullopt otherwise.
///
/// Three transforms apply:
///
///   1. Constant folding: when two or more `constant_int2t` entries are
///      present, sum them into one `BigInt` and re-emit as a single term.
///      Re-typed through `from_integer()` so the rebuilt constant fits the
///      target bit-width (otherwise narrow types like `unsigned char`
///      would see un-truncated sums). The folded constant takes the type
///      of the first integer leaf seen in the term list, not the chain's
///      root type — for pure-integer chains these match, but for pointer
///      chains the root is pointer while the integer offsets are e.g.
///      `signed long`, and we must not synthesize a pointer-typed integer
///      constant.
///
///   2. Identity removal: a lone constant equal to zero contributes
///      nothing to an add/sub chain and is dropped (`x +/- 0 = x`). This
///      is an explicit add/sub identity rule, separate from the multi-
///      constant fold above.
///
///   3. X / -X cancellation: matching pairs (one positive occurrence, one
///      negative occurrence of the same term) annihilate.
///
/// Returning std::nullopt encodes "no rewrite is possible". Returning a
/// vector encodes "here is the new term list" — note that the caller must
/// still verify the new list differs from the input (via
/// signed_terms_equal) before committing, because some transforms above
/// can produce structurally-identical output.
///
/// Constants are never mutated in place: expr2tc shares storage, so writing
/// `to_constant_int2t(c).value +=` would corrupt every other use of that
/// constant in the program.

/// Check that @p value fits in @p type without modular wrap. Used to gate
/// pointer-chain folds: each unfolded `add(pointer, ptr, c)` step casts c to
/// the pointer offset width before the SMT bv add (smt_memspace.cpp). A
/// fold that wraps a narrow constant (e.g. (s8)100 + (s8)100 = (s8)-56)
/// would change the sign-extended offset value, silently shifting the
/// pointer. For non-pointer chains the wrap matches bv semantics, so this
/// guard is only enforced when the chain root is pointer-typed.
static bool fits_in_signed_type(const BigInt &value, const type2tc &type)
{
  if (!is_signedbv_type(type) && !is_unsignedbv_type(type))
    return true;
  const unsigned width = type->get_width();
  if (is_signedbv_type(type))
  {
    BigInt min_val = -(BigInt::power2(width - 1));
    BigInt max_val = BigInt::power2(width - 1) - 1;
    return value >= min_val && value <= max_val;
  }
  if (value < 0)
    return false;
  return value <= BigInt::power2(width) - 1;
}

std::optional<std::vector<signed_term>>
optimize_terms(const type2tc &chain_type, const std::vector<signed_term> &terms)
{
  bool changed = false;

  BigInt acc_value(0);
  bool acc_negative = false;
  bool have_const = false;
  type2tc const_type;
  std::size_t const_count = 0;
  // For the lone-constant case we re-emit the *original* signed_term
  // unchanged. Recreating it via from_integer() would needlessly normalize
  // the constant, and if a later cancel-pair pass flips `changed` to true
  // we would commit a structurally-different but logically-equal term list.
  const signed_term *lone_const = nullptr;
  std::vector<signed_term> result;
  result.reserve(terms.size());

  // Pass 1: separate constants from non-constants and accumulate the
  // constant portion into a single BigInt.
  //
  // Constants are folded only when they all share a single type. If the
  // chain mixes integer types (e.g. a narrow offset combined with a wide
  // one), re-emitting the sum with the first type's width would silently
  // truncate via from_integer. Treat type-incompatible constants as opaque
  // non-constant leaves to keep them in the term list unchanged.
  for (const auto &term : terms)
  {
    if (!is_constant_int2t(term.term))
    {
      result.push_back(term);
      continue;
    }

    if (have_const && term.term->type != const_type)
    {
      result.push_back(term);
      continue;
    }

    ++const_count;
    const BigInt &v = to_constant_int2t(term.term).value;
    if (!have_const)
    {
      acc_value = v;
      acc_negative = term.negative;
      const_type = term.term->type;
      have_const = true;
      lone_const = &term;
    }
    else if (acc_negative == term.negative)
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
      acc_negative = term.negative;
    }
  }

  if (const_count > 1)
  {
    // Pointer-chain soundness: each unfolded `add(pointer, ptr, c)` step
    // sign-extends c to the pointer offset width before the SMT bv add.
    // Folding two narrow constants whose sum wraps would change the
    // sign-extended value (e.g. (s8)100 + (s8)100 = (s8)-56). For
    // non-pointer chains the wrap is part of the bv semantics and the
    // unfolded chain has the same wrap, so we only refuse the fold for
    // pointer-typed chains. Treat the constants as opaque non-constant
    // leaves and keep them in the result.
    if (
      is_pointer_type(chain_type) &&
      !fits_in_signed_type(acc_negative ? -acc_value : acc_value, const_type))
    {
      // Re-emit each constant in original signed_term form so the result
      // matches the input list (no spurious "changed" signal).
      for (const auto &term : terms)
        if (is_constant_int2t(term.term) && term.term->type == const_type)
          result.push_back(term);
    }
    else
    {
      // Two or more constants folded into one — that's a real rewrite.
      if (!acc_value.is_zero())
        result.push_back({acc_negative, from_integer(acc_value, const_type)});
      changed = true;
    }
  }
  else if (const_count == 1 && acc_value.is_zero())
  {
    // Lone zero constant — additive identity, drop it.
    changed = true;
  }
  else if (const_count == 1)
  {
    // Lone non-zero constant — preserve the original term unchanged so
    // any later cancel-pair commit cannot pair it with a recreated copy.
    result.push_back(*lone_const);
  }

  // Pass 2: cancel matching X / -X pairs by marking and compacting. Avoids
  // the O(n^2) shift cost of repeated vector::erase. Each pair found marks
  // both endpoints and stops the inner search; the outer loop skips dead
  // entries, so the total work is O(n^2) comparisons but only one compact
  // walk afterwards.
  if (result.size() >= 2)
  {
    bool any_cancelled = false;
    std::vector<bool> dead(result.size(), false);
    for (std::size_t i = 0; i < result.size(); ++i)
    {
      if (dead[i])
        continue;
      for (std::size_t j = i + 1; j < result.size(); ++j)
      {
        if (dead[j])
          continue;
        if (result[i].negative == result[j].negative)
          continue;
        if (!(result[i].term == result[j].term))
          continue;
        dead[i] = true;
        dead[j] = true;
        any_cancelled = true;
        break;
      }
    }
    if (any_cancelled)
    {
      std::size_t out = 0;
      for (std::size_t i = 0; i < result.size(); ++i)
        if (!dead[i])
          result[out++] = std::move(result[i]);
      result.resize(out);
      changed = true;
    }
  }

  if (!changed)
    return std::nullopt;
  return result;
}

// === Multiplication chain reassoc ============================================

/// Per-element comparison for mul-chain term lists. Mul has no sign field, so
/// the lists are plain `std::vector<expr2tc>` and equality is structural.
bool mul_terms_equal(
  const std::vector<expr2tc> &a,
  const std::vector<expr2tc> &b)
{
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!(a[i] == b[i]))
      return false;
  return true;
}

/// True if @p type is one we know how to fold mul-chain constants in. Pointer
/// is excluded (pointer * _ isn't a valid C operation); float/fixedbv are
/// excluded (mul is not associative under rounding).
bool reassoc_mul_safe_type(const type2tc &type)
{
  return is_bv_type(type) || is_bool_type(type);
}

/// Recursively flatten a mul chain into a list of leaves.
///
/// Descends only when the mul node and both operands have type @p chain_type.
/// Mul doesn't allow mixed-type operands the way pointer-add does (you can't
/// multiply a pointer by an integer in C), so the rule is the simpler
/// strict-match form. Nested muls of mismatched types become opaque leaves.
void linearize_mul(
  const expr2tc &expr,
  const type2tc &chain_type,
  std::vector<expr2tc> &out)
{
  auto can_descend = [&](const expr2tc &n) -> bool {
    if (!is_mul2t(n) || n->type != chain_type)
      return false;
    const mul2t &m = to_mul2t(n);
    return m.side_1->type == chain_type && m.side_2->type == chain_type;
  };

  if (can_descend(expr))
  {
    const mul2t &m = to_mul2t(expr);
    linearize_mul(m.side_1, chain_type, out);
    linearize_mul(m.side_2, chain_type, out);
    return;
  }

  out.push_back(expr);
}

/// Optimize a flattened mul-chain term list. Returns the replacement vector
/// when a real rewrite is possible, std::nullopt otherwise.
///
/// The only transform is constant folding: when two or more `constant_int2t`
/// entries are present, multiply them into a single `BigInt` and re-emit as
/// one term, re-typed via `from_integer()` for bit-width truncation.
///
/// Identity (`x * 1 = x`) and absorber (`x * 0 = 0`) rules are left to the
/// per-op peephole in `mul2t::do_simplify`, which fires before reassoc gets
/// invoked. A surviving lone constant in the term list means the per-op
/// peephole couldn't fold it (e.g. it's a constant operand of a chain mul
/// where the other side is symbolic), so we must preserve it as-is.
std::optional<std::vector<expr2tc>>
optimize_mul_terms(const std::vector<expr2tc> &terms)
{
  std::size_t const_count = 0;
  BigInt acc(1);
  type2tc const_type;
  bool saw_zero = false;
  std::vector<expr2tc> result;
  result.reserve(terms.size());

  // Pass 1: separate constants from non-constants and accumulate the product.
  for (const auto &term : terms)
  {
    if (!is_constant_int2t(term))
    {
      result.push_back(term);
      continue;
    }

    ++const_count;
    const BigInt &v = to_constant_int2t(term).value;
    if (v.is_zero())
      saw_zero = true;
    if (const_count == 1)
    {
      acc = v;
      const_type = term->type;
    }
    else
      acc *= v;
  }

  if (const_count < 2)
    return std::nullopt;

  // Two or more constants folded. Watch for the absorber: if any constant
  // was zero, the product is zero — emit a single zero term. Otherwise emit
  // the folded product (truncated through from_integer for the chain type).
  if (saw_zero)
  {
    result.clear();
    result.push_back(from_integer(BigInt(0), const_type));
    return result;
  }

  // If the folded constant is 1, it's the multiplicative identity and can be
  // dropped — unless the result would be empty, in which case we keep a
  // single 1 term so the rebuilt chain has something to emit.
  if (acc == BigInt(1))
  {
    if (result.empty())
      result.push_back(from_integer(acc, const_type));
    return result;
  }

  result.push_back(from_integer(acc, const_type));
  return result;
}

/// Build a left-leaning mul chain from a list of terms. Returns nil if @p
/// terms is empty (the caller guarantees a non-empty list when commit is
/// possible: `optimize_mul_terms` always returns at least one term, and the
/// caller short-circuits on terms.size() < 2 before calling).
expr2tc
rebuild_mul_chain(const type2tc &type, const std::vector<expr2tc> &terms)
{
  if (terms.empty())
    return gen_zero(type); // unreachable in current call path; defensive

  expr2tc acc = terms[0];
  for (std::size_t i = 1; i < terms.size(); ++i)
    acc = mul2tc(type, acc, terms[i]);
  return acc;
}

// === Bitwise chain reassoc ===================================================

/// True if @p type is one we know how to fold bitwise-chain constants in.
/// Bool is allowed (bitwise on bool is logically meaningful). Pointer and
/// float are excluded — bitwise ops on pointers aren't valid C and on
/// floats aren't defined.
bool reassoc_bitwise_safe_type(const type2tc &type)
{
  return is_bv_type(type) || is_bool_type(type);
}

/// Apply a bitwise op to two BigInts via uint64 round-trip. Returns nullopt
/// if either input doesn't fit in uint64 OR if the destination type is
/// wider than 64 bits (mirrors do_bit_munge_operation's 64-bit-bound for
/// constant folding). The destination width matters because the 64-bit
/// result needs zero/sign extension to the destination, and we can't
/// reliably reconstruct the high bits from a 64-bit pattern: e.g.
/// (s128)-2 & (s128)-4 produces 64-bit 0xFF...FC; with a positive
/// BigInt that zero-extends to 2^64-4 instead of -4 in s128.
template <typename U64Op>
std::optional<BigInt>
bitwise_fold(const BigInt &a, const BigInt &b, const type2tc &dest, U64Op op)
{
  // Refuse to fold for destination types wider than 64 bits. The 64-bit
  // round-trip can't preserve the sign extension above bit 63 reliably.
  if (
    dest && (is_signedbv_type(dest) || is_unsignedbv_type(dest)) &&
    dest->get_width() > 64)
    return std::nullopt;

  // Use the same two's-complement round-trip as do_bit_munge_operation:
  // signed values reach this via int64_t and unsigned via uint64_t. For
  // bitwise ops, what we care about is the underlying bit pattern, so we
  // accept either form as long as it fits in 64 bits.
  auto fits = [](const BigInt &x) { return x.is_int64() || x.is_uint64(); };
  if (!fits(a) || !fits(b))
    return std::nullopt;
  uint64_t la = a.is_uint64() ? a.to_uint64() : (uint64_t)a.to_int64();
  uint64_t lb = b.is_uint64() ? b.to_uint64() : (uint64_t)b.to_int64();
  // Build the BigInt as unsigned. Casting through int64_t would
  // sign-extend any high bit set in the result, and from_integer() for
  // wider destination types (e.g. >64-bit unsignedbv) would then sign-
  // extend further. e.g. (uint128)(1<<63) & (uint128)(1<<63) must stay
  // 2^63, not become 2^128 - 2^63.
  return BigInt((unsigned long long)op(la, lb));
}

void linearize_bitand(
  const expr2tc &expr,
  const type2tc &chain_type,
  std::vector<expr2tc> &out)
{
  if (is_bitand2t(expr) && expr->type == chain_type)
  {
    const bitand2t &op = to_bitand2t(expr);
    if (op.side_1->type == chain_type && op.side_2->type == chain_type)
    {
      linearize_bitand(op.side_1, chain_type, out);
      linearize_bitand(op.side_2, chain_type, out);
      return;
    }
  }
  out.push_back(expr);
}

void linearize_bitor(
  const expr2tc &expr,
  const type2tc &chain_type,
  std::vector<expr2tc> &out)
{
  if (is_bitor2t(expr) && expr->type == chain_type)
  {
    const bitor2t &op = to_bitor2t(expr);
    if (op.side_1->type == chain_type && op.side_2->type == chain_type)
    {
      linearize_bitor(op.side_1, chain_type, out);
      linearize_bitor(op.side_2, chain_type, out);
      return;
    }
  }
  out.push_back(expr);
}

void linearize_bitxor(
  const expr2tc &expr,
  const type2tc &chain_type,
  std::vector<expr2tc> &out)
{
  if (is_bitxor2t(expr) && expr->type == chain_type)
  {
    const bitxor2t &op = to_bitxor2t(expr);
    if (op.side_1->type == chain_type && op.side_2->type == chain_type)
    {
      linearize_bitxor(op.side_1, chain_type, out);
      linearize_bitxor(op.side_2, chain_type, out);
      return;
    }
  }
  out.push_back(expr);
}

expr2tc
rebuild_bitand_chain(const type2tc &type, const std::vector<expr2tc> &terms)
{
  if (terms.empty())
    return gen_zero(type); // unreachable in current call path; defensive
  expr2tc acc = terms[0];
  for (std::size_t i = 1; i < terms.size(); ++i)
    acc = bitand2tc(type, acc, terms[i]);
  return acc;
}

expr2tc
rebuild_bitor_chain(const type2tc &type, const std::vector<expr2tc> &terms)
{
  if (terms.empty())
    return gen_zero(type);
  expr2tc acc = terms[0];
  for (std::size_t i = 1; i < terms.size(); ++i)
    acc = bitor2tc(type, acc, terms[i]);
  return acc;
}

expr2tc
rebuild_bitxor_chain(const type2tc &type, const std::vector<expr2tc> &terms)
{
  if (terms.empty())
    return gen_zero(type);
  expr2tc acc = terms[0];
  for (std::size_t i = 1; i < terms.size(); ++i)
    acc = bitxor2tc(type, acc, terms[i]);
  return acc;
}

/// Optimize a flattened bitand-chain term list. Folds `constant_int2t`
/// entries via bitwise AND. Identity (`x & -1 = x`) and absorber
/// (`x & 0 = 0`) at the chain top are caught by the per-op peephole, but
/// chain-internal leaves (from cross-chain flattening) are handled here.
std::optional<std::vector<expr2tc>>
optimize_bitand_terms(const std::vector<expr2tc> &terms)
{
  std::size_t const_count = 0;
  BigInt acc(-1); // identity for AND: all bits set
  type2tc const_type;
  bool saw_zero = false;
  std::vector<expr2tc> result;
  result.reserve(terms.size());

  for (const auto &term : terms)
  {
    if (!is_constant_int2t(term))
    {
      result.push_back(term);
      continue;
    }

    ++const_count;
    const BigInt &v = to_constant_int2t(term).value;
    if (v.is_zero())
      saw_zero = true;
    if (const_count == 1)
    {
      acc = v;
      const_type = term->type;
    }
    else
    {
      auto folded = bitwise_fold(
        acc, v, const_type, [](uint64_t a, uint64_t b) { return a & b; });
      if (!folded)
        return std::nullopt; // operands too wide for 64-bit fold; bail
      acc = *folded;
    }
  }

  if (const_count < 2)
    return std::nullopt;

  if (saw_zero)
  {
    result.clear();
    result.push_back(from_integer(BigInt(0), const_type));
    return result;
  }

  if (acc == BigInt(-1))
  {
    if (result.empty())
      result.push_back(from_integer(acc, const_type));
    return result;
  }

  result.push_back(from_integer(acc, const_type));
  return result;
}

/// Optimize a flattened bitor-chain term list. Absorber: -1. Identity: 0.
std::optional<std::vector<expr2tc>>
optimize_bitor_terms(const std::vector<expr2tc> &terms)
{
  std::size_t const_count = 0;
  BigInt acc(0);
  type2tc const_type;
  bool saw_minus_one = false;
  std::vector<expr2tc> result;
  result.reserve(terms.size());

  for (const auto &term : terms)
  {
    if (!is_constant_int2t(term))
    {
      result.push_back(term);
      continue;
    }

    ++const_count;
    const BigInt &v = to_constant_int2t(term).value;
    if (v == BigInt(-1))
      saw_minus_one = true;
    if (const_count == 1)
    {
      acc = v;
      const_type = term->type;
    }
    else
    {
      auto folded = bitwise_fold(
        acc, v, const_type, [](uint64_t a, uint64_t b) { return a | b; });
      if (!folded)
        return std::nullopt;
      acc = *folded;
    }
  }

  if (const_count < 2)
    return std::nullopt;

  if (saw_minus_one)
  {
    result.clear();
    result.push_back(from_integer(BigInt(-1), const_type));
    return result;
  }

  if (acc.is_zero())
  {
    if (result.empty())
      result.push_back(from_integer(acc, const_type));
    return result;
  }

  result.push_back(from_integer(acc, const_type));
  return result;
}

/// Optimize a flattened bitxor-chain term list.
///
/// Two transforms:
///   1. Constant folding via bitwise XOR. Identity is 0 (drops out).
///      Bitxor has no absorber.
///   2. Self-cancellation: matching pairs (`x ^ x = 0`) annihilate.
///
/// @p chain_type is the type of the original bitxor root, used to type the
/// final zero leaf when total cancellation occurs and no constant was ever
/// seen (so const_type, which only the constant pass sets, is otherwise
/// nil and would produce a malformed expression via from_integer).
std::optional<std::vector<expr2tc>> optimize_bitxor_terms(
  const type2tc &chain_type,
  const std::vector<expr2tc> &terms)
{
  bool changed = false;

  std::size_t const_count = 0;
  BigInt acc(0);
  type2tc const_type;
  std::vector<expr2tc> result;
  result.reserve(terms.size());

  for (const auto &term : terms)
  {
    if (!is_constant_int2t(term))
    {
      result.push_back(term);
      continue;
    }

    ++const_count;
    const BigInt &v = to_constant_int2t(term).value;
    if (const_count == 1)
    {
      acc = v;
      const_type = term->type;
    }
    else
    {
      auto folded = bitwise_fold(
        acc, v, const_type, [](uint64_t a, uint64_t b) { return a ^ b; });
      if (!folded)
        return std::nullopt;
      acc = *folded;
    }
  }

  if (const_count > 1)
  {
    // Two or more constants folded. Drop the result if it's the identity (0).
    if (!acc.is_zero())
      result.push_back(from_integer(acc, const_type));
    changed = true;
  }
  else if (const_count == 1 && acc.is_zero())
  {
    // Lone zero constant — identity, drop it.
    changed = true;
  }
  else if (const_count == 1)
  {
    // Lone non-zero constant — re-emit it.
    result.push_back(from_integer(acc, const_type));
  }

  // Pass 2: cancel matching pairs `x ^ x = 0`. Mark-and-compact to avoid
  // O(n^2) vector shifts. No sign tracking; any two structurally-equal
  // leaves cancel.
  if (result.size() >= 2)
  {
    bool any_cancelled = false;
    std::vector<bool> dead(result.size(), false);
    for (std::size_t i = 0; i < result.size(); ++i)
    {
      if (dead[i])
        continue;
      for (std::size_t j = i + 1; j < result.size(); ++j)
      {
        if (dead[j])
          continue;
        if (!(result[i] == result[j]))
          continue;
        dead[i] = true;
        dead[j] = true;
        any_cancelled = true;
        break;
      }
    }
    if (any_cancelled)
    {
      std::size_t out = 0;
      for (std::size_t i = 0; i < result.size(); ++i)
        if (!dead[i])
          result[out++] = std::move(result[i]);
      result.resize(out);
      changed = true;
    }
  }

  if (!changed)
    return std::nullopt;

  // Empty result after total cancellation: emit a single zero so rebuild
  // produces a well-formed expression. Use const_type when a constant was
  // seen during the fold; otherwise fall back to the chain root type to
  // avoid passing an uninitialized type2tc to from_integer (which would
  // produce a malformed expression).
  if (result.empty())
  {
    const type2tc &zero_type =
      is_nil_type(const_type) ? chain_type : const_type;
    result.push_back(from_integer(BigInt(0), zero_type));
  }
  return result;
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

  // optimize_terms returns Some(replacement) only when a real transform
  // applied. Even then, double-check structural equality before rebuilding
  // — defense in depth against a future transform that produces a
  // logically-identical replacement and breaks the simplifier's
  // "nil iff unchanged" contract.
  std::optional<std::vector<signed_term>> optimized =
    optimize_terms(expr->type, terms);
  if (!optimized || signed_terms_equal(*optimized, terms))
    return false;

  expr2tc rebuilt = rebuild_chain(expr->type, *optimized);
  if (is_nil_expr(rebuilt))
    return false; // rebuild refused (e.g. pointer chain with no base)
  expr = rebuilt;
  return true;
}

bool reassociate_mul(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return false;

  // Only attempt mul reassoc on bv/bool mul roots. Like reassociate_arith
  // we don't descend into operands here — the caller is expr2t::simplify(),
  // which already simplified them bottom-up.
  if (!is_mul2t(expr) || !reassoc_mul_safe_type(expr->type))
    return false;

  std::vector<expr2tc> terms;
  linearize_mul(expr, expr->type, terms);

  if (terms.size() < 2)
    return false;

  std::optional<std::vector<expr2tc>> optimized = optimize_mul_terms(terms);
  if (!optimized || mul_terms_equal(*optimized, terms))
    return false;

  // optimize_mul_terms guarantees a non-empty replacement when it returns a
  // value, so the rebuild produces a well-formed expression.
  expr = rebuild_mul_chain(expr->type, *optimized);
  return true;
}

bool reassociate_bitand(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return false;
  if (!is_bitand2t(expr) || !reassoc_bitwise_safe_type(expr->type))
    return false;

  std::vector<expr2tc> terms;
  linearize_bitand(expr, expr->type, terms);
  if (terms.size() < 2)
    return false;

  std::optional<std::vector<expr2tc>> optimized = optimize_bitand_terms(terms);
  if (!optimized || mul_terms_equal(*optimized, terms))
    return false;

  expr = rebuild_bitand_chain(expr->type, *optimized);
  return true;
}

bool reassociate_bitor(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return false;
  if (!is_bitor2t(expr) || !reassoc_bitwise_safe_type(expr->type))
    return false;

  std::vector<expr2tc> terms;
  linearize_bitor(expr, expr->type, terms);
  if (terms.size() < 2)
    return false;

  std::optional<std::vector<expr2tc>> optimized = optimize_bitor_terms(terms);
  if (!optimized || mul_terms_equal(*optimized, terms))
    return false;

  expr = rebuild_bitor_chain(expr->type, *optimized);
  return true;
}

bool reassociate_bitxor(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return false;
  if (!is_bitxor2t(expr) || !reassoc_bitwise_safe_type(expr->type))
    return false;

  std::vector<expr2tc> terms;
  linearize_bitxor(expr, expr->type, terms);
  if (terms.size() < 2)
    return false;

  std::optional<std::vector<expr2tc>> optimized =
    optimize_bitxor_terms(expr->type, terms);
  if (!optimized || mul_terms_equal(*optimized, terms))
    return false;

  expr = rebuild_bitxor_chain(expr->type, *optimized);
  return true;
}

void simplify_no_reassoc(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;
  expr2tc tmp = expr->simplify(/*suppress_reassoc=*/true);
  if (!is_nil_expr(tmp))
    expr = tmp;
}

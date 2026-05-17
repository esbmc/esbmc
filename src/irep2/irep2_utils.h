#ifndef UTIL_IREP2_UTILS_H_
#define UTIL_IREP2_UTILS_H_

#include <util/c_types.h>

#include <irep2/irep2_expr.h>
#include <util/migrate.h>
#include <util/message.h>

std::string indent_str_irep2(unsigned int indent);

/** Test whether type is an integer. */
inline bool is_bv_type(const type2tc &t)
{
  return t->type_id == type2t::unsignedbv_id ||
         t->type_id == type2t::signedbv_id;
}

inline bool is_bv_type(const expr2tc &e)
{
  return is_bv_type(e->type);
}

/** Test whether type is a float/double. */
inline bool is_fractional_type(const type2tc &t)
{
  return t->type_id == type2t::fixedbv_id || t->type_id == type2t::floatbv_id;
}

/** Test whether @p e is a relational comparison: ==, !=, <, >, <=, >=. */
inline bool is_comparison_expr(const expr2tc &e)
{
  return is_equality2t(e) || is_notequal2t(e) || is_lessthan2t(e) ||
         is_greaterthan2t(e) || is_lessthanequal2t(e) ||
         is_greaterthanequal2t(e);
}

/** Test whether type is a number type - bv, fixedbv or floatbv. */
inline bool is_number_type(const type2tc &t)
{
  return t->type_id == type2t::unsignedbv_id ||
         t->type_id == type2t::signedbv_id ||
         t->type_id == type2t::fixedbv_id || t->type_id == type2t::floatbv_id ||
         t->type_id == type2t::bool_id;
}

inline bool is_number_type(const expr2tc &e)
{
  return is_number_type(e->type);
}

inline bool is_scalar_type(const type2tc &t)
{
  return is_number_type(t) || is_pointer_type(t) || is_empty_type(t) ||
         is_code_type(t);
}

inline bool is_scalar_type(const expr2tc &e)
{
  return is_scalar_type(e->type);
}

inline bool is_multi_dimensional_array(const type2tc &t)
{
  if (is_array_type(t))
    return is_array_type(to_array_type(t).subtype);

  return false;
}

inline bool is_multi_dimensional_array(const expr2tc &e)
{
  return is_multi_dimensional_array(e->type);
}

// array_type2t and vector_type2t carry the same set of element-shape
// fields (subtype, array_size, size_is_infinite). Callers that have a
// type which they know is one of those two but don't care which use
// these helpers to skip the per-kind switch.
inline const type2tc &array_or_vector_subtype(const type2tc &t)
{
  if (is_array_type(t))
    return to_array_type(t).subtype;
  if (is_vector_type(t))
    return to_vector_type(t).subtype;
  irep2_bad_family_cast(t->type_id, "array_or_vector_subtype");
}

inline const expr2tc &array_or_vector_size(const type2tc &t)
{
  if (is_array_type(t))
    return to_array_type(t).array_size;
  if (is_vector_type(t))
    return to_vector_type(t).array_size;
  irep2_bad_family_cast(t->type_id, "array_or_vector_size");
}

inline bool array_or_vector_size_is_infinite(const type2tc &t)
{
  if (is_array_type(t))
    return to_array_type(t).size_is_infinite;
  if (is_vector_type(t))
    return to_vector_type(t).size_is_infinite;
  irep2_bad_family_cast(t->type_id, "array_or_vector_size_is_infinite");
}

inline bool is_byte_type(const type2tc &t)
{
  return is_bv_type(t) && t->get_width() == 8;
}

inline bool is_byte_type(const expr2tc &e)
{
  return is_byte_type(e->type);
}

inline bool is_constant_number(const expr2tc &t)
{
  return t->expr_id == expr2t::constant_int_id ||
         t->expr_id == expr2t::constant_fixedbv_id ||
         t->expr_id == expr2t::constant_floatbv_id ||
         t->expr_id == expr2t::constant_bool_id;
}

inline bool is_constant_expr(const expr2tc &t)
{
  return is_constant_number(t) || t->expr_id == expr2t::constant_string_id ||
         t->expr_id == expr2t::constant_struct_id ||
         t->expr_id == expr2t::constant_union_id ||
         t->expr_id == expr2t::constant_array_id ||
         t->expr_id == expr2t::constant_array_of_id ||
         t->expr_id == expr2t::constant_vector_id;
}

inline bool is_constant(const expr2tc &t)
{
  if (is_pointer_type(t) && is_symbol2t(t))
  {
    symbol2t s = to_symbol2t(t);
    if (s.thename == "NULL")
      return true;
  }
  return is_constant_expr(t);
}

inline bool is_structure_type(const type2tc &t)
{
  return t->type_id == type2t::struct_id || t->type_id == type2t::union_id;
}

inline bool is_structure_type(const expr2tc &e)
{
  return is_structure_type(e->type);
}

inline bool is_arith_expr(const expr2tc &expr)
{
  return expr->expr_id == expr2t::neg_id || expr->expr_id == expr2t::abs_id ||
         expr->expr_id == expr2t::add_id || expr->expr_id == expr2t::sub_id ||
         expr->expr_id == expr2t::mul_id || expr->expr_id == expr2t::div_id ||
         expr->expr_id == expr2t::modulus_id;
}

inline bool is_comp_expr(const expr2tc &expr)
{
  return expr->expr_id == expr2t::lessthan_id ||
         expr->expr_id == expr2t::lessthanequal_id ||
         expr->expr_id == expr2t::greaterthan_id ||
         expr->expr_id == expr2t::greaterthanequal_id ||
         expr->expr_id == expr2t::equality_id ||
         expr->expr_id == expr2t::notequal_id;
}

/** Test if expr is true. First checks whether the expr is a constant bool, and
 *  then whether it's true-valued. If these are both true, return true,
 *  otherwise return false.
 *  @param expr Expression to check for true value.
 *  @return Whether or not expr is true-valued.
 */
inline bool is_true(const expr2tc &expr)
{
  if (is_constant_bool2t(expr) && to_constant_bool2t(expr).value)
    return true;

  if (is_constant_int2t(expr) && !to_constant_int2t(expr).value.is_zero())
    return true;

  if (
    is_constant_floatbv2t(expr) && !to_constant_floatbv2t(expr).value.is_zero())
    return true;

  if (
    is_constant_fixedbv2t(expr) && !to_constant_fixedbv2t(expr).value.is_zero())
    return true;

  return false;
}

/** Test if expr is false. First checks whether the expr is a constant bool, and
 *  then whether it's false-valued. If these are both true, return true,
 *  otherwise return false.
 *  @param expr Expression to check for false value.
 *  @return Whether or not expr is true-valued.
 */
inline bool is_false(const expr2tc &expr)
{
  if (is_constant_bool2t(expr) && !to_constant_bool2t(expr).value)
    return true;

  if (is_constant_int2t(expr) && to_constant_int2t(expr).value.is_zero())
    return true;

  if (
    is_constant_floatbv2t(expr) && to_constant_floatbv2t(expr).value.is_zero())
    return true;

  if (
    is_constant_fixedbv2t(expr) && to_constant_fixedbv2t(expr).value.is_zero())
    return true;

  return false;
}

inline expr2tc gen_true_expr()
{
  static expr2tc c = constant_bool2tc(true);
  return c;
}

inline expr2tc gen_false_expr()
{
  static expr2tc c = constant_bool2tc(false);
  return c;
}

inline expr2tc gen_long(const type2tc &type, BigInt val)
{
  return constant_int2tc(type, std::move(val));
}

inline expr2tc gen_ulong(BigInt v)
{
  return constant_int2tc(get_uint_type(config.ansi_c.word_size), std::move(v));
}

inline expr2tc gen_ulong(unsigned long val)
{
  return gen_ulong(BigInt(val));
}

inline expr2tc gen_slong(signed long val)
{
  return constant_int2tc(get_int_type(config.ansi_c.word_size), BigInt(val));
}

inline const type2tc &get_array_subtype(const type2tc &type)
{
  return to_array_type(type).subtype;
}

inline const type2tc &get_base_array_subtype(const type2tc &type)
{
  const auto &subtype = to_array_type(type).subtype;
  if (is_array_type(subtype))
    return get_base_array_subtype(subtype);

  return subtype;
}

inline bool simplify(expr2tc &expr)
{
  expr2tc tmp = expr->simplify();
  if (!is_nil_expr(tmp))
  {
    expr = tmp;
    return true;
  }

  return false;
}

/** Negate @p expr in place. Folds constant true/false directly to
 *  gen_false_expr() / gen_true_expr() singletons, peels a redundant
 *  `not(not(x))` to `x`, otherwise wraps in `not2tc`. The result is
 *  swapped into @p expr. */
void make_not(expr2tc &expr);

/** Build the left-associative AND-chain of @p cs:
 *  `and(and(and(c0, c1), c2), ...)`. Returns `gen_true_expr()` when
 *  the input is empty (identity for conjunction). */
expr2tc conjunction(std::vector<expr2tc> cs);

/** Build the left-associative OR-chain of @p cs. Returns
 *  `gen_true_expr()` when the input is empty — note this is the
 *  convention used by the existing callers, not the strict identity
 *  for disjunction (which would be `false`). Preserves callers that
 *  short-circuit on "no constraints means trivially satisfied". */
expr2tc disjunction(std::vector<expr2tc> cs);

/** Build a `sideeffect2t` of kind `nondet` of @p type, i.e. an
 *  expression whose value is left to the SMT solver. Used at frontend
 *  / migration boundaries to seed unspecified inputs. */
expr2tc gen_nondet(const type2tc &type);

/** Build the canonical zero value of @p type. Handles bool, BV
 *  (signed/unsigned, any width), fixedbv, floatbv, vector, array,
 *  pointer (NULL), struct (zero-of-each-member) and union (zero-of-
 *  first-member). Recursive for aggregates.
 *
 *  @param array_as_array_of When true, an array_type collapses to a
 *  single `constant_array_of2t` rather than expanding to N elements.
 *  Useful when downstream consumers prefer the run-length form. */
expr2tc gen_zero(const type2tc &type, bool array_as_array_of = false);

/** Build the canonical one value of @p type. Defined for bool, BV
 *  (any width), fixedbv, floatbv. Aborts on aggregate types — "one"
 *  isn't meaningful for structs/arrays/unions. */
expr2tc gen_one(const type2tc &type);

/**
   * @brief Distribute the functor `func` over op1 and op2
   * at least one of those must be a vector
   * 
   * Here, if one of the operands is `nil` then func must
   * support an operation between the vector subtype and nil
   *
   * @param func the functor operation e.g add, sub, mul
   * @param op1 the first operand
   * @param op2 the second operand
   * @return expr2tc with the resulting vector
   */
template <typename Func>
inline expr2tc distribute_vector_operation(Func func, expr2tc op1, expr2tc op2)
{
  assert(is_constant_vector2t(op1) || is_constant_vector2t(op2));
  /*
   * If both op1 and op2 are vectors the resulting value
   * would be the operation over each member
   *
   * Example:
   *
   * op1 = {1,2,3,4}
   * op2 = {1,1,1,1}
   * func = add
   *
   * This would result in:
   *
   * { add(op1[0], op2[0]), add(op1[1], op2[1]), ...}
   * {2,3,4,5}
   */
  if (is_constant_vector2t(op1) && is_constant_vector2t(op2))
  {
    std::vector<expr2tc> members = to_constant_vector2t(op1).datatype_members;
    const constant_vector2t *vec2 = &to_constant_vector2t(op2);
    for (size_t i = 0; i < members.size(); i++)
    {
      auto &A = members[i];
      auto &B = vec2->datatype_members[i];
      auto new_op = func(A->type, A, B);
      members[i] = new_op;
    }
    return constant_vector2tc(op1->type, std::move(members));
  }
  /*
   * If only one of the operator is a vector, then the result
   * would extract each value of the vector and apply the value to
   * the other operator
   *
   * Example:
   *
   * op1 = {1,2,3,4}
   * op2 = 1
   * func = add
   *
   * This would result in:
   *
   * { add(op1[0], 1), add(op1[1], 1), ...}
   * {2,3,4,5}
   */
  else
  {
    bool is_op1_vec = is_constant_vector2t(op1);
    expr2tc c = !is_op1_vec ? op1 : op2;
    expr2tc v = is_op1_vec ? op1 : op2;
    std::vector<expr2tc> members = to_constant_vector2t(v).datatype_members;
    for (auto &datatype_member : members)
    {
      auto &op = datatype_member;
      auto e1 = is_op1_vec ? op : c;
      auto e2 = is_op1_vec ? c : op;
      auto new_op = func(op->type, e1, e2);
      // do_simplify() returns nil when no per-op peephole fires. Don't
      // store nil into the member slot — keep new_op so the lane
      // expression survives unsimplified for the SMT layer.
      auto folded = new_op->do_simplify();
      datatype_member = is_nil_expr(folded) ? new_op : folded;
    }
    return constant_vector2tc(v->type, std::move(members));
  }
}

/**
   * @brief Distribute a function between one or two operands,
   * at least one of those must be a vector
   *
   * @param id the id for the operation
   * @param op1 the first operand
   * @param op2 the second operand
   * @param rm rounding mode (for ieee)
   * @return expr2tc with the resulting vector
   */
/** Element-wise vector operation by `expr_id` dispatch.
 *  At least one of @p op1 / @p op2 must be a vector. For each lane,
 *  builds the corresponding scalar IR node (neg / add / sub / mul /
 *  div / mod / shl / bit{and,or,xor,not} / ieee_{add,sub,mul,div}).
 *  IEEE variants honour the rounding-mode operand @p rm; the others
 *  ignore it. Returns a fresh constant_vector2t holding the lanes.
 *  Unlike the templated overload, this form doesn't take a functor —
 *  it switches on @p id internally, which is convenient at frontend
 *  / migration sites that already have the expr_id at hand. */
expr2tc distribute_vector_operation(
  expr2t::expr_ids id,
  expr2tc op1,
  expr2tc op2 = expr2tc(),
  expr2tc rm = expr2tc());

// Build a comparison-category struct value with the discriminant set to v.
// Used for the C++20 spaceship operator: strong_ordering / weak_ordering /
// partial_ordering all layout the discriminant as the first field, so
// writing operand[0] is enough; remaining fields (if any) are zero-filled
// for layout safety against custom comparison-category types.
//
// Returns nil if @p t is not a struct or has no fields.
expr2tc make_cmp_value(const type2tc &t, int v);

/** Collect every user-level `symbol2t` referenced by @p expr into
 *  @p symbols. Symbols whose name begins with "__ESBMC_" are treated
 *  as compiler internals and skipped (their subtree is not descended).
 *  Recursive — visits the expression tree via foreach_operand. */
void get_symbols(
  const expr2tc &expr,
  std::unordered_set<expr2tc, irep2_hash> &symbols);

#endif /* UTIL_IREP2_UTILS_H_ */

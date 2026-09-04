#ifndef _ESBMC_PROP_SMT_SMT_SOLVER_H_
#define _ESBMC_PROP_SMT_SMT_SOLVER_H_

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <solvers/pointer_logic.h>
#include <solvers/smt_result.h>
#include <solvers/smt_sort.h>
#include <irep2/irep2_utils.h>
#include <util/message/message.h>
#include <util/symtab/namespace.h>
#include <util/base/threeval.h>

/** @file smt_solver.h
 *  Converting an SSA program into SMT.
 *
 *  An SSA program held by symex_target_equationt becomes a series of boolean
 *  propositions in a solver context; those are asserted, the solver is asked
 *  whether the conjunction is satisfiable, and if it is, the values of symbols
 *  can be read back to build a counterexample.
 *
 *  ESBMC's expressions (irep2) follow no particular formalism, so reaching a
 *  decidable logic takes a lot of translation -- what Kroening calls
 *  'flattening' in CBMC. smt_solver_baset does that work and encodes the result
 *  through camada, which owns the solver-specific part: expressions and sorts
 *  are camada handles (see smt_sort.h), and camada uses each solver's own
 *  theories where it has them and lowers where it does not.
 *
 *  Flattened here, for every backend:
 *    * The C memory address space
 *    * Representation of pointer types
 *    * Casts
 *    * Byte operations on objects (extract/update)
 *    * FixedBV representation of floats
 *    * Unions -> something else
 *    * Bit-vector integer overflow detection
 *
 *  Lowered by camada rather than here, natively supported or not:
 *    * Tuples (and arrays of them) -- TupleEncoding
 *    * Arrays -- ArrayEncoding, either the solver's theory or Ackermann
 *      congruence axioms
 *    * Floating-point -- FPEncoding, native or bit-blasted (--fp2bv)
 *
 *  If you find yourself making the SMT translation do more than this, ask
 *  whether it belongs at another layer, such as symbolic execution:
 *    * Anything involving pointer dereferencing at all
 *    * Anything that considers the control flow guard at any point
 *    * Pointer liveness or dynamic allocation consideration
 *
 *  @see smt_solver_baset
 *  @see symex_target_equationt
 *  @see create_solver
 */

// Forward dec.
class ir_ieee_convt;
class smt_solver_baset;
// True iff the given type lowers to a tuple sort in the SMT layer:
// struct (incl. C++ class data), pointer (as the (object, offset)
// pair lowered via pointer_struct), code (function pointer payload),
// or complex (the (real, imag) pair).
inline bool is_tuple_ast_type(const type2tc &t)
{
  return is_struct_type(t) || is_pointer_type(t) || is_code_type(t) ||
         is_complex_type(t);
}

inline bool is_tuple_ast_type(const expr2tc &e)
{
  return is_tuple_ast_type(e->type);
}

inline bool is_tuple_array_ast_type(const type2tc &t)
{
  if (!is_array_type(t))
    return false;

  const array_type2t &arr_type = to_array_type(t);
  type2tc range = arr_type.subtype;
  while (is_array_type(range))
    range = to_array_type(range).subtype;

  return is_tuple_ast_type(range);
}

/** Converts irep2 expressions into camada SMT expressions.
 *
 *  convert_ast() is the entry point: it deconstructs an expression, encodes the
 *  pieces through camada, and caches the result against the expression and the
 *  current context level. Boolean results are asserted with assert_ast().
 *
 *  One concrete class, no subclasses -- camada supplies the per-solver part, so
 *  create_solver() hands the constructor a built camada solver rather than
 *  selecting behaviour by inheritance.
 *
 *  push_ctx()/pop_ctx() bracket a solver scope. They save and restore what
 *  camada cannot know about: the caches keyed on irep2 expressions, and the C
 *  memory model (address space, pointer logic, renumbering).
 *
 *  @see create_solver
 */
class smt_solver_baset
{
  /* Holds a back-pointer to us and encodes through the same camada solver. */
  friend class ir_ieee_convt;

public:
  /** Shorthand for a vector of SMT expressions */
  typedef std::vector<smt_astt> ast_vec;

  /** Primary constructor. After construction, smt_post_init must be called
   *  before the object is used as a solver converter.
   *
   *  @param _ns Namespace for looking up the type of certain symbols.
   *  @param _options Provide all the needed parameters to configure the solver.
   *  @param _solver The camada solver to encode into. Each create_new_*_solver
   *         builds the one it wants, so there is no backend tag to switch on.
   *  @param _streams_script Camada's SMT-LIB backend writes the script to its
   *         sink as it is built rather than buffering it, so dump_smt() has
   *         nothing to hand back. This is a property of which camada solver was
   *         built, not of the options, so it cannot be derived here.
   */
  smt_solver_baset(
    const namespacet &_ns,
    const optionst &_options,
    std::unique_ptr<camada::SMTSolver> _solver,
    bool _streams_script = false);

  ~smt_solver_baset();

  /** Post-constructor setup method. We must create various pieces of memory
   *  model data for tracking, however can't do it from the constructor because
   *  the solver converter itself won't have been initialized itself at that
   *  point. So, once it's ready, the solver converter should call this from
   *  it's constructor. */
  void smt_post_init();

  // The API that we provide to the rest of the world:
  /** @{
   *  @name External API to smt_solver_baset. */

  /** Push one context on the SMT assertion stack. */
  void push_ctx();
  /** Pop one context on the SMT assertion stack. */
  void pop_ctx();

  /** Whether a satisfiable result can be turned into a model. False for the
   *  subprocess SMT-LIB backends with no interactive model solver attached:
   *  they answer sat/unsat, but get() and l_get() have nothing to read.
   *  Defined in camada_conv.cpp, where the backend's shape is known. */
  bool has_model() const;

  /** Main interface to SMT conversion.
   *  Takes one expression, and converts it into the underlying SMT solver,
   *  returning a single smt_ast that represents the converted expressions
   *  value. The lifetime of the returned pointer is currently undefined.
   *
   *  @param expr The expression to convert into the SMT solver
   *  @return The resulting handle to the SMT value. */
  smt_astt convert_ast(const expr2tc &expr);

  /** Convert a single node, assuming its operands are already in the cache.
   *  This is the per-node body of the conversion; convert_ast() drives it
   *  with an explicit-stack post-order walk so that deeply left-nested
   *  associative chains (e.g. a huge disjunction) don't recurse one C++
   *  frame per level and overflow the stack. The operand walk inside this
   *  body therefore hits the cache instead of recursing. */
  smt_astt convert_ast_node(const expr2tc &expr);

  /** Convert one of the two-operand IEEE arithmetic nodes (add/sub/mul/div/
   *  rem), which share an operand layout and differ only in the solver call
   *  they end up in. */
  smt_astt convert_ieee_arith_2op(const expr2tc &expr);

  /** Interface to specifig SMT conversion.
   *  Takes one expression, and converts it into the underlying SMT solver,
   *  depending on the type of the expression.
   *
   *  @param expr The expression to convert into the SMT solver
   *  @param type The expression's type
   *  @param args The expression's args
   *  @param ops The operations for each sort type
   *  @return The resulting handle to the SMT value. */
  smt_astt convert_ast(
    const expr2tc &expr,
    const type2tc &type,
    smt_astt const *args,
    struct expr_op_convert ops);

  smt_astt convert_assign(const expr2tc &expr);

  /** Convert @p expr and dump the resulting SMT AST in SMT format. Used by
   *  --ssa-smt-trace so callers outside the solver layer can request the
   *  dump without ever touching smt_astt. */
  void dump_expr(const expr2tc &expr);

  /** Create the inverse of an smt_ast. Equivalent to a 'not' operation.
   *  @param a The ast to invert. Must be boolean sorted.
   *  @return The inverted piece of AST. */
  smt_astt invert_ast(smt_astt a);

  /** Create an ipmlication between two smt_ast's.
   *  @param a The ast that implies the truth of the other. Boolean.
   *  @param b The ast whos truth is implied. Boolean.
   *  @return The resulting piece of AST. */
  smt_astt imply_ast(smt_astt a, smt_astt b);

  /** Assert the truth of an ast. Equivalent to the 'assert' directive in the
   *  SMTLIB language, this informs the solver that in the satisfying
   *  assignment it attempts to produce, the formula corresponding to the
   *  smt_ast argument must evaluate to true.
   *  @param a A handle to the formula that must be true. */
  void assert_ast(smt_astt a);

  /** Solve the formula given to the solver. The solver will attempt to produce
   *  a satisfying assignment for all of the variables / symbols used in the
   *  formula, where all the asserted sub-formula are true. Results are either
   *  unsat (the formula is inconsistent), sat (an assignment exists), or that
   *  an error occurred.
   *  @return Result code of the call to the solver. */
  smt_resultt dec_solve();

  void pre_solve();

  /** Get the satisfying assignment using the ast.
   *  @param a Variable to get the value of.
   *  @param type The variable type.
   *  @return Explicit assigned value of expr in the solver. May be nil, in
   *          which case the solver did not assign a value to it for some
   *          reason. */
  expr2tc get_by_ast(const type2tc &type, smt_astt a);

  /** Builds the bitvector based on the value retrieved from the solver.
   *  @param type the type (fixedbv or (un)signedbv),
   *  @param value the value retrieved from the solver.
   *  @return Expression representation of a's value */
  expr2tc get_by_value(const type2tc &type, BigInt value);

  /** Extract the assignment to a rational/real value from the SMT solvers
   * model. Used in integer/real arithmetic mode to get floating point values.
   *  @param a The AST whose value we wish to know.
   *  @param numerator Output parameter for the numerator of the rational.
   *  @param denominator Output parameter for the denominator of the rational.
   *  @return True if the rational value was successfully extracted, false
   * otherwise. */
  bool get_rational(smt_astt a, BigInt &numerator, BigInt &denominator);
  /** Fetch a satisfying assignment from the solver. If a previous call to
   *  dec_solve returned satisfiable, then the solver has a set of assignments
   *  to symbols / variables used in the formula. This method retrieves the
   *  value of a symbol, and formats it into an ESBMC expression.
   *  @param expr Variable to get the value of. Must be a symbol expression.
   *  @return Explicit assigned value of expr in the solver. May be nil, in
   *          which case the solver did not assign a value to it for some
   *          reason. */
  expr2tc get(const expr2tc &expr);

  /** Solver name fetcher. Returns a string naming the solver being used, and
   *  potentially it's version, if available.
   *  @return The name of the solver this smt_solver_baset uses. */
  const std::string solver_text();

  /** Fetch the value of a boolean sorted smt_ast. (The 'l' is for literal, and
   *  is historic). Returns a three valued result, of true, false, or
   *  unassigned.
   *  @param a The boolean sorted ast to fetch the value of.
   *  @return A three-valued return val, of the assignment to a. */
  tvt l_get(smt_astt a);

  /** Fetch the value of a boolean expression from the current model. */
  tvt l_get(const expr2tc &expr);

  /** @} */

  /** @{
   *  @name Internal conversion API between smt_solver_baset and solver
   * converter */

  smt_astt mk_add(smt_astt a, smt_astt b);
  smt_astt mk_sub(smt_astt a, smt_astt b);
  smt_astt mk_mul(smt_astt a, smt_astt b);
  smt_astt mk_mod(smt_astt a, smt_astt b);
  smt_astt mk_bvumod(smt_astt a, smt_astt b);
  smt_astt mk_div(smt_astt a, smt_astt b);
  smt_astt mk_bvsdiv(smt_astt a, smt_astt b);
  smt_astt mk_shl(smt_astt a, smt_astt b);
  smt_astt mk_bvashr(smt_astt a, smt_astt b);
  smt_astt mk_neg(smt_astt a);
  smt_astt mk_bvnot(smt_astt a);
  smt_astt mk_bvxor(smt_astt a, smt_astt b);
  smt_astt mk_bvor(smt_astt a, smt_astt b);
  smt_astt mk_bvand(smt_astt a, smt_astt b);
  smt_astt mk_implies(smt_astt a, smt_astt b);
  smt_astt mk_or(smt_astt a, smt_astt b);
  smt_astt mk_not(smt_astt a);
  smt_astt mk_lt(smt_astt a, smt_astt b);
  smt_astt mk_bvult(smt_astt a, smt_astt b);
  smt_astt mk_gt(smt_astt a, smt_astt b);
  smt_astt mk_bvsgt(smt_astt a, smt_astt b);
  smt_astt mk_le(smt_astt a, smt_astt b);
  smt_astt mk_bvule(smt_astt a, smt_astt b);
  smt_astt mk_ge(smt_astt a, smt_astt b);
  smt_astt mk_bvsge(smt_astt a, smt_astt b);
  smt_astt mk_neq(smt_astt a, smt_astt b);

  /* mk_eq/mk_int_sort/mk_smt_symbol are the surface the unit tests drive from
   * outside the class, where `solver` is private. */
  smt_astt mk_eq(smt_astt a, smt_astt b);
  smt_sortt mk_int_sort();
  smt_astt mk_smt_symbol(const std::string &name, smt_sortt s);
  smt_astt mk_select(smt_astt a, smt_astt b);
  smt_astt mk_int2real(smt_astt a);

  /** @{
   *  @name Sort-directed operations.
   *  These dispatch on the operand's sort rather than on a C++ type: tuples and
   *  arrays need different encodings for the same source-level operation. They
   *  were virtuals on smt_ast until that wrapper was replaced by camada's
   *  expression handle. */

  /** Equality that does the right thing for scalars, tuples and arrays. */
  smt_astt ast_eq(smt_astt a, smt_astt b);

  /** Assign @p value to @p sym. Defaults to asserting an equality. */
  void ast_assign(smt_astt value, smt_astt sym);

  /** An array 'with' or a tuple 'with'.
   *  @param idx Array index or tuple field number.
   *  @param idx_expr For arrays, the expression giving the index. */
  smt_astt ast_update(
    smt_astt a,
    smt_astt value,
    unsigned int idx,
    const expr2tc &idx_expr = expr2tc());

  /** Select from an array, whether of scalars or of tuples. */
  smt_astt ast_select(smt_astt a, const expr2tc &idx);

  /** Project a member from a struct, or a field-array from a struct array.
   *  Only meaningful for tuple sorts, so the base implementation errors. */
  smt_astt ast_project(smt_astt a, unsigned int elem);

  /** @} */
  smt_astt
  mk_quantifier(bool is_forall, std::vector<smt_astt> lhs, smt_astt rhs);

  smt_astt convert_concat_int_mode(
    smt_astt left_ast,
    smt_astt right_ast,
    const expr2tc &expr);

  /** Create an integer or SBV/UBV sort */
  smt_sortt mk_int_bv_sort(std::size_t width)
  {
    if (int_encoding)
      return solver->mkIntSort();

    if (width == 0)
      width = 1;

    return solver->mkBVSort(width);
  }

  /** Create an real or floating-point sort */
  smt_sortt mk_real_fp_sort(std::size_t ew, std::size_t sw)
  {
    if (int_encoding)
      return solver->mkRealSort();

    return mk_fpbv_sort(ew, sw);
  }

  /** Create a floating-point sort, using bitvectors */
  smt_sortt mk_bvfp_sort(std::size_t ew, std::size_t sw);

  /** Create a floating-point rounding mode sort, using bitvectors */
  smt_sortt mk_bvfp_rm_sort();

  /** Create an integer smt_ast. That is, an integer in QF_AUFLIRA, rather than
   *  a bitvector.
   *  @param theint BigInt representation of the number to create.
   *  @return The newly created terminal smt_ast of this integer. */
  smt_astt mk_smt_int(const BigInt &theint);

  // Returns SMT AST representing real zero
  smt_astt get_zero_real();
  // Returns SMT AST representing double precision minimum normal value
  // (2^-1022)
  smt_astt get_double_min_normal();
  // Returns SMT AST representing double precision minimum subnormal value
  // (2^-1074)
  smt_astt get_double_min_subnormal();
  // Returns SMT AST representing double precision maximum normal value
  // (~1.7976931348623157e+308)
  smt_astt get_double_max_normal();
  // Returns SMT AST representing single precision minimum normal value (2^-126)
  smt_astt get_single_min_normal();
  // Returns SMT AST representing single precision minimum subnormal value
  // (2^-149)
  smt_astt get_single_min_subnormal();
  // Returns SMT AST representing single precision maximum normal value
  // (~3.4028234663852886e+38)
  smt_astt get_single_max_normal();
  // Under --ir-ieee, returns real zero when r lies in the region that
  // rounds to zero under the selected rounding mode; otherwise returns r
  // unchanged. This models only the zero/nonzero underflow boundary:
  // signed zero and subnormal-grid quantization are not represented.
  // Returns r unchanged for unsupported float formats.
  smt_astt mk_subnormal_flush(
    smt_astt r,
    const floatbv_type2t &fbv_type,
    const expr2tc &rounding_mode);

  // Returns SMT AST for the integer-encoding sentinel for double +∞:
  // max_normal+1
  smt_astt get_double_inf_sentinel();
  // Returns SMT AST for the integer-encoding sentinel for single +∞:
  // max_normal+1
  smt_astt get_single_inf_sentinel();
  // Returns SMT AST for the double precision relative error bound under
  // round-to-nearest: half machine epsilon = 2^-53 ~ 1.11e-16
  smt_astt get_double_eps_rel();
  // Returns SMT AST for the single precision relative error bound under
  // round-to-nearest: half machine epsilon = 2^-24 ~ 5.96e-08
  smt_astt get_single_eps_rel();
  // Returns SMT AST for the double precision directional error bound under
  // round-toward-+inf: full machine epsilon = 2^-52 ~ 2.22e-16
  smt_astt get_double_eps_up();
  // Returns SMT AST for the single precision directional error bound under
  // round-toward-+inf: full machine epsilon = 2^-23 ~ 1.19e-07
  smt_astt get_single_eps_up();

  /** Create a bitvector.
   *  @param theint Integer representation of the bitvector. Any excess bits
   *         in the stored integer should be ignored.
   *  @param w Width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  smt_astt mk_smt_bv(const BigInt &theint, std::size_t w)
  {
    return mk_smt_bv(theint, mk_int_bv_sort(w));
  }

  /** Create a bitvector.
   *  @param s the sort.
   *  @param theint Integer representation of the bitvector. Any excess bits
   *         in the stored integer should be ignored.
   *  @return The newly created terminal smt_ast of this bitvector. */
  smt_astt mk_smt_bv(const BigInt &theint, smt_sortt s);

  /** Apply an uninterpreted function. Declares (or reuses) a solver function
   *  symbol named @p name whose domain is the sorts of @p args and whose range
   *  is @p rangesort, then applies it to @p args. Every application sharing
   *  @p name resolves to the same declaration, so the solver enforces
   *  functional congruence (equal arguments imply an equal result) natively.
   *
   *  The default implementation Ackermannises the application (a fresh symbol
   *  plus congruence assumptions against earlier applications of the same name)
   *  for backends without native uninterpreted-function support. Solvers that
   *  expose UFs override this to declare and apply them directly.
   *  @param name Stable (mangled) name of the function symbol.
   *  @param args The already-converted argument asts.
   *  @param rangesort The sort of the function's result.
   *  @return The ast denoting name(args). */
  smt_astt mk_smt_uninterpreted_function(
    const std::string &name,
    const std::vector<smt_astt> &args,
    smt_sortt rangesort);

  /** Create an 'extract' func app. Since we can't currently
   *  encode integer constants as function arguments without serious faff,
   *  this can't be performed via the medium of mk_func_app. Hence, this api
   *  call.
   *  @param a The source piece of ast to extract a value from.
   *  @param high The topmost bit to select from the source, down to low.
   *  @param low The lowest bit to select from the source. */
  smt_astt mk_extract(smt_astt a, unsigned int high, unsigned int low);

  /** Given a signed, upwards cast, extends the sign of the given AST to the
   *  desired length.
   *  @param a The bitvector to upcast.
   *  @param topwidth The number of bits to extend the input by
   *  @return A bitvector with topwidth more bits, of the appropriate sign. */
  smt_astt mk_sign_ext(smt_astt a, unsigned int topwidth);

  /** Identical to mk_sign_ext, but extends AST with zeros */
  smt_astt mk_zero_ext(smt_astt a, unsigned int topwidth);

  /** Extract the assignment to a boolean variable from the SMT solver's model.
   *  @param a The AST whose value we wish to know.
   *  @return a's value, or tvt::TV_UNKNOWN when the solver cannot reduce it to
   *          a ground boolean. That is an expected outcome for terms that still
   *          contain a quantifier: callers must not invent a value for it. */
  tvt get_bool(smt_astt a);

  /** Extract the assignment to a bitvector from the SMT solver's model.
   *  @param a The AST whose value we wish to know.
   *  @param is_signed whether the bitvector is signed
   *  @return Expression representation of a's value */
  BigInt get_bv(smt_astt a, bool is_signed);

  /** Extract a fixed-point assignment as its raw scaled value. */
  BigInt get_fxp(smt_astt a);

  /** Reduction or: equals bit0 iff all bits are 0
   * @param op the expr to be reduced
   * @return reduced op
   */
  smt_astt mk_bvredor(smt_astt op);

  /** Reduction and: equals bit1 iff all bits are 1
   * @param op the expr to be reduced
   * @return reduced op
   */
  smt_astt mk_bvredand(smt_astt op);

  /** @} */

  /** @{
   *  @name Integer overflow solver-converter API. */

  /** Detect integer arithmetic overflows. Takes an expression that is one of
   *  add / sub / mul / div / modulus / shl, and evaluates whether its
   *  operation applied to its operands will result in an integer overflow or
   *  underflow.
   *  @param expr Expression to test for arithmetic overflows in.
   *  @return Boolean valued AST representing whether an overflow occurs. */
  smt_astt overflow_arith(const expr2tc &expr);

  /** Detect integer overflows in a cast. Takes a typecast2tc as an argument,
   *  and if it causes a decrease in integer width, then encodes a test that
   *  the dropped bits are never significant / used.
   *  @param expr Cast to test for dropped / overflowed data in.
   *  @return Boolean valued AST representing whether an overflow occurs. */
  smt_astt overflow_cast(const expr2tc &expr);

  /** Detects integer overflows in negation. This only tests for the case where
   *  MIN_INT is being negated, in which case there is no positive
   *  representation of that number, and an overflow occurs. Evaluates to true
   *  if that can occur in the operand.
   *  @param expr A neg2tc to test for overflows in.
   *  @return Boolean valued AST representing whether an overflow occurs. */
  smt_astt overflow_neg(const expr2tc &expr);

  /** Applies IEEE 754 floating-point semantics to a real arithmetic result.
   *  Handles overflow, underflow, and subnormal number behaviors that are
   *  missing when using integer/real encoding for floating-point operations.
   *  Supports both IEEE 754 single precision (32-bit: 8 exponent, 23 fraction)
   *  and double precision (64-bit: 11 exponent, 52 fraction) formats.
   *  For double precision: overflow to ±1.798e+308, underflow below 4.941e-324,
   *  subnormal range [4.941e-324, 2.225e-308). For single precision: overflow
   *  to ±3.403e+38, underflow below 1.401e-45, subnormal range
   * [1.401e-45, 1.175e-38). Other formats return the original result unchanged.
   *  Under --ir-ieee, when rounding_mode is a concrete round-to-nearest
   * constant (ROUND_TO_EVEN == 0), a tight symmetric epsilon enclosure is
   * asserted. For symbolic or directed rounding modes the function falls back
   * to a weak unconstrained enclosure (sound but imprecise); tight directed
   * bounds are deferred to a future PR.
   *  @param real_result The result of exact real arithmetic operation
   *  @param fbv_type The floating-point type information (exponent/fraction
   * bits)
   *  @param operand_zero_check Optional boolean AST for special zero handling
   *         (e.g., multiplication where either operand is zero should yield
   * zero regardless of the other operand, even if it would cause underflow)
   *  @param rounding_mode The rounding mode expr2tc from the IR operation node;
   *         typically a constant_int2t or the __ESBMC_rounding_mode symbol.
   *  @return SMT AST representing the result with IEEE 754 semantics applied */
  smt_astt apply_ieee754_semantics(
    smt_astt real_result,
    const floatbv_type2t &fbv_type,
    smt_astt operand_zero_check = {},
    const expr2tc &rounding_mode = expr2tc());

  /** Method to dump the SMT formula */
  std::string dump_smt();

  /** Method to print the SMT model */
  void print_model();

  /** @} */

  /** @{
   *  @name Array operations solver-converter API. */

  /** High level index expression conversion. Deals with several annoying
   *  corner cases that must be addressed, such as flattening multidimensional
   *  arrays into one domain sort, or turning bool arrays into bit arrays.
   *  XXX, why is this virtual?
   *  @param expr An index2tc expression to convert to an SMT AST.
   *  @return An AST representing the index operation in the expression. */
  smt_astt convert_array_index(const expr2tc &expr);

  /** Partner method to convert_array_index, for stores.
   *  XXX, why is this virtual?
   *  @param expr with2tc operation to convert to SMT.
   *  @return AST representing the result of evaluating expr. */
  smt_astt convert_array_store(const expr2tc &expr);

  /** @} */

  /** @{
   *  @name Internal foo. */

  /** Create a free variable with the given sort, and a unique name, with the
   *  prefix given in 'tag' */
  smt_astt mk_fresh(smt_sortt s, const std::string &tag, smt_sortt st = {});
  /** Create a previously un-used variable name with the prefix given in tag */
  std::string mk_fresh_name(const std::string &tag);

  void renumber_symbol_address(
    const expr2tc &guard,
    const expr2tc &addr_symbol,
    const expr2tc &new_size);

  /** Convert a type2tc into an smt_sort. This dispatches control to the
   *  appropriate method in the subclassing solver converter for type
   *  conversion */
  smt_sortt convert_sort(const type2tc &type);
  /** Convert a terminal expression into an SMT AST. This dispatches control to
   *  the appropriate method in the subclassing solver converter for the
   * terminal conversion */
  smt_astt convert_terminal(const expr2tc &expr);

  /** Flatten pointer arithmetic. When faced with an addition or subtraction
   *  between a pointer and some integer or other pointer, perform whatever
   *  multiplications or casting is required to honor the C semantics of
   *  pointer arith. */
  smt_astt convert_pointer_arith(const expr2tc &expr, const type2tc &t);
  /** Compare two pointers. This attempts to optimize cases where we can avoid
   *  comparing the integer representation of a pointer, as that's hugely
   *  inefficient sometimes (and gets bitblasted).
   *  @param expr First pointer to compare
   *  @param expr2 Second pointer to compare
   *  @param templ_expr The comparision expression -- this method will look at
   *         the kind of comparison being performed, and make an appropriate
   *         decision.
   *  @return Boolean valued AST as appropriate to the requested comparision */
  smt_astt convert_ptr_cmp(
    const expr2tc &expr,
    const expr2tc &expr2,
    const expr2tc &templ_expr);
  /** Take the address of some kind of expression. This will abort if the given
   *  expression isn't based on some symbol in some way. (i.e., you can't take
   *  the address of an addition, but you can take the address of a member of
   *  a struct, for example). */
  smt_astt convert_addr_of(const expr2tc &expr);
  /** Handle union/struct based corner cases for member2tc expressions */
  smt_astt convert_member(const expr2tc &expr);
  /** Convert an identifier to a pointer. When given the name of a variable
   *  that we want to take the address of, this inspects our current tracking
   *  of addresses / variables, and returns a pointer for the given symbol.
   *  If it hasn't had its address taken before, it performs any computations or
   *  address space juggling required to make a new pointer.
   *  @param expr The symbol2tc expression of this symbol.
   *  @param sym The textual representation of this symbol.
   *  @param type Optionally a pointer to the type of the symbol in the context.
   *  @return A pointer-typed AST representing the address of this symbol. */
  smt_astt convert_identifier_pointer(
    const expr2tc &expr,
    const std::string &sym,
    const typet *type);

  smt_astt init_pointer_obj(
    unsigned int obj_num,
    const expr2tc &size,
    const typet *type);

  /** Checks for equality with NaN representation. */
  smt_astt convert_is_nan(const expr2tc &expr);
  /** Checks for equality with inf representation. */
  smt_astt convert_is_inf(const expr2tc &expr);
  /** Checks for equality with normal representation. */
  smt_astt convert_is_normal(const expr2tc &expr);
  /** Checks for equality with finite representation. */
  smt_astt convert_is_finite(const expr2tc &expr);
  /** Converts signbit representation. */
  smt_astt convert_signbit(const expr2tc &expr);
  /** Converts popcount representation. */
  smt_astt convert_popcount(const expr2tc &expr);
  /** Converts bswap representation. */
  smt_astt convert_bswap(const expr2tc &expr);
  /** Converts rounding mode for ieee fp operations. */
  smt_astt convert_rounding_mode(const expr2tc &expr);
  /** Convert a byte_extract2tc, pulling a byte from the byte representation
   *  of some piece of data. */
  smt_astt convert_byte_extract(const expr2tc &expr);
  /* Integer mode byte extraction helper functions */
  smt_astt convert_byte_extract_int_mode(
    const byte_extract2t &data,
    expr2tc source,
    expr2tc offs,
    unsigned int src_width);
  /* Bit-vector mode byte extraction helper functions */
  smt_astt convert_byte_extract_bv_mode(
    const byte_extract2t &data,
    expr2tc source,
    expr2tc offs,
    unsigned int src_width);
  /* Helper function for integer arithmetic right shift simulation */
  expr2tc create_int_right_shift(expr2tc source, expr2tc shift_amount);
  /** Convert a byte_update2tc, inserting a byte into the byte representation
   *  of some piece of data. */
  smt_astt convert_byte_update(const expr2tc &expr);
  /** Convert a byte_update2tc in integer arithmetic mode, handling both
   *  constant and non-constant offsets with proper type preservation. */
  smt_astt convert_byte_update_int_mode(const byte_update2t &data);
  /** Convert a byte_update2tc with offset in integer mode,
   *  using conditional expressions for all possible byte positions. */
  expr2tc convert_byte_update_int_mode_expr(
    const byte_update2t &data,
    expr2tc source,
    expr2tc offs,
    expr2tc update_value,
    unsigned int src_width);
  /** Convert a byte_update2tc using bitvector operations, preserving
   *  the original bitvector-based implementation. */
  smt_astt convert_byte_update_bv_mode(const byte_update2t &data);
  /** Convert a bitcast2tc, converting an expr to its bit representation. */
  smt_astt convert_bitcast(const expr2tc &expr);
  /** Convert the given expr to AST, then assert that AST */
  void assert_expr(const expr2tc &e);
  /** Record every division's operand pair in @p expr, recursively.
   *  convert_modulus lowers a remainder compositionally only when its
   *  operands appear here; unconditional lowering costs 3-5x on
   *  rem-heavy proofs. */
  void note_division_operands(const expr2tc &expr);
  /** Encode a remainder: compositional via the matching division when
   *  one exists in the formula, the solver's rem primitive otherwise. */
  smt_astt convert_modulus(const modulus2t &m, smt_astt a, smt_astt b);
  /** Convert constant_array2tc's and constant_array_of2tc's */
  smt_astt array_create(const expr2tc &expr);

  /** Initialize tracking data for the address space records. This also sets
   *  up the symbols / addresses of 'NULL', '0', and the invalid pointer */
  void init_addr_space_array();
  /** Stores handle for the tuple interface. */
  /* ---- Tuples ----
   * Implemented by the backend: camada uses the solver's datatypes where it
   * has them and lowers to per-field symbols where it does not, so there is
   * no separate flattener to install. */
  /** Create a sort representing a struct. i.e., a tuple. Ideally this should
   *  actually be part of the overridden tuple api, but due to history it isn't
   *  yet. If solvers don't support tuples, implement this to abort.
   *  @param type The struct type to create a tuple representation of.
   *  @return The tuple representation of the type, wrapped in an smt_sort. */
  smt_sortt mk_struct_sort(const type2tc &type);

  /** Create a new tuple from a struct definition.
   *  @param structdef A constant_struct2tc, describing all the members of the
   *         tuple to create.
   *  @return AST representing the created tuple */
  smt_astt tuple_create(const expr2tc &structdef);

  /** Create a fresh tuple, with freely valued fields.
   *  @param s Sort of the tuple to create
   *  @return AST representing the created tuple */
  smt_astt tuple_fresh(smt_sortt s, std::string name = "");

  // XXX XXX XXX docs gap
  smt_astt tuple_array_create(
    const type2tc &array_type,
    smt_astt *inputargs,
    bool const_array,
    smt_sortt domain);

  /** Create a potentially /large/ array of tuples. This is called when we
   *  encounter an array_of operation, with a very large array size, of tuple
   *  sort.
   *  @param Expression of tuple value to populate this array with.
   *  @param domain_width The size of array to create, in domain bits.
   *  @return An AST representing an array of the tuple value, init_value. */
  smt_astt
  tuple_array_of(const expr2tc &init_value, unsigned long domain_width);

  /** Convert a symbol to a tuple_smt_ast */
  smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s);

  /** Like mk_tuple_symbol, but for arrays */
  smt_astt mk_tuple_array_symbol(const expr2tc &expr);

  /** Extract the assignment to a tuple-typed symbol from the SMT solvers
   *  model */
  expr2tc tuple_get(const expr2tc &expr);
  expr2tc tuple_get(const type2tc &type, smt_astt a);

  expr2tc
  tuple_get_array_elem(smt_astt array, uint64_t index, const type2tc &subtype);

  void add_tuple_constraints_for_solving()
  {
  }
  /** Stores handle for the array interface. */
  /* ---- Arrays ----
   * Implemented by the backend: camada uses the solver's theory of arrays
   * where it has one and lowers to Ackermann congruence axioms where it does
   * not, so there is no separate flattener to install. */
  smt_astt
  mk_array_symbol(const std::string &name, smt_sortt sort, smt_sortt subtype);

  /** Extract an element from the model of an array, at an explicit index.
   *  @param array AST representing the array we are extracting from
   *  @param index The index of the element we wish to expect
   *  @param subtype The type of the element we are extracting, i.e., array
   * range
   *  @return Expression representation of the element */
  expr2tc get_array_elem(smt_astt a, uint64_t idx, const type2tc &subtype);

  /** Create an array with a single initializer. This may be a small, fixed
   *  size array, or it may be a nondeterministically sized array with a
   *  word-sized domain. Default implementation is to repeatedly store into
   *  the array for as many elements as necessary; subclassing class should
   *  override if it has a more efficient method.
   *  Nondeterministically sized memory with an initializer is very rare;
   *  the only real users of this are fixed-sized (but large) static arrays
   *  that are zero initialized, or some infinite-domain modelling arrays
   *  used in ESBMC.
   *  @param init_val The value to initialize each element with.
   *  @param domain_width The size of the array to create, in domain bits.
   *  @return An AST representing the created constant array. */
  smt_astt convert_array_of(smt_astt init_val, unsigned long domain_width);

  void add_array_constraints_for_solving()
  {
  }
  /** Stores handle for the floating-point interface. */
  /* ---- Floating-point ----
   * Implemented by the backend: camada encodes floating-point natively or
   * bit-blasts it (FPEncoding::BV, selected by --fp2bv), so there is no
   * separate lowering object to install. */

  /** Create a floating point bitvector
   *  @param thereal the floating-point number
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  smt_astt mk_smt_fpbv(const ieee_floatt &thereal);

  /** Create a sort representing a floating-point number.
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The floating-point representation of the type, wrapped in an
   * smt_sort. */
  smt_sortt mk_fpbv_sort(const unsigned ew, const unsigned sw);

  /** Create a sort representing a floating-point rounding mode.
   *  @return The floating-point rounding mode representation of the type,
   *  wrapped in an smt_sort. */
  smt_sortt mk_fpbv_rm_sort();

  /** Create a NaN floating point bitvector
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  smt_astt mk_smt_fpbv_nan(bool sgn, unsigned ew, unsigned sw);

  /** Create a (+/-)inf floating point bitvector
   *  @param sgn Whether this bitvector is negative or positive.
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  smt_astt mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw);

  /** Create a rounding mode to be used by floating point cast and arith ops
   *  @param rm the kind of rounding mode
   *  @return The newly created rounding mode smt_ast. */
  smt_astt mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm);

  /** Convert a ieee subtraction
   *  @param lhs left hand side of the subtraction
   *  @param rhs right hand side of the subtraction
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  smt_astt mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm);

  /** Convert the ieee division
   *  @param lhs left hand side of the division
   *  @param rhs right hand side of the division
   *  @param rm the rounding mode
   *  @return The newly created cast smt_ast. */
  smt_astt mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm);

  /** Convert the ieee arithmetic square-root (sqrt)
   *  @param op the sqrt radicand
   *  @param rm the rounding mode
   *  @return The newly created sqrt smt_ast */
  smt_astt mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm);

  /** Convert an ieee greater than
   *  @param lhs left hand side
   *  @param rhs right hand side
   *  @return The newly created fp.gt smt_ast. */
  smt_astt mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs);

  /** Convert an ieee greater than or equal
   *  @param lhs left hand side
   *  @param rhs right hand side
   *  @return The newly created fp.gt smt_ast. */
  smt_astt mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs);

  /** Convert an ieee is_nan operation
   *  @param op the operand
   *  @return The newly created fp.isNaN smt_ast. */
  smt_astt mk_smt_fpbv_is_nan(smt_astt op);

  /** Convert an ieee is_normal operation
   *  @param op the operand
   *  @return The newly created fp.isNormal smt_ast. */
  smt_astt mk_smt_fpbv_is_normal(smt_astt op);

  /** Convert an ieee is_neg operation
   *  @param op the operand
   *  @return The newly created fp.isNegative smt_ast. */
  smt_astt mk_smt_fpbv_is_negative(smt_astt op);

  /** Convert an ieee is_pos operation
   *  @param op the operand
   *  @return The newly created fp.isPositive smt_ast. */
  smt_astt mk_smt_fpbv_is_positive(smt_astt op);

  /** Convert an ieee abs operation
   *  @param op the operand
   *  @return The newly created fp.abs smt_ast. */
  smt_astt mk_smt_fpbv_abs(smt_astt op);

  /** Extract the assignment to a floating-point from the SMT solvers model.
   *  @param a the AST whos value we wish to know.
   *  @return the ieee floating-point */
  ieee_floatt get_fpbv(smt_astt a);

  /** Convert FP to BV
   * @param op the floating-point
   */
  smt_astt mk_from_fp_to_bv(smt_astt op);

  /** Stores handle for the real-arithmetic/enclosure interface. */

  void bump_addrspace_array(unsigned int idx, const expr2tc &val);
  /** Get the symbol name for the current address-allocation record array. */
  std::string get_cur_addrspace_ident();

  /** Operand pairs of every division seen by note_division_operands. */
  std::set<std::pair<expr2tc, expr2tc>> divided_operand_pairs;
  /** Create and assert address space constraints on the given object ID
   *  number. Essentially, this asserts that all the objects to date don't
   *  overlap with /this/ one. */
  void finalize_pointer_chain(unsigned int obj_num);

  /** Typecast data to bools */
  smt_astt convert_typecast_to_bool(const typecast2t &cast);
  /** Typecast to a fixedbv in bitvector mode */
  smt_astt convert_typecast_to_fixedbv_nonint(const expr2tc &cast);
  /** Typecast anything to an integer (but not pointers) */
  smt_astt convert_typecast_to_ints(const typecast2t &cast);
  smt_astt convert_typecast_to_ints_intmode(const typecast2t &cast);
  smt_astt convert_typecast_to_ints_from_fbv_sint(const typecast2t &cast);
  smt_astt convert_typecast_to_ints_from_unsigned(const typecast2t &cast);
  smt_astt convert_typecast_to_ints_from_bool(const typecast2t &cast);
  /** Typecast something (i.e. an integer) to a pointer */
  smt_astt convert_typecast_to_ptr(const typecast2t &cast);
  /** Typecast a pointer to an integer */
  smt_astt convert_typecast_from_ptr(const typecast2t &cast);
  /** Typecast structs to other structs */
  smt_astt convert_typecast_to_struct(const typecast2t &cast);
  /** Despatch a typecast expression to a more specific typecast method */
  smt_astt convert_typecast(const expr2tc &expr);
  /** Typecast to a floatbv*/
  smt_astt convert_typecast_to_fpbv(const typecast2t &cast);
  /** Typecast from a floatbv */
  smt_astt convert_typecast_from_fpbv(const typecast2t &cast);
  /** Round a real to an integer; not straightforward at all. */
  smt_astt round_real_to_int(smt_astt a);
  /** Resize a shift amount's raw bits to a fixed-point operand's width. */
  smt_astt fxp_shift_amount(smt_astt amount, unsigned int width);
  /** Convert a mixed-format fixed-point result into the C result type. */
  smt_astt fxp_align_result(
    smt_astt a,
    const expr2tc &expr,
    const expr2tc &side_1,
    const expr2tc &side_2);
  /** Round an SMT integer to the nearest representable float/double using
   *  IEEE 754 round-to-nearest-even. Used for int->fp casts under --ir-ieee.
   *  source_width is the bit-width of the source integer type. */
  smt_astt round_int_to_fp(
    smt_astt int_val,
    const floatbv_type2t &fbv_type,
    unsigned int source_width);

  /** Prep call for creating a tuple array */
  smt_astt tuple_array_create_despatch(const expr2tc &expr, smt_sortt domain);

  /** Convert a boolean to a bitvector with one bit. */
  smt_astt make_bool_bit(smt_astt a);
  /** Convert a bitvector with one bit to a boolean. */
  smt_astt make_bit_bool(smt_astt a);

  /** Given an array index, extract the lower n bits of it, where n is the
   *  bitwidth of the array domain. */
  expr2tc fix_array_idx(const expr2tc &idx, const type2tc &array_type);
  /** For a multi-dimensional array, convert the type into a single dimension
   *  array. This works by concatenating the domain widths together into one
   *  large domain. */
  type2tc flatten_array_type(const type2tc &type);
  /** For a multi-dimensional constant array, flatten the actual definition of
   *  it down to a single dimension */
  expr2tc flatten_array_body(const expr2tc &expr);
  /** Get the base subtype of an array, delving through any intermediate
   *  multidimensional arrays. */
  type2tc get_flattened_array_subtype(const type2tc &type);
  /** Fetch the number of elements in an array (the domain). */
  expr2tc array_domain_to_width(const type2tc &type);

  /** For the given type, replace all instances of a pointer type with the
   *  struct representation of it. */
  void rewrite_ptrs_to_structs(type2tc &type);

  /** When dealing with multi-dimensional arrays, and selecting one element
   *  out of several dimensions, reduce it to an expression on a single
   *  dimensional array, by concatenating the indexes. Works in conjunction
   *  with flatten_array_type. */
  expr2tc decompose_select_chain(const expr2tc &expr, expr2tc &base);
  /** Like decompose_select_chain, but for multidimensional stores. */
  expr2tc decompose_store_chain(const expr2tc &expr, expr2tc &base);

  /** Prepare an array_of expression by flattening its dimensions, if it
   *  has more than one. */
  smt_astt convert_array_of_prep(const expr2tc &expr);
  /** Create an array of pointers; expects the init_val to be null, because
   *  there's no other way to initialize a pointer array in C, AFAIK. */
  smt_astt pointer_array_of(const expr2tc &init_val, unsigned long array_width);

  unsigned int
  get_member_name_field(const type2tc &t, const irep_idt &name) const;
  unsigned int
  get_member_name_field(const type2tc &t, const expr2tc &name) const;

  // Ours:
  /** Given an array expression, attempt to extract its valuation from the
   *  solver model, computing a constant_array2tc by calling get_array_elem. */
  expr2tc get_array(const type2tc &type, smt_astt array);

  /** @} */

  // Types

  // Type for (optional) AST cache

  struct smt_cache_entryt
  {
    const expr2tc val;
    smt_astt ast;
    unsigned int level;
  };

  typedef boost::multi_index_container<
    smt_cache_entryt,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(smt_cache_entryt, const expr2tc, val)>,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(smt_cache_entryt, unsigned int, level),
        std::greater<unsigned int>>>>
    smt_cachet;

  typedef std::unordered_map<type2tc, smt_sortt, type2_hash> smt_sort_cachet;

  // Members
  /** Number of un-popped context pushes encountered so far. */
  unsigned int ctx_level;

  /** A cache mapping expressions to converted SMT ASTs. */
  smt_cachet smt_cache;
  /** A mutex lock for writing to the cache. */
  std::mutex smt_cache_mutex;
  /** A cache of converted type2tc's to smt sorts */
  smt_sort_cachet sort_cache;

  /** Model-value cache for l_get(). Holds the current model's boolean
   *  assignments; cleared on every model-changing transition (pre_solve,
   *  push_ctx, pop_ctx) so a hit always reflects the latest solve. Keyed
   *  by the arena-owned expression behind the boolean smt_astt (solver ASTs
   *  are hash-consed, so identical pointer ⇒ identical term ⇒ identical
   *  model value). */
  std::unordered_map<const camada::SMTExpr *, tvt> l_get_cache;
  /** Pointer_logict object, which contains some code for formatting how
   *  pointers are displayed in counter-examples. This is a list so that we
   *  can push and pop data when context push/pop operations occur. */
  std::list<pointer_logict> pointer_logic;
  /** Constant struct representing the implementation of the pointer type --
   *  i.e., the struct type that pointers get translated to. */
  type2tc pointer_struct;
  /** The type of the machine integer that can store a pointer. */
  type2tc machine_ptr;
  /** Sort for booleans. For fast access. */
  smt_sortt boolean_sort;
  /** Whether we are encoding expressions in integer mode or not. */
  bool int_encoding;
  /** Whether --ir-ieee mode is active (integer encoding with IEEE float
   * semantics). */
  bool ir_ieee;
  /** A namespace containing all the types in the program. Used to resolve the
   *  rare case where we're doing some pointer arithmetic and need to have the
   *  concrete type of a pointer. */
  const namespacet &ns;
  /* Options contain all the parameters set by the user to run ESBMC */
  const optionst &options;

  bool ptr_foo_inited;

  smt_astt null_ptr_ast;
  smt_astt invalid_ptr_ast;

  /** Counter for generating unique bound-variable names in quantifiers. */
  size_t quantifier_counter;

  /** One recorded uninterpreted-function application in the Ackermannisation
   *  fallback: the argument asts, the fresh result ast, and the context level
   *  at which it was created (so it can be pruned on the matching pop, since
   *  pop_ctx deletes the asts made since the push). */
  struct uf_ackermann_entry
  {
    std::vector<smt_astt> args;
    smt_astt result;
    unsigned int level;
  };
  /** Per-name history of uninterpreted-function applications, used only by the
   *  Ackermannisation fallback in the base mk_smt_uninterpreted_function (for
   *  backends without native UF support). A later application of the same name
   *  is tied to each earlier one by an (args equal => results equal)
   *  assumption. */
  std::unordered_map<std::string, std::vector<uf_ackermann_entry>>
    uf_ackermann_history;
  /** Counter for the fresh result symbols minted by the Ackermann fallback. */
  size_t uf_ackermann_counter;

  /** Map from SSA symbol name to its forall/exists irep2 expression.
   *  Populated in convert_assign when a symbol is assigned a quantifier
   *  expression; used in convert_ast to inline nested quantifier bodies
   *  before substituting the outer bound variable. */
  std::unordered_map<irep_idt, expr2tc, irep_id_hash> forall_defs_;

  /** Mapping of name prefixes to use counts: when we want a fresh new name
   *  with a particular prefix, this map stores how many times that prefix has
   *  been used, and thus what number should be appended to make the name
   *  unique. */
  std::map<std::string, unsigned int> fresh_map;

  /** Integer recording how many times the address space allocation record
   *  array has been modified. Essentially, this is like the SSA variable
   *  number, for an array we build / modify at conversion time. In a list so
   *  that we can support push/pop operations. */
  std::list<unsigned int> addr_space_sym_num;
  /** Type of the address space allocation records. Currently a start address
   *  integer and an end address integer. */
  type2tc addr_space_type;
  /** Type of the array of address space allocation records. */
  type2tc addr_space_arr_type;
  /** List of address space allocation sizes. A map from the object number to
   *  the nubmer of bytes allocated. In a list to support pushing and
   *  popping. */
  std::list<std::map<unsigned, unsigned>> addr_space_data;

  /** One deferred `__ESBMC_addrspace_arr_N = with(arr_N-1, idx, val)` step. */
  struct addrspace_store
  {
    unsigned int idx;
    expr2tc val;
  };

  /** Address-space stores not yet handed to the solver.
   *
   *  The address-space array is written for every tracked object but read only
   *  by the two pointer-cast paths in smt_casts.cpp, both of which go through
   *  get_cur_addrspace_ident(). Programs that never cast between integers and
   *  pointers therefore never select from it, and asserting the store chain
   *  regardless costs one SMT array declaration per version per tuple field --
   *  enough to pull the script from QF_BV up into QF_ABV, and to roughly double
   *  the node count bitwuzla builds (nn-tanh_5_unsafe: 110k -> 55k initial
   *  nodes once the stores are deferred). Solve time on that benchmark is
   *  unchanged, so this buys formula size and the honest logic fragment, not
   *  speed.
   *
   *  Pre-camada ESBMC avoided this in its own array flattener, which kept an
   *  unbounded array as bookkeeping until a select forced it out
   *  (array_conv.cpp's is_unbounded_array path). Camada has no equivalent, and
   *  could not: ESBMC chooses the logic string before camada emits anything,
   *  and bitwuzla rejects array declarations under QF_BV, so eliding the
   *  arrays and staying in QF_BV has to happen here.
   *
   *  In a list to support push/pop, like addr_space_sym_num beside it. */
  std::list<std::vector<addrspace_store>> pending_addrspace_stores;

  /** Assert every deferred address-space store, oldest first, and stop
   *  deferring. Called when something is about to read the array. */
  void flush_addrspace_stores();

  /** Holds the `__ESBMC_alloc` symbol convert_terminal() was last invoked with.
   */
  expr2tc current_valid_objects_sym;

  /** Holds the `__ESBMC_is_dynamic` symbol convert_terminal() was last invoked
   * with.
   */
  expr2tc cur_dynamic;

  // XXX - push-pop will break here.
  typedef std::map<std::string, smt_astt> renumber_mapt;
  std::vector<renumber_mapt> renumber_map;

  std::unique_ptr<ir_ieee_convt> ir_ieee_api;

  // Workaround for integer shifts. This is an array of the powers of two,
  // up to 2^64.
  smt_astt int_shift_op_array;

private:
  double convert_rational_to_double(
    const BigInt &numerator,
    const BigInt &denominator);

  /** The camada solver every mk_* method encodes into. */
  std::unique_ptr<camada::SMTSolver> solver;

  /** The external one-shot program named by --smtlib-oneshot-prog, empty when
   *  that option was not given. Declared before `oneshot`, which is derived
   *  from it: members initialise in declaration order. */
  const std::string oneshot_prog;

  /** Set when the SMT-LIB backend drives an external one-shot program. */
  const bool oneshot;

  /** Camada's SMT-LIB backend streams the script to its sink as it is built
   *  rather than buffering it, so dump_smt() has nothing to return. */
  const bool streams_script = false;

  /** Where the one-shot script is written; empty unless `oneshot`. */
  const std::string formula_path;

  bool solved = false;

  const char *oneshot_label() const;
  camada::FPEncoding fp_encoding() const;
  camada::SMTExprRef
  make_index_expr(const camada::SMTSortRef &sort, uint64_t index);
  smt_astt fp_sign_test(smt_astt op, bool negative);
  smt_resultt oneshot_dec_solve();

  /* SMT-LIB's Int/Real theory has no bitwise operations, but ESBMC still
   * routes C's &/|/^/~ through the mk_bv* entry points in integer mode (see
   * the bitand_id case in smt_solver.cpp). Bridge by converting to a
   * bit-vector of the signed word width, applying the bit-vector operation,
   * and converting back. */
  template <typename Fn>
  smt_astt int_bitwise_binary(smt_astt a, smt_astt b, Fn &&op)
  {
    const unsigned width = signed_size_type2()->get_width();
    auto a_bv = solver->mkInt2BV(width, a);
    auto b_bv = solver->mkInt2BV(width, b);
    return solver->mkBV2Int(op(a_bv, b_bv), true);
  }
};

/** Given an array type, create a type2tc representing its domain. */
type2tc make_array_domain_type(const array_type2t &arr);

/** Return the SMT domain bit-width for an array type.
 *  For constant-size arrays this is the minimum bit-width needed to represent
 *  every valid index.  For VLA, dynamic, or infinite arrays the size is not
 *  statically known, so the machine word size is returned as a safe default. */
unsigned long array_domain_width_or_word_size(const array_type2t &arr);

/* Type for push/pop-aware symbol table cache, required by some solvers */

struct symtab_entryt
{
  std::string val;
  smt_astt ast;
  unsigned int level;

  symtab_entryt(std::string val, smt_astt ast, unsigned int level)
    : val(std::move(val)), ast(ast), level(level)
  {
  }
};

typedef boost::multi_index_container<
  symtab_entryt,
  boost::multi_index::indexed_by<
    boost::multi_index::hashed_unique<
      BOOST_MULTI_INDEX_MEMBER(symtab_entryt, std::string, val)>,
    boost::multi_index::ordered_non_unique<
      BOOST_MULTI_INDEX_MEMBER(symtab_entryt, unsigned int, level),
      std::greater<unsigned int>>>>
  symtabt;

#endif /* _ESBMC_PROP_SMT_SMT_SOLVER_H_ */

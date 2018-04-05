/*******************************************************************\

Module: Pointer Dereferencing

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_POINTER_ANALYSIS_DEREFERENCE_H
#define CPROVER_POINTER_ANALYSIS_DEREFERENCE_H

#include <pointer-analysis/value_sets.h>
#include <set>
#include <util/expr.h>
#include <util/guard.h>
#include <util/hash_cont.h>
#include <util/namespace.h>
#include <util/options.h>

/** @file dereference.h
 *  The dereferencing code's purpose is to take a symbol with pointer type that
 *  is being dereferenced, and translate it to an expression representing the
 *  values it read or wrote to/from. Note that dereferences can be
 *  'dereference2t' expressions, or indexes that have a base with pointer type.
 *
 *  The inputs to this process are an expression that contains a dereference
 *  that is to be translated, and a value_sett object that contains the set of
 *  references that pointers may point at.
 *
 *  The output is an expression referring that nondeterministically evaluates
 *  to the piece of data that the dereference points at. This (usually) takes
 *  the shape of a large chain of 'if' expressions, with appropriate guards to
 *  switch between which data object the pointer points at, according to the
 *  model that the SAT solver is building. If the pointer points at something
 *  unknown (or dereferencing fails), then it evalates to a 'failed symbol', a
 *  free variable that isn't used anywhere else.
 *
 *  The dereferencing process also outputs a series of dereference assertions,
 *  the failure of which indicate that the dereference performed is invalid in
 *  some way, and that the data used  (almost certainly) ends up being a free
 *  variablea free variable. These are recorded by calling methods in the
 *  dereference_callbackt object. Assertions can be disabled with the
 *  --no-pointer-check and --no-align-check options, but code modelled then
 *  relies on undefined behaviour.
 *
 *  There are four steps to the dereferencing process:
 *   1) Interpretation of the expression surrounding the dereference. The
 *      presence of tertiary operators, short-circuit logic and so forth can
 *      affect whether a dereference is performed, and thus whether assertions
 *      fire. Identifying index/member operations applied also helps to simplify
 *      the resulting dereference.
 *   2) Collect data objects that the dereferenced pointer points at, the offset
 *      an alignment they're accessed with, and have references to each built
 *      (part 3). Each built reference comes with a guard that is true when the
 *      pointer variable points at that data object, which are used to combine
 *      all referred to objects into one object (as a gigantic 'if' chain).
 *   3) Reference building -- this is where all the dirty work is. Given a data
 *      object, and a (possibly nondeterminstic) offset, this code must create
 *      an expression that evaluates to the value in the data object, at the
 *      given offset, with the desired type of this dereference.
 *   4) Assertions: We encode assertion that the offset being accessed lies
 *      withing the data object being referred to, but also that the access has
 *      the correct alignment, and doesn't access padding bytes.
 *
 *  A particular point of interest is the byte layout of the data objects we're
 *  dealing with. The underlying SMT objects mean nothing during byte
 *  addressing, only the code in 'type_byte_size' is relevant. To aid
 *  dereferencing, that code ensures that all data objects are word aligned in
 *  all struct fields (and that trailing padding exists). This is so that we
 *  can avoid all scenarios where dereferences cross field boundries, as that
 *  will ultimately be an alignment violation.
 *
 *  The value set tracking code also maintains a 'minimum alignment' piece of
 *  data when the offset is nondeterministic, guarenteeing that the offset used
 *  will be aligned to at least that many bytes. This allows for reducing the
 *  number of behaviours we have to support when dereferencing, particularly
 *  useful when accessing array elements.
 *
 *  The majority of code is in the reference building stuff. It would be aimless
 *  to document all it's ecentricities, but here are a few guidelines:
 *   * There are two different flavours of offsets, the offset of the pointer
 *     being dereferenced, and the offset caused by the expression that's
 *     dereferencing it.
 *   * 'Scalar step lists' are a list of expressions applied to a struct or
 *     array to access a scalar within the base object. For example, the expr
 *     "foo->bar[baz]" dereferences foo and applies a member then index expr.
 *     The idea behind this is that we can directly pull the scalar value out
 *     of the underlying type if possible, instead of having to compute a
 *     reference to a struct or array in dereference code.
 *   * In that vein, building a reference to a struct only happens as a last
 *     resort, and is vigorously asserted against. An exception to this rule
 *     is when the underlying object is a byte array: we have to support this
 *     so that one can malloc memory for structure objects.
 *   * Sometimes in the worst case scenario, we're unable to access a data
 *     object in any sane way (like an int access to a short array), and end up
 *     needing to reconstruct the desired type from the byte representation.
 *     This tends to get referred to as 'stitching it together from bytes'.
 */

/** Class providing interface to value set tracking code.
 *  This class allows dereference code to get more data out of the environment
 *  in which it is dereferencing, fetching the set of values that a pointer
 *  points at, and also encoding any asertions that occur regarding the validity
 *  of the dereference.
 */
class dereference_callbackt
{
public:
  virtual ~dereference_callbackt() = default;

  /** Triggers a 'valid object' check when accessing a dynamically allocated
   *  object. This is legacy, and will be deleted at some point. */
  virtual bool is_valid_object(const irep_idt &identifier) = 0;

  /** Encode a dereference failure assertion. If a dereference does, or can
   *  trigger undefined or illegal behaviour, then this method is called to
   *  record it so that it can be asserted against.
   *  @param property Classification of the assertion being violated.
   *  @param msg Description of the reason for this assertion.
   *  @param guard Guard for this assertion -- evaluates to true whe this
   *         assertion has been violated.
   */

  virtual void dereference_failure(
    const std::string &property,
    const std::string &msg,
    const guardt &guard) = 0;

  /** Fetch the set of values that the given pointer variable can point at.
   *  @param expr Pointer symbol to get the value set of.
   *  @param value_set A value set to store the output of this call into.
   */
  virtual void
  get_value_set(const expr2tc &expr, value_setst::valuest &value_set) = 0;

  /** Check whether a failed symbol already exists for the given symbol.
   *  This is legacy, and will be removed at some point soon. */
  virtual bool
  has_failed_symbol(const expr2tc &expr, const symbolt *&symbol) = 0;

  /** Optionally rename the given expression. This exists to provide potential
   *  optimisation expansion in the future, it isn't currently used by anything.
   *  @param expr An expression to be renamed
   */
  virtual void rename(expr2tc &expr __attribute__((unused)))
  {
  }

  struct internal_item
  {
    expr2tc object;
    expr2tc offset;
    expr2tc guard;
  };

  virtual void dump_internal_state(const std::list<struct internal_item> &data
                                   __attribute__((unused)))
  {
  }
};

/** Class containing expression dereference logic.
 *  This class doesn't actually store any state, in that all the side-effects
 *  of what it does are either returned to the caller, or passed through the
 *  dereference_callbackt object's virtual methods. Members of this class only
 *  exist to provide context to the dereference being performed.
 */
class dereferencet
{
public:
  /** Primary constructor.
   *  @param _ns Namespace to make type lookups against
   *  @param _new_context Context to store new (failed) symbols in.
   *  @param _options Options to behave under (i.e., disable ptr checks etc).
   *  @param _dereference_callback Callback object to invoke when we need
   *         external information or otherwise need to interacte with the
   *         context.
   */
  dereferencet(
    const namespacet &_ns,
    contextt &_new_context,
    const optionst &_options,
    dereference_callbackt &_dereference_callback)
    : ns(_ns),
      new_context(_new_context),
      options(_options),
      dereference_callback(_dereference_callback),
      block_assertions(false)
  {
    is_big_endian =
      (config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN);
  }

  virtual ~dereferencet() = default;

  /** The different ways in which a pointer may be accessed. */
  typedef enum {
    READ,     /// The result of the expression is only read.
    WRITE,    /// The result of the expression will be written to.
    FREE,     /// The referred to object will be freed.
    INTERNAL, /// Calling code only wants the internal value-set data.
  } modet;

  /** Take an expression and dereference it.
   *  This will descend through the whole of the expression given, and
   *  dereference any dereferences contained within it. The given expr will
   *  be modified in place. It also doesn't necessarily have to contain a
   *  dereference at all (in which case nothing will happen), or start at a
   *  relevant part of the expression, as this method will recurse through the
   *  whole thing.
   *  @param expr The expression to be dereferenced.
   *  @param guard Guard to be added to any dereference failure assertions
   *         generated.
   *  @param The way in which this dereference is being accessed. Only affects
   *         the assertions that are generated.
   */
  virtual void dereference_expr(expr2tc &expr, guardt &guard, modet mode);

  virtual expr2tc dereference(
    const expr2tc &dest,
    const type2tc &type,
    const guardt &guard,
    modet mode,
    const expr2tc &extra_offset);

  /** Does the given expression have a dereference in it somewhere?
   *  @param expr The expression to check for existance of a dereference.
   *  @return True when the given expression does have a dereference.
   */
  bool has_dereference(const expr2tc &expr) const;

private:
  /** Namespace to perform type lookups against. */
  const namespacet &ns;
  /** Context in which to store new (failed) symbols that are generated. */
  contextt &new_context;
  /** Options from the command line. */
  const optionst &options;
  /** The callback object to funnel all interactions with the context through.*/
  dereference_callbackt &dereference_callback;
  /** The number of failed symbols that we've generated (they're numbered
   *  individually. */
  static unsigned invalid_counter;
  /** Whether or not we're operating in a big endian environment. Value for this
   *  is taken from config.ansi_c.endianness. */
  bool is_big_endian;
  /** List of internal state items -- these contain all the data of interest
   *  to build_reference_to, but in INTERNAL mode we skip the construction
   *  of a reference, and instead return the data to the caller via the
   *  callback. */
  std::list<dereference_callbackt::internal_item> internal_items;
  /** Flag for discarding all assertions encoded. */
  bool block_assertions;

  /** Interpret an expression that modifies the guard. i.e., an 'if' or a
   *  piece of logic that can be short-circuited.
   *  @param expr The expression that we're looking for dereferences in.
   *  @param guard The guard for evaluating this expression.
   *  @param mode The manner in which the result of this deref is accessed.
   */
  virtual void dereference_guard_expr(expr2tc &expr, guardt &guard, modet mode);

  /** Interpret an address-of expression. There's the potential that it's just
   *  taking the address of a dereference, which just evaluates to some pointer
   *  arithemtic, and thus can be handled without actually dereferencing.
   *  @param expr Address-of expression we're attempting to handle.
   *  @param guard Guard of this expression being evaluated.
   *  @param mode The manner iin which the result of this deref is accessed.
   */
  virtual void
  dereference_addrof_expr(expr2tc &expr, guardt &guard, modet mode);

  /** Interpret an actual dereference (or pointer-index) expression. First
   *  dereferences the pointer expression, then dereferences the pointer itself,
   *  and stores the result in the 'expr' reference.
   *  @param expr The expression we're going to be dereferencing.
   *  @param guard Guard of this expression being evaluated.
   *  @param mode The manner iin which the result of this deref is accessed.
   */
  virtual void dereference_deref(expr2tc &expr, guardt &guard, modet mode);

  /** Interpret an expression that accesses a nonscalar type. This means that
   *  it's an index or member (or some other glue expr) on top of a dereference
   *  that evaluates to an array or struct. This code collects all of these
   *  expressions into a list, and supplies it to other dereference code, so
   *  that we can directly build a reference to the field this expr wants. This
   *  means that we don't have to build any intermediate struct or array
   *  references, which is beneficial.
   *  @param dest The expression that we're dereferencing.
   *  @param guard Guard of this expression being evaluated.
   *  @param mode The manner iin which the result of this deref is accessed.
   *  @param scalar_step_list A list in which we're accumulating the exprs used
   *         to build a scalar access to an aggregate type.
   */
  virtual expr2tc dereference_expr_nonscalar(
    expr2tc &dest,
    guardt &guard,
    modet mode,
    std::list<expr2tc> &scalar_step_list);

  /** Check whether an (aggregate) type is compatible with the desired
   *  dereference type. This looks at various things, such as whether the given
   *  struct is a subclass of the desired type, and inserts typecasts as
   *  appropriate.
   *  @param object An object with an aggregate type, that we're checking to
   *         see whether it's compatible with the desired type.
   *  @param dereference_type The desired type.
   *  @return True if the types are compatible, possibly after typecast applied.
   */
  bool dereference_type_compare(
    expr2tc &object,
    const type2tc &dereference_type) const;

  /** Create a new, free, symbol of the given type. This happens when we've
   *  failed to dereference for some reason, but we still need to build a valid
   *  SMT formula so that the relevant assertion failure can be reached.
   *  @param out_type The type of the failed symbol to create.
   *  @return The new, free variable.
   */
  expr2tc make_failed_symbol(const type2tc &out_type);

  /** Try to build a reference to a data object. When we have a data object that
   *  a pointer (might) point at and need an expression to access it, this
   *  performs the require juggling. Some very strange approaches may come out
   *  of this (such as stitching together the result from bytes). It may also
   *  just completely fail and return a nil expression.
   *  Almost all of the time, the base data object and the offset into it from
   *  the dereferenced variable are stored in the 'what' parameter, as an
   *  object_descriptor2t.
   *  @param what The data object we're accessing. Possibly an unknown2t or
   *         invalid2t, but more frequently an object_descriptor2t. As this
   *         comes directly from the value set code, the base object might have
   *         some gratuitous index and member expressions on top of it.
   *  @param mode The manner in which the reference is going to be accessed.
   *  @param deref_expr The expression that is being dereferenced (i.e., the
   *         pointer variable or similar).
   *  @param type The desired outcome type from this dereference.
   *  @param guard The guard of this dereference occuring.
   *  @param lexical_offset Offset introduced by lexical expressions, i.e.
   *         indexes and member operations applied to a dereferenced struct or
   *         union.
   *  @param pointer_guard Output expression: this is set to be the guard
   *         against the pointer variable being the same object as the referred
   *         to type.
   *  @return If successful, an expression that refers to the object in 'what'.
   *          Otherwise, a nil expression.
   */
  expr2tc build_reference_to(
    const expr2tc &what,
    modet mode,
    const expr2tc &deref_expr,
    const type2tc &type,
    const guardt &guard,
    const expr2tc &lexical_offset,
    expr2tc &pointer_guard);

  void
  deref_invalid_ptr(const expr2tc &deref_expr, const guardt &guard, modet mode);

  static const expr2tc &get_symbol(const expr2tc &object);
  void bounds_check(
    const expr2tc &expr,
    const expr2tc &offset,
    const type2tc &type,
    const guardt &guard);
  void valid_check(const expr2tc &expr, const guardt &guard, modet mode);
  expr2tc *extract_bytes_from_array(
    const expr2tc &array,
    unsigned int bytes,
    const expr2tc &offset);
  expr2tc *extract_bytes_from_scalar(
    const expr2tc &object,
    unsigned int bytes,
    const expr2tc &offset);
  void stitch_together_from_byte_array(
    expr2tc &value,
    const type2tc &type,
    const expr2tc *bytes);
  void wrap_in_scalar_step_list(
    expr2tc &value,
    std::list<expr2tc> *scalar_step_list,
    const guardt &guard);
  void dereference_failure(
    const std::string &error_class,
    const std::string &error_name,
    const guardt &guard);
  void alignment_failure(const std::string &error_name, const guardt &guard);

  void bad_base_type_failure(
    const guardt &guard,
    const std::string &wants,
    const std::string &have);

  void check_code_access(
    expr2tc &value,
    const expr2tc &offset,
    const type2tc &type,
    const guardt &guard,
    modet mode);
  void check_data_obj_access(
    const expr2tc &value,
    const expr2tc &offset,
    const type2tc &type,
    const guardt &guard);
  void check_alignment(
    unsigned long minwidth,
    const expr2tc &&offset,
    const guardt &guard);

public:
  void build_reference_rec(
    expr2tc &value,
    const expr2tc &offset,
    const type2tc &type,
    const guardt &guard,
    modet mode,
    unsigned long alignment = 0);

private:
  void construct_from_const_offset(
    expr2tc &value,
    const expr2tc &offset,
    const type2tc &type);
  void construct_from_dyn_offset(
    expr2tc &value,
    const expr2tc &offset,
    const type2tc &type);
  void construct_from_const_struct_offset(
    expr2tc &value,
    const expr2tc &offset,
    const type2tc &type,
    const guardt &guard,
    modet mode);
  void construct_from_dyn_struct_offset(
    expr2tc &value,
    const expr2tc &offset,
    const type2tc &type,
    const guardt &guard,
    unsigned long alignment,
    modet mode,
    const expr2tc *failed_symbol = nullptr);
  void construct_from_multidir_array(
    expr2tc &value,
    const expr2tc &offset,
    const type2tc &type,
    const guardt &guard,
    unsigned long alignment,
    modet mode);
  void construct_struct_ref_from_const_offset_array(
    expr2tc &value,
    const expr2tc &offs,
    const type2tc &type,
    const guardt &guard,
    modet mode,
    unsigned long alignment);
  void construct_struct_ref_from_const_offset(
    expr2tc &value,
    const expr2tc &offs,
    const type2tc &type,
    const guardt &guard);
  void construct_struct_ref_from_dyn_offset(
    expr2tc &value,
    const expr2tc &offs,
    const type2tc &type,
    const guardt &guard,
    modet mode);
  void construct_struct_ref_from_dyn_offs_rec(
    const expr2tc &value,
    const expr2tc &offs,
    const type2tc &type,
    const expr2tc &accuml_guard,
    modet mode,
    std::list<std::pair<expr2tc, expr2tc>> &output);
  void construct_from_array(
    expr2tc &value,
    const expr2tc &offset,
    const type2tc &type,
    const guardt &guard,
    modet mode,
    unsigned long alignment = 0);

public:
  void set_block_assertions(void)
  {
    block_assertions = true;
  }

  void clear_block_assertions(void)
  {
    block_assertions = false;
  }
};

#endif

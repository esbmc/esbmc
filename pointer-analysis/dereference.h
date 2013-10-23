/*******************************************************************\

Module: Pointer Dereferencing

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_POINTER_ANALYSIS_DEREFERENCE_H
#define CPROVER_POINTER_ANALYSIS_DEREFERENCE_H

#include <set>

#include <expr.h>
#include <hash_cont.h>
#include <guard.h>
#include <namespace.h>
#include <options.h>

#include "value_sets.h"

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
 *     resort, and is vigorously asserted against.
 */

class dereference_callbackt
{
public:
  virtual ~dereference_callbackt()
  {
  }

  virtual bool is_valid_object(const irep_idt &identifier)=0;

  virtual void dereference_failure(
    const std::string &property,
    const std::string &msg,
    const guardt &guard)=0;

  typedef hash_set_cont<exprt, irep_hash> expr_sett;

  virtual void get_value_set(
    const expr2tc &expr,
    value_setst::valuest &value_set)=0;
  
  virtual bool has_failed_symbol(
    const expr2tc &expr,
    const symbolt *&symbol)=0;

  virtual void rename(expr2tc &expr __attribute__((unused)))
  {
    return;
  }
};

class dereferencet
{
public:
  dereferencet(
    const namespacet &_ns,
    contextt &_new_context,
    const optionst &_options,
    dereference_callbackt &_dereference_callback):
    ns(_ns),
    new_context(_new_context),
    options(_options),
    dereference_callback(_dereference_callback)
  {
    is_big_endian =
      (config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN);
  }

  virtual ~dereferencet() { }
  
  typedef enum { READ, WRITE, FREE } modet;

  virtual void dereference_expr(expr2tc &expr, guardt &guard, modet mode);
  virtual void dereference_guard_expr(expr2tc &expr, guardt &guard, modet mode);
  virtual void dereference_addrof_expr(expr2tc &expr, guardt &guard,
                                       modet mode);
  virtual void dereference_deref(expr2tc &expr, guardt &guard, modet mode);

  virtual expr2tc dereference_expr_nonscalar(
    expr2tc &dest,
    guardt &guard,
    modet mode,
    std::list<expr2tc> &scalar_step_list);

  virtual expr2tc dereference(
    const expr2tc &dest,
    const type2tc &type,
    const guardt &guard,
    modet mode,
    std::list<expr2tc> *scalar_step_list);

  bool has_dereference(const expr2tc &expr) const;

  typedef hash_set_cont<exprt, irep_hash> expr_sett;

private:
  const namespacet &ns;
  contextt &new_context;
  const optionst &options;
  dereference_callbackt &dereference_callback;
  static unsigned invalid_counter;
  bool is_big_endian;

  bool dereference_type_compare(
    expr2tc &object,
    const type2tc &dereference_type) const;

  expr2tc make_failed_symbol(const type2tc &out_type);

  expr2tc build_reference_to(
    const expr2tc &what,
    modet mode,
    const expr2tc &deref_expr,
    const type2tc &type,
    const guardt &guard,
    std::list<expr2tc> *scalar_step_list,
    expr2tc &pointer_guard);

  static const expr2tc &get_symbol(const expr2tc &object);
  void bounds_check(const expr2tc &expr, const expr2tc &offset,
                    const type2tc &type, const guardt &guard);
  void valid_check(const expr2tc &expr, const guardt &guard, modet mode);
  void stitch_together_from_byte_array(expr2tc &value, const type2tc &type,
                                       const expr2tc &offset);
  void wrap_in_scalar_step_list(expr2tc &value,
                                std::list<expr2tc> *scalar_step_list,
                                const guardt &guard);
  void dereference_failure(const std::string &error_class,
                           const std::string &error_name,
                           const guardt &guard);
  void alignment_failure(const std::string &error_name, const guardt &guard);

  void check_code_access(expr2tc &value, const expr2tc &offset,
                         const type2tc &type, const guardt &guard, modet mode);
  void check_data_obj_access(const expr2tc &value, const expr2tc &offset,
                             const type2tc &type, const guardt &guard);

  void build_reference_rec(expr2tc &value, const expr2tc &offset,
                           const type2tc &type, const guardt &guard, modet mode,
                           unsigned long alignment = 0,
                           std::list<expr2tc> *scalar_step_list = NULL);
  void construct_from_const_offset(expr2tc &value, const expr2tc &offset,
                                   const type2tc &type);
  void construct_from_dyn_offset(expr2tc &value, const expr2tc &offset,
                                 const type2tc &type);
  void construct_from_const_struct_offset(expr2tc &value,
                                             const expr2tc &offset,
                                             const type2tc &type,
                                             const guardt &guard,
                                             modet mode);
  void construct_from_dyn_struct_offset(expr2tc &value,
                                           const expr2tc &offset,
                                           const type2tc &type,
                                           const guardt &guard,
                                           unsigned long alignment,
                                           const expr2tc *failed_symbol = NULL);
  void construct_from_multidir_array(expr2tc &value, const expr2tc &offset,
                                        const type2tc &type,
                                        const guardt &guard,
                                        unsigned long alignment,
                                        modet mode);
  void construct_struct_ref_from_const_offset(expr2tc &value,
                                        const expr2tc &offs,
                                        const type2tc &type,
                                        const guardt &guard);
  void construct_struct_ref_from_dyn_offset(expr2tc &value,
                                        const expr2tc &offs,
                                        const type2tc &type,
                                        const guardt &guard,
                                        std::list<expr2tc> *scalar_step_list);
  void construct_struct_ref_from_dyn_offs_rec(const expr2tc &value,
                              const expr2tc &offs, const type2tc &type,
                              const expr2tc &accuml_guard,
                              std::list<std::pair<expr2tc, expr2tc> > &output);
  void construct_from_array(expr2tc &value, const expr2tc &offset,
                            const type2tc &type, const guardt &guard,
                            modet mode, unsigned long alignment = 0);
};

#endif

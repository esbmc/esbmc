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

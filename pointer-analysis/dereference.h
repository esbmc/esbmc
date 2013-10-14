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

  virtual void dereference_expr(
    expr2tc &dest,
    guardt &guard,
    const modet mode,
    bool checks_only = false);

  virtual expr2tc dereference_expr_nonscalar(
    expr2tc &dest,
    guardt &guard,
    const modet mode,
    std::list<expr2tc> &scalar_step_list,
    bool checks_only = false);

  virtual expr2tc dereference(
    const expr2tc &dest,
    const type2tc &type,
    const guardt &guard,
    const modet mode,
    std::list<expr2tc> *scalar_step_list,
    bool checks_only = false);

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

  void build_reference_to(
    const expr2tc &what,
    const modet mode,
    const expr2tc &deref_expr,
    const type2tc &type,
    expr2tc &value,
    expr2tc &pointer_guard,
    const guardt &guard,
    std::list<expr2tc> *scalar_step_list);

  static const expr2tc &get_symbol(const expr2tc &object);

  void bounds_check(const type2tc &type, const expr2tc &offset,
                    unsigned int access_size, const guardt &guard);
  void valid_check(const expr2tc &expr, const guardt &guard, const modet mode);

  void construct_from_zero_offset(expr2tc &value, const type2tc &type,
                                  const guardt &guard,
                                  std::list<expr2tc> *scalar_step_list);
  void construct_from_const_offset(expr2tc &value, const expr2tc &offset,
                                   const type2tc &type, const guardt &guard,
                                  std::list<expr2tc> *scalar_step_list,
                                  bool checks = true);
  void construct_from_dyn_offset(expr2tc &value, const expr2tc &offset,
                                 const type2tc &type, const guardt &guard,
                                 unsigned long alignment, bool checks = true);
  void construct_from_const_struct_offset(expr2tc &value,
                                             const expr2tc &offset,
                                             const type2tc &type,
                                             const guardt &guard);
  void construct_from_dyn_struct_offset(expr2tc &value,
                                           const expr2tc &offset,
                                           const type2tc &type,
                                           const guardt &guard,
                                           unsigned long alignment,
                                           const expr2tc *failed_symbol = NULL);
  void construct_from_multidir_array(expr2tc &value, const expr2tc &offset,
                                        const type2tc &type,
                                        const guardt &guard,
                                        std::list<expr2tc> *scalar_step_list,
                                        unsigned long alignment);

  void construct_struct_ref_from_const_offset(expr2tc &value,
                                        const expr2tc &offs,
                                        const type2tc &type,
                                        const guardt &guard,
                                        std::list<expr2tc> *scalar_step_list);

  void construct_struct_ref_from_dyn_offset(expr2tc &value,
                                        const expr2tc &offs,
                                        const type2tc &type,
                                        const guardt &guard,
                                        std::list<expr2tc> *scalar_step_list);

  bool memory_model(
    expr2tc &value,
    const type2tc &type,
    const guardt &guard,
    expr2tc &new_offset);

  bool memory_model_bytes(
    expr2tc &value,
    const type2tc &type,
    const guardt &guard,
    expr2tc &new_offset);

  unsigned int fabricate_scalar_access(const type2tc &src_type,
                                       std::list<expr2tc> &scalar_step_list);
  void wrap_in_scalar_step_list(expr2tc &value,
                                std::list<expr2tc> *scalar_step_list,
                                const guardt &guard);
};

#endif

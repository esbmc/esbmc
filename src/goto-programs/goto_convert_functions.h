#ifndef CPROVER_GOTO_CONVERT_FUNCTIONS_H
#define CPROVER_GOTO_CONVERT_FUNCTIONS_H

#include <goto-programs/goto_convert_class.h>
#include <goto-programs/goto_functions.h>

// just convert it all
void goto_convert(
  contextt &context,
  optionst &options,
  goto_functionst &functions);

class goto_convert_functionst : public goto_convertt
{
public:
  typedef std::map<irep_idt, std::set<irep_idt>> typename_mapt;
  typedef std::set<irep_idt> typename_sett;

  void goto_convert();
  void convert_function(symbolt &symbol);
  void thrash_type_symbols();
  void collect_type(const irept &type, typename_sett &set);
  void collect_expr(const irept &expr, typename_sett &set);
  void
  rename_types(irept &type, const symbolt &cur_name_sym, const irep_idt &sname);
  void
  rename_exprs(irept &expr, const symbolt &cur_name_sym, const irep_idt &sname);

  // Const probes mirroring rename_types/rename_exprs: return true iff the
  // subtree actually contains a `symbol` type that rename_types would
  // replace. rename_*() only mutate at those points, but the mutable
  // Forall_* traversal detaches (deep-copies, under COW) every node it
  // walks regardless. Gating the mutable descent on these read-only
  // checks avoids detaching the overwhelming majority of nodes that
  // have nothing to rename.
  bool type_needs_rename(const irept &type, const irep_idt &sname) const;
  bool expr_needs_rename(const irept &expr, const irep_idt &sname) const;
  void
  wallop_type(irep_idt name, typename_mapt &typenames, const irep_idt &sname);

  goto_convert_functionst(
    contextt &_context,
    optionst &_options,
    goto_functionst &_functions);

protected:
  goto_functionst &functions;

  static bool hide(const goto_programt &goto_program);

  //
  // function calls
  //
  void add_return(goto_functiont &f, const locationt &location);

  void wallop_type_impl(
    irep_idt name,
    typename_mapt &typenames,
    const irep_idt &sname,
    std::set<irep_idt> &in_progress);
};

#endif

/*******************************************************************\

Module: Goto Programs with Functions

Author: Daniel Kroening

Date: June 2003

\*******************************************************************/

#ifndef CPROVER_GOTO_FUNCTIONS_H
#define CPROVER_GOTO_FUNCTIONS_H

#define Forall_goto_functions(it, functions)                                   \
  for(goto_functionst::function_mapt::iterator it =                            \
        (functions).function_map.begin();                                      \
      it != (functions).function_map.end();                                    \
      it++)

#define forall_goto_functions(it, functions)                                   \
  for(goto_functionst::function_mapt::const_iterator it =                      \
        (functions).function_map.begin();                                      \
      it != (functions).function_map.end();                                    \
      it++)

#include <goto-programs/goto_program.h>
#include <util/std_types.h>

class goto_functiont
{
  bool inlined;

public:
  goto_programt body;
  code_typet type;
  bool body_available;

  // The set of functions that have been inlined into this one. Necessary to
  // make symex renaming work.
  std::set<std::string> inlined_funcs;

  void set_inlined(bool i)
  {
    inlined = i;
  }

  bool is_inlined() const
  {
    return inlined;
  }

  goto_functiont() : inlined(false), body_available(false)
  {
  }
};

class goto_functionst
{
public:
  typedef std::map<irep_idt, goto_functiont> function_mapt;
  function_mapt function_map;

  ~goto_functionst() = default;
  void clear()
  {
    function_map.clear();
  }

  void output(const namespacet &ns, std::ostream &out) const;
  void dump() const;

  void compute_location_numbers();
  void compute_loop_numbers();
  void compute_target_numbers();

  void update()
  {
    compute_target_numbers();
    compute_location_numbers();
    compute_loop_numbers();
  }

  irep_idt main_id() const
  {
    return "__ESBMC_main";
  }

  void swap(goto_functionst &other)
  {
    function_map.swap(other.function_map);
  }
};

#endif

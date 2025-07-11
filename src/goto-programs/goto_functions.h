#ifndef CPROVER_GOTO_FUNCTIONS_H
#define CPROVER_GOTO_FUNCTIONS_H

#define Forall_goto_functions(it, functions)                                   \
  for (goto_functionst::function_mapt::iterator it =                           \
         (functions).function_map.begin();                                     \
       it != (functions).function_map.end();                                   \
       it++)

#define forall_goto_functions(it, functions)                                   \
  for (goto_functionst::function_mapt::const_iterator it =                     \
         (functions).function_map.begin();                                     \
       it != (functions).function_map.end();                                   \
       it++)

#include <goto-programs/goto_program.h>
#include <util/std_types.h>
#include <util/options.h>

class goto_functiont
{
public:
  goto_programt body;
  code_typet type;
  bool body_available = false;

  // The set of functions that have been inlined into this one. Necessary to
  // make symex renaming work.
  std::set<std::string> inlined_funcs;

  /// update the function member in each instruction
  /// \param function_id: the `function_id` used for assigning empty function
  ///   members
  void update_instructions_function(const irep_idt &function_id)
  {
    body.update_instructions_function(function_id);
  }
};

class goto_functionst
{
public:
  typedef std::map<irep_idt, goto_functiont> function_mapt;
  function_mapt function_map;

  // For coverage and multi-property
  // Store and pass the coverage data in incr/kind mode
  static std::unordered_set<std::string> reached_claims;
  static std::unordered_multiset<std::string> reached_mul_claims;
  static std::unordered_set<std::string> verified_claims;

  static std::mutex reached_claims_mutex;
  static std::mutex reached_mul_claims_mutex;
  static std::mutex verified_claims_mutex;

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

void get_local_identifiers(const goto_functiont &, std::set<irep_idt> &dest);

#endif

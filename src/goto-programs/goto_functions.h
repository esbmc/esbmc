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
#include <util/exception_specification.h>

class goto_functiont
{
public:
  goto_programt body;
  // The function signature (a code_type2t). Unlike the legacy code_typet this
  // replaced, a default-constructed type2tc is nil rather than an empty code
  // type, so it must be assigned (migrate_type(symbol.type)) before any read:
  // to_code_type() dereferences it. All construction paths set it before use.
  type2tc type;
  bool body_available = false;

  // Cached, decoded copy of the function's C++ exception specification. The
  // canonical representation lives on the function symbol's typet (and so
  // survives GOTO-binary serialization); this is derived from it whenever the
  // function type is (re)built. Used by symex to enforce the specification at
  // the function-frame boundary.
  exception_specificationt exception_spec;

  // The set of functions that have been inlined into this one. Necessary to
  // make symex renaming work.
  std::set<std::string> inlined_funcs;

  /// update the function member in each instruction
  /// \param function_id: the `function_id` used for assigning empty function
  ///   members
  /// \param force: when true, overwrite already-set function members too
  void
  update_instructions_function(const irep_idt &function_id, bool force = false)
  {
    body.update_instructions_function(function_id, force);
  }
};

class goto_functionst
{
public:
  /// Order function_map by the id's string content, not by irep_idt's default
  /// operator< (which compares the string-pool interning index). That index is
  /// assigned in first-interned order, so it varies between builds/link orders
  /// and made GOTO output (e.g. goto2c's emitted C, --goto-functions-only
  /// dumps) reorder nondeterministically. Comparing the string makes iteration
  /// reproducible. function_map is small and string compares are cheap, so the
  /// O(log n) cost is negligible; nothing relies on the previous ordering.
  struct id_string_order
  {
    bool operator()(const irep_idt &a, const irep_idt &b) const
    {
      return a.as_string() < b.as_string();
    }
  };

  typedef std::map<irep_idt, goto_functiont, id_string_order> function_mapt;
  function_mapt function_map;

  // For coverage and multi-property
  // Store and pass the coverage data in incr/kind mode
  static std::unordered_set<std::string> reached_claims;
  static std::unordered_multiset<std::string> reached_mul_claims;

  static std::mutex reached_claims_mutex;
  static std::mutex reached_mul_claims_mutex;

  // Serialises clear_verified_claims_in_goto across parallel multi-property
  // claims, which may concurrently make_skip() the same assert instruction.
  // One shared lock replaces what used to be a per-instruction std::mutex
  // (40 bytes on every GOTO instruction in the program).
  static std::mutex clear_claims_mutex;

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

#ifndef ESBMC_GOTO2LEAN_H
#define ESBMC_GOTO2LEAN_H

#include <goto-programs/goto_functions.h>
#include <util/symtab/namespace.h>

#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

/// Translates the GOTO program reachable from an entry function into Lean 4
/// definitions in a weakest-precondition (continuation-passing) style: each
/// function `f(args)` becomes `f args (post : R → Prop) : Prop`, NONDET and
/// uninitialised declarations become `∀`, ASSUME becomes `→`, ASSERT becomes
/// `∧`, and each loop becomes a `coinductive_fixpoint` (greatest fixpoint, so
/// partial correctness: no assertion fails on any terminating execution).
///
/// Only a scalar fragment is supported (bool and fixed-width integers,
/// structured control flow, non-recursive calls, quantifiers in guards);
/// anything else raises goto2lean_unsupported.
class goto2leant
{
public:
  goto2leant(
    const namespacet &ns,
    const goto_functionst &goto_functions,
    const std::string &entry,
    bool int_mode = false,
    bool div_checked = true);

  std::string translate();

private:
  using instructiont = goto_programt::instructiont;

  struct vart
  {
    std::string name; // Lean identifier
    type2tc type;
    irep_idt id;
  };
  using scopet = std::vector<vart>;

  struct loopt
  {
    std::size_t head, exit;
    std::string fn;
    scopet vars;
  };

  struct functiont
  {
    std::string lean_name, source;
    std::vector<const instructiont *> insts;
    std::map<std::size_t, std::size_t> loop_last; // head -> last back edge
    std::map<const instructiont *, std::size_t> index;
    std::map<std::string, std::string> names; // ESBMC id -> Lean identifier
    std::set<std::string> used_names;
    type2tc ret_type;
    unsigned loop_count = 0;
  };

  const namespacet &ns;
  const goto_functionst &goto_functions;
  std::string entry;
  // --ir: integers are unbounded, so they become Lean `Int`, not `BitVec n`.
  bool int_mode;
  bool div_checked; // a zero divisor fails ESBMC's division-by-zero check
  std::map<std::string, int> divisor_signs; // constant divisor -> Int.sign

  std::vector<std::string> order;     // callees before callers
  std::set<std::string> global_names; // functions and loop definitions
  std::string entry_id;
  std::vector<vart> entry_params;
  const std::vector<vart> *in_scope = nullptr; // bindings value() may read
  std::vector<irep_idt> qbinders;
  std::map<std::string, std::string> fn_names;
  std::vector<std::string> aux_defs;
  bool has_loops = false;
  // a quantifier the generated proof script cannot introduce by itself
  bool manual_quantifiers = false;
  unsigned join_count = 0;

  functiont *cur = nullptr;

  void collect(const std::string &id, std::set<std::string> &on_stack);
  void check_startup_code() const;
  std::string translate_function(const std::string &id);
  std::string proof_script() const;

  std::string trans(
    std::size_t pc,
    std::size_t end,
    scopet scope,
    const loopt *loop,
    const std::string &on_end,
    unsigned ind);
  std::string trans_decl(
    std::size_t pc,
    std::size_t end,
    scopet &scope,
    const loopt *loop,
    const std::string &on_end,
    unsigned ind);
  std::string trans_assign(
    std::size_t pc,
    std::size_t end,
    scopet &scope,
    const loopt *loop,
    const std::string &on_end,
    unsigned ind);
  std::string trans_call(
    std::size_t pc,
    std::size_t end,
    scopet &scope,
    const loopt *loop,
    const std::string &on_end,
    unsigned ind);
  std::string trans_goto(
    std::size_t pc,
    std::size_t end,
    const scopet &scope,
    const loopt *loop,
    const std::string &on_end,
    unsigned ind);
  std::size_t branch_join(std::size_t pc, std::size_t t, std::size_t end) const;
  std::string trans_loop(
    std::size_t pc,
    std::size_t end,
    const scopet &scope,
    const loopt *outer,
    const std::string &on_end,
    unsigned ind);
  scopet live(const scopet &scope, std::size_t pc, const loopt *loop) const;
  bool assigned_before_use(std::size_t pc, std::size_t end, const irep_idt &id)
    const;
  std::string jump(
    std::size_t target,
    std::size_t end,
    const scopet &scope,
    const loopt *loop,
    const std::string &on_end,
    unsigned ind);

  std::string value_in(const scopet &scope, const expr2tc &e);
  std::string prop_in(const scopet &scope, const expr2tc &e);
  std::string asserted(const scopet &scope, const expr2tc &guard);
  std::string bind(scopet &scope, const expr2tc &sym);

  std::string value(const expr2tc &e);
  std::string symbol_value(const irep_idt &id);
  std::string int_literal(const expr2tc &e) const;
  std::string int_division(const expr2tc &e);
  std::string bv_division(const expr2tc &e);
  std::string bitwise(const expr2tc &e);
  std::string cast(const expr2tc &e);
  std::string relation(const expr2tc &e);
  std::string overflow(const expr2tc &e);
  std::string overflow_neg(const expr2tc &e);
  void reject_under_ir(const expr2tc &e) const;
  std::string infix(const expr2tc &a, const std::string &op, const expr2tc &b);
  std::string infix(const expr2tc &e, const std::string &op);
  std::string apply(const std::string &fn, const expr2tc &a, const expr2tc &b);

  std::string prop(const expr2tc &e);
  std::string connective(const expr2tc &e, const std::string &op);
  std::string prop_relation(const expr2tc &e);
  std::string quantifier(const expr2tc &e);
  std::string type(const type2tc &t);
  std::string var_name(const irep_idt &id);
  std::string source_name(const irep_idt &id) const;
  std::string zero(const type2tc &t) const;
  std::string fresh(const std::string &base);
  std::string ret_cont(const loopt *loop) const;
  static std::string args_of(const scopet &scope);
  std::string lambda(const scopet &scope, const std::string &body);

  [[noreturn]] void unsupported(
    const std::string &what,
    const locationt &loc = locationt()) const;
};

class goto2lean_unsupported : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

#endif

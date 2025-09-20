#pragma once

#include <goto-programs/abstract-interpretation/ai.h>

#include <util/symbol.h>
#include <map>
#include <util/algorithms.h>

/**
 * @brief Abstract domain to keep all variables used in asserts
 *
 * The domain is the list of symbols that may affect next statements
 *
 * - TOP --> depends on all symbols (not needed)
 * - BOTTOM --> empty dependencies
 * - Meet operator: union
 * - Transform operator checks whether the current statement affects
 *   any of the dependencies
 * - Backwards
 */

class slicer_domaint : public ai_domain_baset
{
public:

  slicer_domaint()
  {
  }

  virtual void transform(
    goto_programt::const_targett from,
    goto_programt::const_targett to,
    ai_baset &ai,
    const namespacet &ns) override;

  virtual void output(std::ostream &out) const override;

  virtual void make_bottom() override
  {
    dependencies.clear();
  }

  virtual void make_entry() override
  {
    dependencies.clear();
  };
  virtual void make_top() override{

  };

  virtual bool is_bottom() const override
  {
    return dependencies.size() == 0;
  }
  virtual bool is_top() const override
  {
    return false;
  }

  virtual bool ai_simplify(expr2tc &, const namespacet &) const override
  {
    return true;
  }

protected:
  bool join(const slicer_domaint &b);

public:
  bool merge(
    const slicer_domaint &b,
    goto_programt::const_targett,
    goto_programt::const_targett)
  {
    return join(b);
  }

  std::map<std::string, std::set<std::string>> dependencies;

protected:
  void assign(const expr2tc &assignment);
  void declaration(const expr2tc &decl);

private:
  bool should_skip_symbol(const std::string &symbol) const;
};

/**
 * Slicer for Goto Programs
 */
class goto_slicer : public goto_functions_algorithm
{
public:
  explicit goto_slicer(const namespacet &ns, bool base_case = true)
    : goto_functions_algorithm(true), ns(ns)
  {
    if(base_case)
      set_base_slicer();
    else
      set_forward_slicer();
  }

  /*
   * Set options for forward (forward step) mode
   */
  void set_forward_slicer();

  /*
   * Set options for base case (preprocessing) mode
   */
  void set_base_slicer();

  unsigned instructions_sliced = 0;
  unsigned loops_sliced = 0;

protected:
  virtual bool
  runOnFunction(std::pair<const dstring, goto_functiont> &F) override;
  virtual bool runOnLoop(loopst &loop, goto_programt &goto_program) override;
  virtual bool runOnProgram(goto_functionst &) override;
  virtual bool postProcessing(goto_functionst &) override;
  const namespacet &ns;

  bool sliceLoop(loopst &loop, goto_programt &goto_program);

  bool contains_global_var(std::set<std::string> symbols) const;

  /// Instructions that should contribute to the dependency of the slicer
  std::set<GOTO_INSTRUCTION_TYPE> dependency_instruction_type;

  /// Instructions types that can be sliced away (if possible)
  std::set<GOTO_INSTRUCTION_TYPE> sliceable_instruction_type;

  /// Flag to check whether we are slicing a forward condition. This is useful for determining whether we should slice while(1) or while(0) loops.
  bool forward_analysis = false;
  bool is_loop_empty(const loopst &loop) const;
  bool is_loop_affecting_assertions(const loopst &loop);
  bool is_trivial_loop(const loopst &loop) const;

  std::set<std::string> remaining_deps;
  ait<slicer_domaint> slicer;
  /// TODO: This will be needed until we have support for pointers
  bool slicer_failed = false;
  bool should_skip_function(const std::string &func) const;
};

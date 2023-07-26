
#pragma once

#include <goto-programs/abstract-interpretation/ai.h>
#include <pointer-analysis/value_set_analysis.h>
/**
 * @brief Abstract domain to obtain all available expressions
 *
 * The domain is the list of symbols that may affect next statements
 *
 * - TOP --> all expressions possible for the program (not needed)
 * - BOTTOM --> no expressions available
 * - Meet operator: see join function
 * - Transform operator: adds/remove available expressions recursively
 * - Flow-sensitive
 * - Context-insensitive
 */

class cse_domaint : public ai_domain_baset
{
public:
  cse_domaint()
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
    available_expressions.clear();
  }

  virtual void make_entry() override
  {
    available_expressions.clear();
  };
  virtual void make_top() override{

  };

  virtual bool is_bottom() const override
  {
    return available_expressions.size() == 0;
  }
  virtual bool is_top() const override
  {
    return false;
  }

  virtual bool ai_simplify(expr2tc &, const namespacet &) const override
  {
    return false;
  }

protected:
  bool join(const cse_domaint &b);

public:
  bool merge(
    const cse_domaint &b,
    goto_programt::const_targett,
    goto_programt::const_targett)
  {
    return join(b);
  }

  std::unordered_set<expr2tc, irep2_hash> available_expressions;

protected:
  void assign(const expr2tc &assignment, const goto_programt::const_targett &);
  void check_expression(const expr2tc &e);
  void havoc_expr(const expr2tc &e, const goto_programt::const_targett &);
  void havoc_symbol(const irep_idt &sym);
  bool remove_expr(const expr2tc &taint, const expr2tc &src) const;
  bool remove_expr(const irep_idt &sym, const expr2tc &src) const;

public:
  static std::unique_ptr<value_set_analysist> vsa;
};

#include <util/algorithms.h>
/**
 * Common Subexpression Elimination algorithm
 */
class goto_cse : public goto_functions_algorithm
{
public:
  explicit goto_cse(const namespacet &ns)
    : goto_functions_algorithm(true), ns(ns)
  {
  }

  virtual bool runOnProgram(goto_functionst &) override;
  virtual bool
  runOnFunction(std::pair<const dstring, goto_functiont> &F) override;

  unsigned threshold = 1;
  bool verbose_mode = true;

protected:
  ait<cse_domaint> available_expressions;
  const namespacet &ns;
  expr2tc obtain_max_sub_expr(const expr2tc &e, const cse_domaint &state) const;
  void replace_max_sub_expr(
    expr2tc &e,
    std::unordered_map<expr2tc, expr2tc, irep2_hash> &expr2symbol) const;
};

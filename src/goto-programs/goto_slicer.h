#pragma once

#include <goto-programs/ai.h>
#include <goto-programs/interval_template.h>
#include <util/ieee_float.h>
#include <irep2/irep2_utils.h>
#include <map>
#include <util/algorithms.h>


class slicer_domaint : public ai_domain_baset
{
public:
  // Trivial, conjunctive interval domain for both float
  // and integers. The categorization 'float' and 'integers'
  // is done by is_int and is_float.

  slicer_domaint()
  {
  }

  virtual void transform(
    goto_programt::const_targett from,
    goto_programt::const_targett to,
    ai_baset &ai,
    const namespacet &ns) override;

  virtual void output(std::ostream &out) const override;

  virtual void make_bottom() override {
    dependencies.clear();
  }

  virtual void make_entry() override {
    dependencies.clear();
  };
  virtual void make_top() override {

  };

  virtual bool is_bottom() const override {
    return dependencies.size() == 0;
  }
  virtual bool is_top() const override {return false; }

  virtual bool ai_simplify(expr2tc &condition, const namespacet &ns) const override {
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

protected:
  std::map<std::string, std::set<std::string>> dependencies;

  void assign(const expr2tc &assignment);
  void declaration(const expr2tc &decl);
};

class goto_slicer : public goto_functions_algorithm {
public:
  explicit goto_slicer(const namespacet &ns) : goto_functions_algorithm(true), ns(ns) {}
protected:
  virtual bool runOnFunction(std::pair<const dstring, goto_functiont> &F) override;
  //virtual bool runOnLoop(loopst &loop, goto_programt &goto_program);
  virtual bool runOnProgram(goto_functionst &) override;
  const namespacet &ns;
private:
  ait<slicer_domaint> sl;
};
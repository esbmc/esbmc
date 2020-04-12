//
// Created by Rafael Sá Menezes on 07/04/20.
//

#ifndef ESBMC_SSA_FLATTENER_H
#define ESBMC_SSA_FLATTENER_H

#include <cache/ssa_step_algorithm.h>
#include <cache/green/green_storage.h>

class ssa_flattener : public ssa_step_algorithm_hidden
{
public:
  explicit ssa_flattener(symex_target_equationt::SSA_stepst &steps)
    : ssa_step_algorithm_hidden(steps)
  {
    gs.load();
  }

  void dump()
  {
    for(const auto &[key, value] : items)
    {
      std::cout << "Guard: " << key << std::endl;
      for(const auto &expression : value)
      {
        std::cout << "\t" << expression << std::endl;
      }
    }
  }

  void dump_collisions()
  {
    for(const auto &[key, value] : items)
    {
      if(gs.get(value))
      {
        std::cout << "Found an UNSAT guard!!!\n";
      }
    }
  }

  std::unordered_map<std::string, std::set<expr_hash>> get_map()
  {
    return items;
  }

private:
  bool is_guard(const expr2tc &expr, std::string &name);
  green_storage gs;
  std::set<expr_hash> parse_guard(const expr2tc &expr);
  void parse_implication_guard(
    const expr2tc &expr,
    std::set<expr_hash> &inner_items);
  std::unordered_map<std::string, std::set<expr_hash>> items;

protected:
  void run_on_assert(symex_target_equationt::SSA_stept &step) override;
  void run_on_assignment(symex_target_equationt::SSA_stept &step) override;
  void run_on_assume(symex_target_equationt::SSA_stept &step) override
  {
  }

  void run_on_output(symex_target_equationt::SSA_stept &step) override
  {
  }
  void run_on_skip(symex_target_equationt::SSA_stept &step) override
  {
  }
  void run_on_renumber(symex_target_equationt::SSA_stept &step) override
  {
  }
};

#endif //ESBMC_SSA_FLATTENER_H

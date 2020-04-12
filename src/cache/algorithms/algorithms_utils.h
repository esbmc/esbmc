/*******************************************************************\
 Module: Algorithm Utilities

 Author: Rafael Sá Menezes

 Description:
  This file contains helper functions to manipulate SSA's and
  Expressions that are common between all algorithms.

  - Only static methods are available
\*******************************************************************/

#ifndef ESBMC_ALGORITHMS_UTILS_H
#define ESBMC_ALGORITHMS_UTILS_H

#include <util/irep2.h>
#include <util/irep2_expr.h>

// CLASSES

class ssa_algorithm_util
{
private:
  ssa_algorithm_util() = default;
};

class expr_algorithm_util
{
public:
  static std::string get_symbol_name(expr2tc &expr) noexcept
  {
    // This is meant to be used only in symbols
    assert(expr->expr_id != expr2t::expr_ids::symbol_id);
    std::shared_ptr<symbol_data> symbol;
    symbol = std::dynamic_pointer_cast<symbol_data>(expr);
    return symbol->get_symbol_name();
  }

private:
  expr_algorithm_util() = default;
};

#endif //ESBMC_ALGORITHMS_UTILS_H

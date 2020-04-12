/*******************************************************************\
 Module: Expr Algorithms Interface

 Author: Rafael Sá Menezes

 Date: April 2020
\*******************************************************************/

#ifndef ESBMC_EXPR_ALGORITHM_H
#define ESBMC_EXPR_ALGORITHM_H

#include <util/irep2.h>

/**
 * Base interface to run an algorithm in an expression
 * the algorithm may extract information or change the
 * expression at will.
 */
class expr_algorithm
{
public:
  explicit expr_algorithm(expr2tc &expr) : expr(expr)
  {
  }

  virtual ~expr_algorithm()
  {
  }
  virtual void run() = 0;

protected:
  expr2tc &expr;
};

#endif //ESBMC_EXPR_ALGORITHM_H

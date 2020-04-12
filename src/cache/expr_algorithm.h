//
// Created by Rafael Sá Menezes on 11/04/20.
//

#ifndef ESBMC_EXPR_ALGORITHM_H
#define ESBMC_EXPR_ALGORITHM_H

#include <util/irep2.h>

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

//
// Created by rafaelsa on 10/03/2020.
//

#ifndef ESBMC_CACHE_H
#define ESBMC_CACHE_H

#include <cache/ssa_step_algorithm.h>

class cache
{
public:
  explicit cache()
  {
  }
  virtual void run(symex_target_equationt::SSA_stepst &steps) = 0;
};

#endif //ESBMC_CACHE_H

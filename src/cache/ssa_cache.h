//
// Created by rafaelsa on 10/03/2020.
//

#ifndef ESBMC_SSA_CACHE_H
#define ESBMC_SSA_CACHE_H

#include <cache/ssa_step_algorithm.h>

class ssa_cache
{
public:
  explicit ssa_cache()
  {
  }
  virtual void run(symex_target_equationt::SSA_stepst &steps) = 0;
};

#endif //ESBMC_SSA_CACHE_H

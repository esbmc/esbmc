//
// Created by rafaelsa on 10/03/2020.
//

#ifndef ESBMC_GREEN_CACHE_H
#define ESBMC_GREEN_CACHE_H

#include <cache/ssa_cache.h>

/**
 *  A Green implementation for SSA steps
 *
 *  Based on the paper: "Green: Reducing, Reusing and Recycling Constraint in
 *  Program analysis"
 *
 */
class green_cache : public ssa_cache
{
public:
  green_cache();
  void run(symex_target_equationt::SSA_stepst &steps) override;
};

#endif //ESBMC_GREEN_CACHE_H

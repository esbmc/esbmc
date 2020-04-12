//
// Created by rafaelsa on 10/03/2020.
//

#ifndef ESBMC_GREEN_CACHE_H
#define ESBMC_GREEN_CACHE_H

#include <cache/cache.h>
#include "green_storage.h"

/**
 *  A Green implementation for SSA steps
 *
 *  Based on the paper: "Green: Reducing, Reusing and Recycling Constraint in
 *  Program analysis"
 *
 */
class green_cache : public cache
{
public:
  green_cache();
  void run(symex_target_equationt::SSA_stepst &steps) override;
  void succesful();

private:
  std::unordered_map<std::string, std::set<expr_hash>> items;
  void simplify();
};

#endif //ESBMC_GREEN_CACHE_H

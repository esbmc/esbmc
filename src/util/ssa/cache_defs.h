#pragma once
#include <irep2/irep2.h>
#include <unordered_map>
#include <vector>

// This header will prevent the dependency hell
// between irep2, message and config

/// Each claim without its path guard, mapped to the guard conjuncts it has
/// been asserted under.
using assert_db =
  std::unordered_map<expr2tc, std::vector<std::vector<expr2tc>>, irep2_hash>;

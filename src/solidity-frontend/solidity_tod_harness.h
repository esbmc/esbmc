#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

enum class TodHarnessMode
{
  Balance,
  Race,
};

/// Generate a TOD (Transaction Order Dependence) harness as compilable
/// Solidity source.
///
/// Race mode emits a self-describing harness body of shape:
///   C c1 = new C();
///   __ESOL_nondet_state_forward(c1);   // reach any reachable state S
///   C c2 = __ESOL_deep_copy(c1);       // c2 starts at same S
///   c1.fa(); c1.fb();                  // order 1
///   c2.fb(); c2.fa();                  // order 2
///   __tod_race_check(...);             // assert state equivalence
/// The __ESOL_* intrinsic stubs are emitted at file scope; ESBMC's
/// Solidity frontend intercepts both call sites and lowers them to
/// _ESBMC_state_forward_<C> / _ESBMC_clone_<C>.
///
/// Balance mode keeps the two-copy rename approach (V_C1 / V_C2) so
/// that ETH transfers on the distinct singleton dispatchers do not
/// alias.
std::string generate_tod_harness(
  const std::string &sol_source,
  const nlohmann::json &ast,
  const std::string &contract,
  const std::string &func_a,
  const std::string &func_b,
  TodHarnessMode mode);

/// Multi-pair variant of generate_tod_harness().
std::string generate_tod_harness_multi(
  const std::string &sol_source,
  const nlohmann::json &ast,
  const std::string &contract,
  const std::vector<std::pair<std::string, std::string>> &pairs,
  TodHarnessMode mode);

#pragma once

#include <nlohmann/json.hpp>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace solidity_tod
{

/// Virtual id for the contract's ETH balance (`address(this).balance`).
/// Real Solidity AST ids are positive; we use a negative sentinel so it
/// can flow through the same R/W set machinery as ordinary state vars
/// without a separate field.  Picked specifically to not collide with
/// the negative ids solc assigns to built-ins like `this` (-28),
/// `msg` (-15), etc.
constexpr int kBalanceId = -2;

/// Read/write footprint of a single function over the contract's
/// (public/private) state variables, identified by AST id.  Negative
/// ids are reserved for virtual tokens (currently only kBalanceId).
struct RWSet
{
  std::set<int> reads;
  std::set<int> writes;
};

/// Pair of externally callable function names with overlapping footprints.
/// `shared` lists the AST ids that triggered the candidacy (informational).
struct Pair
{
  std::string func_a;
  std::string func_b;
  std::set<int> shared;
};

/// Compute R/W sets for every FunctionDefinition (and modifier body) in
/// the given ContractDefinition node, then propagate footprints transitively
/// over the intra-contract internal call graph.  External calls and
/// `this.f()` self-external calls are conservatively ignored — they are out
/// of scope for the Tier-2 algorithm.  Returns a map keyed by function name.
///
/// When `ast` is non-null AND the contract declares a
/// `linearizedBaseContracts` chain, state vars and functions from every
/// reachable base contract are folded into the walk (subject to MRO
/// override rules on function-name collisions, and to visibility on
/// state vars — a `private` var in a base contract is not accessible
/// from a derived contract).
std::map<std::string, RWSet> compute_rw_sets(
  const nlohmann::json &contract_def,
  const nlohmann::json *ast = nullptr);

/// Mode filter for find_tod_candidates().
///   Any         — any non-empty overlap qualifies.
///   BalanceOnly — the shared footprint must include kBalanceId.
///   RaceOnly    — the shared footprint must include at least one
///                 non-balance state variable id.
enum class Mode
{
  Any,
  BalanceOnly,
  RaceOnly,
};

/// Find pairs of public/external functions whose footprints satisfy
///   W(f1) ∩ (R(f2) ∪ W(f2))  ∪  W(f2) ∩ (R(f1) ∪ W(f1))  ≠ ∅
/// Skips view/pure functions, the constructor, fallback and receive.
/// Pairs are returned sorted with `func_a < func_b` lexicographically.
///
/// When `ast` is non-null, inherited functions and state vars are
/// included via `linearizedBaseContracts` (same semantics as
/// `compute_rw_sets`).
std::vector<Pair> find_tod_candidates(
  const nlohmann::json &contract_def,
  Mode mode = Mode::Any,
  const nlohmann::json *ast = nullptr);

} // namespace solidity_tod

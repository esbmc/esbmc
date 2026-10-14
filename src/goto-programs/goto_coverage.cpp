#include <goto-programs/goto_coverage.h>
#include <util/config/focus_function.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_inline.h>
#include <util/arith/arith_tools.h>
#include <goto-programs/k_path_spanning.h>
#include <goto-programs/remove_no_op.h>
#include <irep2/irep2_utils.h>
#include <util/base/i2string.h>
#include <util/config/options.h>
#include <util/irep/std_types.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace
{
/// True when the guard is constant once folded, so the instruction is not a
/// branch at all: `if (1)` lowers to `IF !1 THEN GOTO`, and instrumenting both
/// directions of that leaves one probe unsatisfiable by construction, which
/// --dead-code-check reports as CWE-561 (#7267).
bool guard_folds_to_constant(const expr2tc &guard)
{
  expr2tc simplified = guard->simplify();
  const expr2tc &folded = is_nil_expr(simplified) ? guard : simplified;
  return is_true(folded) || is_false(folded);
}
} // namespace

// Defined later in this TU; used by solidity_path_coverage() for --contract
// scoping. Extracts the contract from a Solidity mangled id "sol:@C@<C>@F@...".
static std::string contract_of(const std::string &mangled_id);

size_t goto_coveraget::total_assert = 0;
size_t goto_coveraget::total_assert_ins = 0;
std::set<std::pair<std::string, std::string>> goto_coveraget::total_cond;
size_t goto_coveraget::total_branch = 0;
size_t goto_coveraget::total_func_branch = 0;
size_t goto_coveraget::total_kpath = 0;
size_t goto_coveraget::total_kpath_spanning = 0;
std::set<std::pair<std::string, std::string>>
  goto_coveraget::k_path_spanning_redundant;
std::set<std::pair<std::string, std::string>> goto_coveraget::all_claims;
std::map<std::pair<std::string, std::string>, std::string>
  goto_coveraget::claim_negation;
std::set<std::pair<std::string, std::string>> goto_coveraget::covered_set;
std::map<std::string, char> goto_coveraget::claim_outcome;
std::mutex goto_coveraget::claim_outcome_mutex;
std::set<std::pair<std::string, std::string>> goto_coveraget::revert_paths;
std::set<std::pair<std::string, std::string>>
  goto_coveraget::rollback_revert_paths;
std::set<std::pair<std::string, std::string>>
  goto_coveraget::undetermined_exit_paths;
std::set<std::pair<std::string, std::string>> goto_coveraget::normal_exit_paths;
std::map<std::pair<std::string, std::string>, std::string>
  goto_coveraget::named_obstacle_paths;
std::map<std::pair<std::string, std::string>, std::string>
  goto_coveraget::truncation_weakened;
std::map<std::string, std::vector<std::string>>
  goto_coveraget::degraded_call_sites;
std::map<std::string, goto_coveraget::path_ce_t> goto_coveraget::path_ce;
std::map<std::string, std::vector<goto_coveraget::path_ce_t>>
  goto_coveraget::path_ce_all;
std::map<std::string, goto_coveraget::path_ce_t>
  goto_coveraget::path_covered_payload;
std::map<std::pair<std::string, std::string>, uint64_t>
  goto_coveraget::path_decision_depth;
std::map<std::string, std::vector<goto_coveraget::path_decisiont>>
  goto_coveraget::path_decision_table;
std::map<std::string, std::map<uint64_t, uint32_t>>
  goto_coveraget::path_decision_index;
std::map<std::string, std::vector<std::string>>
  goto_coveraget::path_event_table;
std::map<std::string, std::map<uint64_t, std::map<uint32_t, uint32_t>>>
  goto_coveraget::path_event_index;
std::string goto_coveraget::covered_set_outpath;
std::set<std::string> goto_coveraget::path_covered_ids;
std::map<std::pair<std::string, std::string>, std::string>
  goto_coveraget::path_stable_id;
std::string goto_coveraget::path_covered_outpath;
std::map<std::string, std::string> goto_coveraget::units_not_entered;
bool goto_coveraget::path_cov_certify_mode = false;
std::vector<std::string> goto_coveraget::path_cov_certify_box_names;
std::vector<std::array<std::string, 3>> goto_coveraget::path_cov_certify_box;
std::map<std::string, std::string> goto_coveraget::path_cov_certify_ce;
std::pair<std::string, std::string>
  goto_coveraget::path_cov_certify_nonvacuous_key;
std::vector<std::pair<std::string, std::string>>
  goto_coveraget::path_cov_certify_exit_keys;
std::map<std::string, std::vector<std::string>>
  goto_coveraget::path_cov_certify_holes;
bool goto_coveraget::path_cov_outer_box_mode = false;
std::vector<goto_coveraget::outer_box_probet>
  goto_coveraget::path_cov_outer_box_probes;
std::vector<std::pair<uint64_t, uint64_t>>
  goto_coveraget::path_cov_outer_box_paths;
std::map<std::pair<uint64_t, std::string>, std::string>
  goto_coveraget::path_cov_outer_box_ce;
std::map<std::string, std::pair<std::string, std::string>>
  goto_coveraget::path_cov_outer_box_type_range;
std::vector<std::pair<std::string, std::string>>
  goto_coveraget::path_cov_outer_box_pins;
std::string goto_coveraget::path_cov_outer_box_obstacle;
std::map<std::string, std::string> goto_coveraget::path_cov_refused_coords;
bool goto_coveraget::path_cov_assert_mode = false;
std::vector<goto_coveraget::assert_candidatet>
  goto_coveraget::path_cov_assert_candidates;
std::pair<std::string, std::string>
  goto_coveraget::path_cov_assert_nonvacuous_key;
std::string goto_coveraget::path_cov_fingerprint;
std::atomic<bool> goto_coveraget::branch_cov_active{false};
std::atomic<size_t> goto_coveraget::total_branch_atomic{0};
std::atomic<bool> goto_coveraget::covered_set_mode{false};
std::atomic<size_t> goto_coveraget::live_reached{0};
std::atomic<size_t> goto_coveraget::covered_run{0};
std::atomic<size_t> goto_coveraget::live_decided{0};
std::atomic<size_t> goto_coveraget::claims_total_atomic{0};
std::string goto_coveraget::path_cov_partial_reason;
std::set<std::string> goto_coveraget::claims_in_solve_loop;
size_t goto_coveraget::claim_budget_seconds = 0;
std::atomic<size_t> goto_coveraget::claim_budget_exceeded{0};
std::string goto_coveraget::claim_budget_mechanism;
std::atomic<bool> goto_coveraget::path_cov_active{false};
std::atomic<size_t> goto_coveraget::total_paths_atomic{0};
std::atomic<size_t> goto_coveraget::live_F{0};

void goto_coveraget::write_covered_set_atomic()
{
  if (covered_set_outpath.empty())
    return;
  nlohmann::json out;
  out["version"] = 1;
  out["covered"] = nlohmann::json::array();
  for (const auto &[cond, loc] : covered_set)
    out["covered"].push_back({{"cond", cond}, {"loc", loc}});
  const std::string tmp = covered_set_outpath + ".tmp";
  {
    std::ofstream f(tmp);
    if (!f)
    {
      log_warning("coverage-covered-set: cannot write {}", tmp);
      return;
    }
    f << out.dump(2) << "\n";
  }
  // Atomic publish: a kill between the two writes leaves the previous
  // valid file intact (never a truncated/corrupt covered-set).
  if (std::rename(tmp.c_str(), covered_set_outpath.c_str()) != 0)
    log_warning(
      "coverage-covered-set: atomic rename to {} failed", covered_set_outpath);
}

// FNV-1a 64. Used only to name things (path ids, fingerprints), never to make
// a soundness decision: a collision would merge two path ids, so the space is
// kept at 64 bits and the id is printed in full hex so a human can compare it.
static uint64_t fnv1a(const std::string &s, uint64_t h = 1469598103934665603ULL)
{
  for (unsigned char c : s)
  {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

static std::string hex64(uint64_t v)
{
  static const char *d = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i, v >>= 4)
    out[i] = d[v & 0xF];
  return out;
}

// ---- The CE payload <-> JSON round trip for the cross-run covered set ----
//
// Deliberately field-by-field rather than a blanket dump: every field that is
// NOT written here is a field a carried-over `F` will be missing, and the
// report's readers (the Foundry emitter, the certify audit, the stage-2 ladder)
// each consume a different subset. The pairs are written as ORDERED ARRAYS, not
// as JSON objects, because `inputs` is a sequence with a meaning -- an object
// would silently re-sort it and lose the call-argument order.
static nlohmann::json
pairs_to_json(const std::vector<std::pair<std::string, std::string>> &v)
{
  nlohmann::json a = nlohmann::json::array();
  for (const auto &[n, val] : v)
    a.push_back({{"name", n}, {"value", val}});
  return a;
}

static std::vector<std::pair<std::string, std::string>>
pairs_from_json(const nlohmann::json &a)
{
  std::vector<std::pair<std::string, std::string>> v;
  if (!a.is_array())
    return v;
  for (const auto &e : a)
    v.emplace_back(
      e.value("name", std::string()), e.value("value", std::string()));
  return v;
}

static nlohmann::json path_ce_to_json(
  const goto_coveraget::path_ce_t &ce,
  const std::string &claim_msg,
  const std::string &claim_loc)
{
  nlohmann::json j;
  // The claim this payload belongs to. Not needed to READ the payload back --
  // the stable id is the key -- but it is what makes the file diagnosable by a
  // human, and a payload filed under the wrong path is the one error this whole
  // content-addressed scheme exists to prevent.
  j["claim"] = claim_msg;
  j["loc"] = claim_loc;
  j["inputs"] = pairs_to_json(ce.inputs);
  j["env"] = pairs_to_json(ce.env);
  j["extcall_returns"] = pairs_to_json(ce.extcall_returns);
  j["entry_storage"] = pairs_to_json(ce.entry_storage);
  j["final_state"] = pairs_to_json(ce.final_state);
  j["state_written_unrendered"] = ce.state_written_unrendered;
  // BOTH halves, and both unconditionally. A carried-over `F` that lost its
  // return value would be indistinguishable from a unit that returns nothing --
  // the same two-ledger split this file already pays for elsewhere, one field
  // later.
  j["return_value"] = ce.return_value;
  j["return_value_known"] = ce.return_value_known;
  j["entry_storage_known"] = ce.entry_storage_known;
  j["dropped_internal"] = ce.dropped_internal;
  j["sliced"] = ce.sliced;
  j["compact_trace"] = ce.compact_trace;
  j["payload_symbols_protected"] = ce.payload_symbols_protected;
  j["scoped_to_claim"] = ce.scoped_to_claim;
  j["revert_pre_rollback"] = ce.revert_pre_rollback;
  return j;
}

static goto_coveraget::path_ce_t path_ce_from_json(const nlohmann::json &j)
{
  goto_coveraget::path_ce_t ce;
  ce.inputs = pairs_from_json(j.value("inputs", nlohmann::json::array()));
  ce.env = pairs_from_json(j.value("env", nlohmann::json::array()));
  ce.extcall_returns =
    pairs_from_json(j.value("extcall_returns", nlohmann::json::array()));
  ce.entry_storage =
    pairs_from_json(j.value("entry_storage", nlohmann::json::array()));
  ce.final_state =
    pairs_from_json(j.value("final_state", nlohmann::json::array()));
  for (const auto &s :
       j.value("state_written_unrendered", nlohmann::json::array()))
    ce.state_written_unrendered.push_back(s.get<std::string>());
  ce.entry_storage_known = j.value("entry_storage_known", false);
  ce.return_value = j.value("return_value", std::string());
  // Defaults FALSE: a file written by a build that did not persist this field
  // must come back "unknown", never "this unit returns nothing".
  ce.return_value_known = j.value("return_value_known", false);
  ce.dropped_internal = j.value("dropped_internal", (size_t)0);
  ce.sliced = j.value("sliced", true);
  ce.compact_trace = j.value("compact_trace", true);
  ce.payload_symbols_protected = j.value("payload_symbols_protected", false);
  ce.scoped_to_claim = j.value("scoped_to_claim", false);
  ce.revert_pre_rollback = j.value("revert_pre_rollback", false);
  return ce;
}

std::string goto_coveraget::path_ce_journal_path;

void goto_coveraget::write_path_ce_journal_atomic(
  const std::string &when,
  bool complete)
{
  if (path_ce_journal_path.empty())
    return;
  const size_t claims_decided = live_decided.load(std::memory_order_relaxed);
  const size_t claims_total =
    claims_total_atomic.load(std::memory_order_relaxed);
  nlohmann::json out;
  out["version"] = PATH_COVERED_SET_VERSION;
  out["kind"] = "solidity-complete-path-ce-journal";
  out["fingerprint"] = path_cov_fingerprint;
  // THE FIRST FIELD A READER MUST SEE. This file is a live journal, not a
  // report: on every write but the last it describes a run that has not
  // finished, and a consumer that read it as a finished report would deflate
  // every numerator it computed from it.
  out["complete"] = complete;
  out["partial"] = !complete;
  out["claims_decided"] = claims_decided;
  out["claims_total"] = claims_total;
  out["witnesses"] = nlohmann::json::object();
  size_t written = 0;
  {
    std::lock_guard lock(claim_outcome_mutex);
    for (const auto &[sig, ce] : path_ce)
    {
      auto o = claim_outcome.find(sig);
      if (o == claim_outcome.end() || o->second != 'F')
        continue;
      const auto tab = sig.rfind('\t');
      const std::string msg =
        tab == std::string::npos ? sig : sig.substr(0, tab);
      const std::string loc =
        tab == std::string::npos ? std::string() : sig.substr(tab + 1);
      nlohmann::json e = path_ce_to_json(ce, msg, loc);
      // The stable path id when there is one, so a consumer can join this file
      // against a covered set or against a later round's report. Absent rather
      // than faked when the id was not recorded (the stage-2/3 modes do not
      // populate path_stable_id at all).
      auto sid = path_stable_id.find({msg, loc});
      if (sid != path_stable_id.end())
        e["path_id_stable"] = sid->second;
      // Under --all-witnesses a path has several payloads, and a journal that
      // kept only the first would lose exactly what that flag was turned on to
      // obtain -- on the run that most needs the journal, the one that dies.
      // The count is emitted in both cases so "this path has one witness" and
      // "this path had several and we kept one" can never look alike.
      auto wa = path_ce_all.find(sig);
      const size_t nw = wa == path_ce_all.end() ? 1 : wa->second.size();
      e["witness_count"] = nw;
      if (nw > 1)
      {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto &w : wa->second)
          arr.push_back(path_ce_to_json(w, msg, loc));
        e["witnesses"] = arr;
      }
      out["witnesses"][sig] = e;
      ++written;
    }
  }
  const std::string tmp = path_ce_journal_path + ".tmp";
  {
    std::ofstream f(tmp);
    if (!f)
    {
      log_warning("path-cov CE journal: cannot write {}", tmp);
      return;
    }
    f << out.dump(2) << "\n";
  }
  if (std::rename(tmp.c_str(), path_ce_journal_path.c_str()) != 0)
  {
    log_warning(
      "path-cov CE journal: atomic rename to {} failed", path_ce_journal_path);
    return;
  }

  // Read back off the disk, for the same reason the covered-set census is: the
  // claim being made is "the payload is in a file", and only a file that was
  // re-opened supports it. `written` is kept beside the disk count so a
  // disagreement between what was serialised and what landed is visible rather
  // than averaged away.
  size_t on_disk = 0, with_inputs = 0;
  bool readback_ok = false;
  {
    std::ifstream rb(path_ce_journal_path);
    if (rb)
    {
      try
      {
        nlohmann::json v;
        rb >> v;
        const nlohmann::json w = v.value("witnesses", nlohmann::json::object());
        on_disk = w.size();
        for (auto it = w.begin(); it != w.end(); ++it)
          if (it.value().contains("inputs") && !it.value()["inputs"].empty())
            ++with_inputs;
        readback_ok = true;
      }
      catch (const std::exception &)
      {
        readback_ok = false;
      }
    }
  }
  if (!readback_ok)
  {
    log_error(
      "--solidity-path-coverage: CE journal {} was published but could not be "
      "read back. The counterexample payload is the deliverable a dying run is "
      "supposed to keep; an unreadable journal means it was NOT kept",
      path_ce_journal_path);
    return;
  }
  if (on_disk != written)
  {
    log_error(
      "--solidity-path-coverage: CE journal {} holds {} witness(es) but {} "
      "were serialised. These must be equal; a journal that silently loses "
      "entries is the failure it exists to prevent",
      path_ce_journal_path,
      on_disk,
      written);
    return;
  }
  log_success(
    "--solidity-path-coverage: CE journal {} updated {}: {} witnessed path(s) "
    "on disk, {} with non-empty inputs (complete={})",
    path_ce_journal_path,
    when,
    on_disk,
    with_inputs,
    complete ? "true" : "false");
}

const goto_coveraget::path_ce_t *goto_coveraget::path_payload_earlier(
  const std::pair<std::string, std::string> &claim_key)
{
  auto it = path_stable_id.find(claim_key);
  if (it == path_stable_id.end())
    return nullptr;
  auto p = path_covered_payload.find(it->second);
  return p == path_covered_payload.end() ? nullptr : &p->second;
}

void goto_coveraget::write_path_covered_set_atomic(const std::string &when)
{
  if (path_covered_outpath.empty())
    return;
  nlohmann::json out;
  out["version"] = PATH_COVERED_SET_VERSION;
  out["kind"] = "solidity-complete-path";
  // The fingerprint is written so the NEXT run can refuse this file outright.
  out["fingerprint"] = path_cov_fingerprint;
  out["covered"] = nlohmann::json::array();
  out["payloads"] = nlohmann::json::object();
  // A path counts as covered only when a counterexample was actually obtained
  // ('F'). 'P' (no witness at this bound) and 'U' (undecided) are not evidence
  // of anything and must never enter a cross-run cover.
  std::set<std::string> ids = path_covered_ids;
  // Loaded payloads are carried forward FIRST, then this run's overwrite them.
  // A path skipped this round (already witnessed) has no entry in path_ce, so
  // without this line the very first write-back of round 2 would delete every
  // payload round 1 persisted -- the exact loss this map exists to prevent,
  // arriving one round later.
  std::map<std::string, nlohmann::json> payloads;
  for (const auto &[id, ce] : path_covered_payload)
    payloads[id] = path_ce_to_json(ce, "", "");
  {
    std::lock_guard lock(claim_outcome_mutex);
    for (const auto &[claim, id] : path_stable_id)
    {
      const std::string sig = claim.first + "\t" + claim.second;
      auto o = claim_outcome.find(sig);
      if (o == claim_outcome.end() || o->second != 'F')
        continue;
      ids.insert(id);
      auto c = path_ce.find(sig);
      if (c != path_ce.end())
        payloads[id] = path_ce_to_json(c->second, claim.first, claim.second);
    }
  }
  for (const auto &id : ids)
    out["covered"].push_back(id);
  for (const auto &[id, j] : payloads)
    out["payloads"][id] = j;
  const std::string tmp = path_covered_outpath + ".tmp";
  {
    std::ofstream f(tmp);
    if (!f)
    {
      log_warning("coverage-covered-set: cannot write {}", tmp);
      return;
    }
    f << out.dump(2) << "\n";
  }
  if (std::rename(tmp.c_str(), path_covered_outpath.c_str()) != 0)
  {
    log_warning(
      "coverage-covered-set: atomic rename to {} failed", path_covered_outpath);
    return;
  }

  // ---- THE CENSUS IS READ BACK OFF THE DISK ----
  //
  // Not from `ids`/`payloads` above. A count taken from the producer's own
  // in-memory state is a statement about what the producer intended, and this
  // very function spent an unknown period never being called at all while every
  // number around it looked right. Re-opening the file it just published makes
  // "the payload is on disk" and "the line says so" the same statement.
  size_t on_disk_ids = 0, on_disk_payloads = 0, on_disk_with_inputs = 0;
  bool readback_ok = false;
  {
    std::ifstream rb(path_covered_outpath);
    if (rb)
    {
      try
      {
        nlohmann::json v;
        rb >> v;
        on_disk_ids = v.value("covered", nlohmann::json::array()).size();
        const nlohmann::json p = v.value("payloads", nlohmann::json::object());
        on_disk_payloads = p.size();
        for (auto it = p.begin(); it != p.end(); ++it)
          if (it.value().contains("inputs") && !it.value()["inputs"].empty())
            ++on_disk_with_inputs;
        readback_ok = true;
      }
      catch (const std::exception &)
      {
        readback_ok = false;
      }
    }
  }
  if (!readback_ok)
  {
    log_error(
      "--solidity-path-coverage: covered-set {} was published but could not be "
      "read back. The counterexample payload is the deliverable a dying run is "
      "supposed to keep; an unreadable file means it was NOT kept, and saying "
      "nothing here would leave a lost witness looking exactly like a saved "
      "one",
      path_covered_outpath);
    return;
  }
  log_success(
    "--solidity-path-coverage: covered-set persisted {}: {} path(s) on disk, "
    "{} with CE payload, {} with non-empty inputs ({})",
    when.empty() ? std::string("at run end") : when,
    on_disk_ids,
    on_disk_payloads,
    on_disk_with_inputs,
    path_covered_outpath);
}

bool goto_coveraget::path_witnessed_earlier(
  const std::pair<std::string, std::string> &claim_key)
{
  if (path_covered_ids.empty())
    return false;
  auto it = path_stable_id.find(claim_key);
  return it != path_stable_id.end() && path_covered_ids.count(it->second) != 0;
}

const std::vector<std::string> &goto_coveraget::path_u_reason_tokens()
{
  // Report order == classification priority, so the printed line reads in the
  // same order the decision is made. `unit-not-entered` sits second on purpose;
  // see path_u_reason_token.
  // `run-died-before-solving` sits LAST and is the only token that is a fact
  // about the RUN rather than about the path. It splits what used to be one
  // bucket: `not-solved-this-run` means the claim was simplified away at symex
  // time and never reached `assertion()`, which is a property of the claim and
  // is the same on every re-run; this one means the process stopped issuing
  // jobs, which says nothing about the claim at all. Folding them together made
  // a report unable to explain the very thing it was reporting -- a partial run
  // would show a large `not-solved-this-run` count that reads as "the
  // simplifier took them", when in fact nobody asked.
  static const std::vector<std::string> tokens = {
    "named-obstacle",
    "unit-not-entered",
    "bounded-holds",
    "solver-unknown",
    "not-solved-this-run",
    "run-died-before-solving",
    "claim-budget-exceeded"};
  return tokens;
}

std::string goto_coveraget::path_u_reason_token(
  const std::pair<std::string, std::string> &claim_key)
{
  // Most specific first. A disqualified unit's path is not an "unknown" that
  // better solving could resolve — no verdict can put it back in play — so the
  // obstacle wins over whatever the solver happened to answer.
  if (named_obstacle_paths.count(claim_key) != 0)
    return "named-obstacle";

  // SECOND, and above both verdict-derived tokens on purpose: when the harness
  // never entered the unit, no classification OF THE PATH means anything. A
  // vacuously-holding claim answers 'P' and would be filed "no witness within
  // the bound"; a claim that was never generated has no verdict and would be
  // filed "not solved this run". Both are statements about the path, and both
  // are strictly less informative than the truth, which is that nothing ran.
  {
    const size_t p = claim_key.first.rfind(":path:");
    const std::string unit =
      p == std::string::npos ? claim_key.first : claim_key.first.substr(0, p);
    if (units_not_entered.count(unit) != 0)
      return "unit-not-entered";
  }

  char v = 0;
  {
    std::lock_guard lock(claim_outcome_mutex);
    auto it = claim_outcome.find(claim_key.first + "\t" + claim_key.second);
    if (it != claim_outcome.end())
      v = it->second;
  }
  switch (v)
  {
  case 'P':
    // Proven at THIS exploration. Deliberately not called "unreachable": see
    // path_cov_can_prove_unreachable() in bmc.cpp.
    return "bounded-holds";
  case 'U':
    return "solver-unknown";
  case 'B':
    // THE QUERY WAS ABANDONED, and this is deliberately not one of the three
    // tokens it could plausibly be folded into. `solver-unknown` is the solver
    // ANSWERING "I do not know" -- it looked and gave up, which is information.
    // `bounded-holds` is it answering "no witness at this exploration".
    // `not-solved-this-run` is never having asked. Here we asked, the solver
    // was still working, and WE stopped it: nothing whatsoever is known about
    // this path, and unlike the other three the fix is a bigger
    // --path-cov-claim- timeout rather than a different bound, a different
    // query or nothing at all. Same rule that gave the truncation case
    // `UNDECIDED-TRUNCATED` its own word: a verdict that cannot be trusted has
    // to be a distinct one a driver can branch on.
    return "claim-budget-exceeded";
  case 0:
    // Instrumented, never decided -- and there are TWO ways to be in this state
    // that a reader must not have to guess between.
    //
    // The claim was QUEUED and the job loop stopped before reaching it. That is
    // a fact about the run, it is not reproducible from the claim, and
    // re-running with a bigger budget is the fix.
    //
    // The claim was NEVER QUEUED: the simplifier folded it to `true` at symex
    // time and it never reached `assertion()`. That is a fact about the claim,
    // identical on every re-run, and no budget changes it.
    //
    // Same cell, opposite meanings, opposite next actions -- and the membership
    // test is what separates them. Testing only `path_cov_partial_reason` would
    // sweep BOTH into the first bucket on any partial run: measured on aqua,
    // 1826 paths reported as lost to the death when ~901 were.
    if (
      !path_cov_partial_reason.empty() &&
      claims_in_solve_loop.count(claim_key.first))
      return "run-died-before-solving";
    return "not-solved-this-run";
  default:
    // THE ABORT PATH IS THE DEFAULT, deliberately. 'F' means witnessed, hence
    // not a U at all, so reaching here means the caller's F/I/U split and this
    // classification disagree — the very defect this function exists to
    // surface. Returning "" makes the caller hard-fail.
    //
    // DO NOT map `default` to a token. The claim that these tokens partition
    // the possible states rests on there being no catch-all: with one, a fifth
    // verdict value would be quietly absorbed, the abort would become dead
    // code, and the whole invariant would read as passing. A new verdict value
    // gets its own `case`.
    return std::string();
  }
}

void goto_coveraget::audit_certify_witness(bool ce_payload_requested)
{
  if (!path_cov_certify_mode)
    return;
  // No harvest was requested, so nothing is owed. See the header comment: this
  // narrowing came from the audit's first real run, where it accused a
  // perfectly correct `--result-only` run of losing a witness that was never
  // collected in the first place.
  if (!ce_payload_requested)
    return;

  // Only a coordinate that is NOT an EVM environment value is expected in
  // `inputs`; a box bounding only `msg.value` legitimately leaves it empty.
  bool box_has_argument = false;
  for (const auto &n : path_cov_certify_box_names)
    if (
      n.rfind("msg.", 0) != 0 && n.rfind("tx.", 0) != 0 &&
      n.rfind("block.", 0) != 0)
      box_has_argument = true;
  if (!box_has_argument)
    return;

  // The NON-VACUITY WITNESS is refuted on every SUCCESSFUL certification, and
  // it is not a refutation of the box -- it is the proof that the box is not
  // empty. Nothing is owed for it: there is no escaping input to name and no
  // side to cut on, and demanding one would abort every certification that
  // works. This is the same narrowing as `ce_payload_requested` above, and it
  // has to be applied at BOTH loops below: the audit's own hard failure and the
  // shrink suggestion, which would otherwise propose cutting the box on a
  // claim that is not asking for a cut.
  auto is_nonvacuity_claim = [](const std::pair<std::string, std::string> &k) {
    return !path_cov_certify_nonvacuous_key.first.empty() &&
           k == path_cov_certify_nonvacuous_key;
  };

  // ---- THE AUDIT AND THE SHRINK SUGGESTION MUST LOOK IN THE SAME PLACES ----
  //
  // This check used to be `ce->second.inputs.empty()` -- ONE of the three maps
  // a witness can live in. `witness_of` below looks in `inputs`, `env` AND
  // `entry_storage`, with the harvest's name normalisation (`msg.sender` is
  // stored `msg_sender`, `state.bal` is stored `bal`). So a refutation whose
  // witness is an ENVIRONMENT coordinate was found by the suggestion printer
  // and NOT by the audit, and the audit aborted the run.
  //
  // MEASURED on FarmingPool.exit with `--env-coord msg.sender`, one run, in
  // this order:
  //     PUNCH SUGGESTION  ... add msg.sender != 219152383 to the box's `holes`
  //     SHRINK SUGGESTION ... retry with msg.sender in [1, 219152382]
  //     ERROR: INTERNAL DEFECT -- 1 certification claim(s) were REFUTED but
  //            carry no witness input: ...exit#6694:path:8105#exit0
  //     SIGABRT
  // 36 of that unit's 37 paths died this way, and the run threw away the usable
  // cut it had just printed two lines above.
  //
  // THE GATE IS NOT WEAKENED. What is owed is a witness naming a coordinate the
  // BOX ACTUALLY BOUNDS -- a refutation that names nothing the box constrains
  // is still unusable and still aborts. Only the set of places searched
  // changes, so that the audit and the shrink agree about what "has a witness"
  // means.
  auto names_a_bounded_coordinate = [&](const path_ce_t &c) {
    for (const auto &coord : path_cov_certify_box_names)
    {
      std::string env = coord, bare = coord;
      for (auto &ch : env)
        if (ch == '.')
          ch = '_';
      if (coord.rfind("state.", 0) == 0)
        bare = coord.substr(6);
      for (const auto &[n, val] : c.inputs)
        if (n == coord || n == bare)
          return true;
      for (const auto &[n, val] : c.env)
        if (n == env || n == coord)
          return true;
      for (const auto &[n, val] : c.entry_storage)
        if (n == bare || n == coord)
          return true;
      // A struct parameter is stored ONCE under its base name, so `p.field` is
      // named by an entry keyed `p`. Same allowance witness_of makes below; not
      // making it here would re-create the split this comment is about, one
      // coordinate kind over.
      const size_t d = coord.find('.');
      if (d != std::string::npos)
      {
        const std::string b0 = coord.substr(0, d);
        for (const auto &[n, val] : c.inputs)
          if (n == b0)
            return true;
      }
    }
    return false;
  };

  std::vector<std::string> witnessless;
  {
    std::lock_guard lock(claim_outcome_mutex);
    for (const auto &key : all_claims)
    {
      if (is_nonvacuity_claim(key))
        continue;
      const std::string sig = key.first + "\t" + key.second;
      auto v = claim_outcome.find(sig);
      if (v == claim_outcome.end() || v->second != 'F')
        continue; // not a refutation: nothing is owed
      auto ce = path_ce.find(sig);
      if (ce == path_ce.end() || !names_a_bounded_coordinate(ce->second))
        witnessless.push_back(key.first);
    }
  }
  // ---- Turn the refutation into the NEXT BOX, not just a verdict ----
  //
  // The witness is an input inside the box that leaves the path, so the box has
  // to be cut on the witness's side. Which side that is comes from the path's
  // own counterexample: it is a known member of the domain, so the cut keeps it
  // and excludes the witness. That is a LANDING POINT, not a bisection — the
  // shrink goes straight to the witness rather than halving blindly, which is
  // what the withdrawn widening search did.
  //
  // Only suggested, never applied: the tool measures, the driver decides. And
  // it is suggested per coordinate with the least-loss cut chosen, mirroring
  // the subtraction's greedy rule.
  if (!path_cov_certify_ce.empty() && !path_cov_certify_box.empty())
  {
    std::lock_guard lock(claim_outcome_mutex);
    for (const auto &key : all_claims)
    {
      if (is_nonvacuity_claim(key))
        continue;
      const std::string sig = key.first + "\t" + key.second;
      auto v = claim_outcome.find(sig);
      if (v == claim_outcome.end() || v->second != 'F')
        continue;
      auto ce = path_ce.find(sig);
      if (ce == path_ce.end())
        continue;
      // Find the witness's value on a coordinate. The harvest keys parameters
      // by their base name, the environment by `msg_value`-style names, and
      // entry storage by the bare field name — so a coordinate written
      // `state.bal` has to be looked up as `bal`. Reported as "not named" when
      // absent rather than skipped, because a coordinate the witness does not
      // mention is exactly a coordinate the shrink cannot use.
      auto witness_of =
        [&](const std::string &coord, std::string &out) -> bool {
        std::string env = coord, bare = coord;
        for (auto &ch : env)
          if (ch == '.')
            ch = '_';
        if (coord.rfind("state.", 0) == 0)
          bare = coord.substr(6);
        for (const auto &[n, val] : ce->second.inputs)
          if (n == coord || n == bare)
          {
            out = val;
            return true;
          }
        for (const auto &[n, val] : ce->second.env)
          if (n == env || n == coord)
          {
            out = val;
            return true;
          }
        for (const auto &[n, val] : ce->second.entry_storage)
          if (n == bare || n == coord)
          {
            out = val;
            return true;
          }
        // ---- STRUCT FIELDS: `immutables.taker` under the key `immutables`
        // ----
        //
        // The harvest stores a struct parameter ONCE, under its base name, as a
        // pretty-printed aggregate. So a `param.field` coordinate matched
        // nothing above, `any_named` stayed false, and this function emitted
        // NEITHER the shrink suggestion NOR the "no single-coordinate shrink"
        // note -- which the driver then reported as "refuted with no
        // single-coordinate cut available".
        //
        // That reading was WRONG in a way that matters: it attributed to
        // proposition 11 (a genuinely multi-dimensional corner) what was only a
        // name that did not round-trip. MEASURED on EscrowSrc.withdraw enc=6,
        // whose witness differs on exactly ONE coordinate with an obviously
        // legal cut: the run printed a bare VERIFICATION FAILED and no
        // suggestion at all.
        const size_t d = coord.find('.');
        if (d != std::string::npos)
        {
          const std::string b0 = coord.substr(0, d), f0 = coord.substr(d + 1);
          for (const auto &[n, val] : ce->second.inputs)
          {
            if (n != b0 || val.empty() || val[0] != '{')
              continue;
            // Depth-1 `.field=value` only, matching the driver's decomposition
            // exactly: a nested aggregate's members belong to their own dotted
            // name, and lifting one here would answer about a different
            // quantity than the coordinate names.
            int depth = 0;
            for (size_t i = 0; i < val.size(); ++i)
            {
              if (val[i] == '{')
                ++depth;
              else if (val[i] == '}')
                --depth;
              else if (val[i] == '.' && depth == 1)
              {
                const size_t eq = val.find('=', i);
                if (eq == std::string::npos)
                  break;
                if (val.substr(i + 1, eq - i - 1) == f0)
                {
                  size_t k = eq + 1;
                  while (k < val.size() && val[k] != ',' && val[k] != '}')
                    ++k;
                  std::string v = val.substr(eq + 1, k - eq - 1);
                  while (!v.empty() && v.front() == ' ')
                    v.erase(v.begin());
                  while (!v.empty() && v.back() == ' ')
                    v.pop_back();
                  // A nested aggregate or a non-value is not a witness value:
                  // leaving it unmatched keeps the honest "could not look it
                  // up" rather than inventing a number.
                  if (!v.empty() && v[0] != '{' && v != "nil")
                  {
                    out = v;
                    return true;
                  }
                }
                i = eq;
              }
            }
          }
        }
        return false;
      };
      std::string best_coord, best_lo, best_hi;
      BigInt best_width;
      bool best = false, any_named = false;
      // Coordinates on which the witness could be PUNCHED OUT instead of cut
      // around (Definition 5). Collected alongside the side cuts rather than
      // instead of them: the two are different policies with different costs
      // and different termination behaviour, and choosing between them is the
      // driver's job — the tool measures and reports both.
      std::vector<std::string> punchable;
      for (const auto &b : path_cov_certify_box)
      {
        auto cit = path_cov_certify_ce.find(b[0]);
        std::string wtxt;
        if (cit == path_cov_certify_ce.end() || !witness_of(b[0], wtxt))
          continue;
        any_named = true;
        // Values come back as decimal or 0x-prefixed hex depending on width.
        auto parse = [](const std::string &s) {
          return (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
                   ? BigInt(s.c_str() + 2, 16)
                   : string2integer(s);
        };
        const BigInt w = parse(wtxt), c = parse(cit->second);
        const BigInt lo = string2integer(b[1]), hi = string2integer(b[2]);
        if (w == c)
          continue; // the witness agrees here: this coordinate cannot separate
        // The witness is inside the box and outside the path; the path's own
        // counterexample is inside the box and inside the path and differs
        // here. So removing the single value `c == w` excludes the witness
        // while keeping a known member of the domain — the same legality rule
        // the side cuts obey, at a cost of ONE value instead of a side.
        if (w >= lo && w <= hi)
          punchable.push_back(b[0] + " != " + integer2string(w));
        if (w > c)
        {
          const BigInt nhi = w - 1;
          if (nhi >= lo && (!best || (nhi - lo) > best_width))
          {
            best = true;
            best_coord = b[0];
            best_lo = integer2string(lo);
            best_hi = integer2string(nhi);
            best_width = nhi - lo;
          }
        }
        else
        {
          const BigInt nlo = w + 1;
          if (nlo <= hi && (!best || (hi - nlo) > best_width))
          {
            best = true;
            best_coord = b[0];
            best_lo = integer2string(nlo);
            best_hi = integer2string(hi);
            best_width = hi - nlo;
          }
        }
      }
      if (!punchable.empty())
      {
        std::string names;
        for (const auto &p : punchable)
          names += (names.empty() ? "" : "; ") + p;
        log_status(
          "--path-cov-certify: PUNCH SUGGESTION for '{}' — instead of cutting "
          "the interval, remove the witness itself: add {} to the box's "
          "`holes` "
          "(Definition 5). Legal by the same rule as a side cut (this path's "
          "own "
          "counterexample differs there and survives), and it costs ONE value "
          "rather than a whole side — the difference between the two was "
          "measured at 5.7e45 on an address coordinate. It is NOT strictly "
          "better: punching converges only where the excluded set is a few "
          "points, while a side cut is what makes progress when the boundary "
          "is "
          "an interval. Which to use is the driver's policy; both are reported",
          key.first,
          names);
      }
      if (best)
        log_status(
          "--path-cov-certify: SHRINK SUGGESTION for '{}' — the witness lies "
          "outside the path on coordinate '{}', and the path's own "
          "counterexample lies on the other side of it, so retry with {} in "
          "[{}, {}] (everything else unchanged). The cut lands ON the witness "
          "rather than halving the interval: the refutation already says where "
          "the boundary is not",
          key.first,
          best_coord,
          best_coord,
          best_lo,
          best_hi);
      else if (any_named && punchable.empty())
        log_status(
          "--path-cov-certify: no single-coordinate shrink for '{}' — on every "
          "bounded coordinate the witness agrees with the path's own "
          "counterexample, so neither a cut NOR a hole separates them while "
          "keeping a known member of the domain. The region has to be split, "
          "or "
          "the path falls back to its concrete counterexample test",
          key.first);
    }
  }

  if (witnessless.empty())
    return;

  std::string names;
  for (const auto &n : witnessless)
    names += (names.empty() ? "" : "; ") + n;
  log_error(
    "--path-cov-certify: INTERNAL DEFECT — {} certification claim(s) were "
    "REFUTED but carry no witness input: {}. The box bounds at least one call "
    "argument, so a refutation is obliged to name the input inside the box "
    "that "
    "leaves the path — that input IS the value the box gets shrunk with. A "
    "verdict without it is not a weaker result, it is an unusable one, and the "
    "verdict alone still prints as if everything worked.",
    witnessless.size(),
    names);
  abort();
}

// ---- THE THIRD STATE: a vacuity verdict the UNWIND BOUND may have made up
// ----
//
// Both stage-2/3 gates rest on one proposition:
//
//     "a VACUOUS verdict means no execution the region admits walks this path"
//
// and that proposition is FALSE whenever an unwinding ASSUMPTION removed
// executions on the path. Under `--solidity-path-coverage` that assumption is
// not a user choice: the pass forces `no-unwinding-assertions`
// (esbmc_parseoptions.cpp:4305), so `loop_bound_exceeded` takes its `else`
// branch and emits `assume(!guard)` instead of an unwinding assertion
// (symex_goto.cpp:482-506). Every execution that needed one more iteration is
// then simply gone -- and the non-vacuity witness `assert(tr != enc || cnt !=
// depth)` HOLDS for want of those executions, which is byte for byte what a
// genuinely empty region produces.
//
// MEASURED, aqua `Aqua.dock` enc=12 depth=3, `--contract Aqua --focus-function
// dock --solidity-max-tx 1`, read out of cov-report.json rather than from exit
// codes:
//
//     config                          F   bounded-holds   decision steps
//     default                         2   61              4
//     --no-simplify                   0   63              0
//     --no-simplify --partial-loops   2   61              4
//
// The two lost witnesses do not become "never asked"; they become
// `bounded-holds`, i.e. the tool asserts the path does not hold when it does.
// `--unwindset 64:512` also restores F=2 while `1:64`, `62:16` and `64:64` do
// not, so the single loop responsible is loop 64 = `__memset_impl`
// (src/c2goto/library/string.c:298), entered only because `--path-cov-assert`
// forces `--no-simplify` (esbmc_parseoptions.cpp:4223) and therefore stops
// folding loop guards (symex_goto.cpp:20). On that contract
// `--path-cov-certify` printed CERTIFIED and `--path-cov-assert` printed
// VACUOUS for the IDENTICAL region, and scripts/solidity_path_put.py refused
// the PUT on the false signal.
//
// SO THE VERDICT BECOMES A THIRD, EXPLICIT TOKEN rather than being folded into
// either of the other two. Folding it into VACUOUS is the defect. Folding it
// into CERTIFIED would be worse. "No verdict" is not "no", and a verdict that
// cannot be trusted has to be a distinct word a driver can branch on -- which
// is why the caller prints `RESULT: UNDECIDED-TRUNCATED` and not a warning
// beside `RESULT: VACUOUS`.
//
// ---- WHY THIS READS goto_functionst::truncated_loops AND ADDS NO COUNTER ----
//
// That set is the bookkeeping the generic "Coverage may be UNDER-REPORTED: N
// loop(s) hit the unwind bound" warning already uses (bmc.cpp:806-823), written
// at exactly one place (symex_goto.cpp:501-506). A second, independent counter
// would drift from it -- the two would disagree about a run and there would be
// no way to tell which was right.
//
// It also needs NO `no-unwinding-assertions` lookup of its own, and that is a
// property of the write site rather than an omission: the insert lives INSIDE
// the `else` branch that fires only when `no_unwinding_assertions` is true, and
// `loop_bound_exceeded` returns before reaching either branch under
// `--partial-loops` (symex_goto.cpp:474). So a non-empty set already means
// "a loop was cut AND the cut was silent AND the executions were really
// discarded" -- a strictly stronger condition than the warning's own
// `options.get_bool_option("no-unwinding-assertions")` guard.
//
// The set is process-global and is never cleared, so this is monotone across
// k-induction phases and thread interleavings: once a run has truncated
// something, no later phase of that process may print a confident vacuity.
// That is the safe direction for a soundness gate.
//
// Returns "" when nothing was truncated -- i.e. when the confident verdict is
// still the tool's to give.
static std::string path_cov_truncated_loops()
{
  std::lock_guard<std::mutex> lk(goto_functionst::truncated_loops_mutex);
  std::string s;
  for (const auto &l : goto_functionst::truncated_loops)
    s += (s.empty() ? "" : "; ") + l;
  return s;
}

// The shared body of the third state, so the two gates cannot drift apart in
// what they claim. `mode` is the flag name, `enc_txt` the path, `why` the
// reason the non-vacuity witness did not come back refuted.
static void path_cov_report_truncated(
  const char *mode,
  const std::string &enc_txt,
  const std::string &why,
  const std::string &loops)
{
  log_error(
    "{}: RESULT: UNDECIDED-TRUNCATED -- the non-vacuity witness for path "
    "enc={} did NOT come back refuted ({}), AND this run cut at least one loop "
    "at the unwind bound while unwinding assertions were disabled, so the "
    "executions needing one more iteration were ASSUMED AWAY rather than "
    "explored. VACUOUS would mean \"no execution the region admits walks this "
    "path\"; that proposition is FALSE whenever a truncation assumption "
    "removed "
    "executions on the path, so THIS RUN IS NOT ENTITLED TO IT and the "
    "confident word is withheld. This is NOT a weaker VACUOUS and must not be "
    "read as one: it is the explicit third state, and the region may well be "
    "perfectly non-empty. MEASURED on aqua Aqua.dock enc=12 (--focus-function "
    "dock --solidity-max-tx 1): --path-cov-certify answered CERTIFIED while "
    "--path-cov-assert answered VACUOUS for the IDENTICAL region, and the "
    "whole "
    "difference was one truncated library loop (__memset_impl, "
    "src/c2goto/library/string.c:298) whose executions --unwindset 64:512 "
    "brings back. TO GET A VERDICT: raise --unwind, use "
    "--unwindset/--unwindsetname for the loop(s) named here, or pass "
    "--partial-loops (which suppresses the assumption), then re-run. Loops "
    "truncated: {}",
    mode,
    enc_txt,
    why,
    loops);
}

void goto_coveraget::report_path_cov_certify()
{
  if (!path_cov_certify_mode)
    return;

  auto verdict_of = [](const std::pair<std::string, std::string> &k) -> char {
    if (k.first.empty())
      return '?';
    std::lock_guard<std::mutex> lock(claim_outcome_mutex);
    auto it = claim_outcome.find(k.first + "\t" + k.second);
    return it == claim_outcome.end() ? '?' : it->second;
  };

  // The path number, read back out of the witness claim's own comment
  // (`<unit-id>:path:<enc>#nonvacuous`) rather than kept in a second static:
  // one source for the identity the whole line is about, so the message and
  // the claim cannot name different paths.
  std::string enc_txt = "<unknown>";
  {
    const std::string &c = path_cov_certify_nonvacuous_key.first;
    const size_t p = c.rfind(":path:");
    const size_t h = c.rfind("#nonvacuous");
    if (p != std::string::npos && h != std::string::npos && h > p + 6)
      enc_txt = c.substr(p + 6, h - p - 6);
  }

  // ---- NON-VACUITY FIRST, and it decides the whole line ----
  //
  // Read before the `#exitN` verdicts, because if the box admits no execution
  // that walks pi then those verdicts say nothing: every one of them holds for
  // want of an execution, which is byte for byte what a certified box produces.
  const char nv = verdict_of(path_cov_certify_nonvacuous_key);
  if (nv != 'F')
  {
    // ---- THE TRUNCATION GATE, IN FRONT OF THE VACUITY VERDICT ----
    //
    // Read BEFORE the VACUOUS line is composed, because by the time that line
    // exists the confident word has already been said. See
    // path_cov_truncated_loops() above for why a non-empty set is exactly the
    // condition under which "nothing walks this path" stops being a statement
    // about the region and becomes a statement about the bound.
    //
    // It covers the WHOLE `nv != 'F'` branch, not only `nv == 'P'`. A witness
    // that never reached the solver, or that came back unknown, is no more
    // trustworthy under a truncated exploration than one that held -- and all
    // three land on the same confident sentence today.
    const std::string why =
      nv == 'P' ? "the antecedent held on every execution, i.e. no admitted "
                  "input reaches this path"
                : (nv == '?' ? "the witness claim never reached the solver"
                             : "the solver returned unknown for the witness "
                               "claim");
    const std::string truncated = path_cov_truncated_loops();
    if (!truncated.empty())
    {
      path_cov_report_truncated("--path-cov-certify", enc_txt, why, truncated);
      exit(1);
    }

    log_error(
      "--path-cov-certify: RESULT: VACUOUS — the box admits NO execution that "
      "walks path enc={} of this unit ({}). Every exit assert therefore holds "
      "FOR WANT OF AN EXECUTION, which is indistinguishable from a certified "
      "box: this run establishes NOTHING about the box. The four structural "
      "gates in front of it are SYNTACTIC (lo>hi, a name bounded twice, holes "
      "emptying the interval, a decimal outside the coordinate's type) and "
      "none "
      "of them can see this — contract state is NOT havoc'd at this "
      "transaction "
      "bound, so a box naming an entry state the constructor never produces is "
      "well-formed, in-type, non-empty and admits nothing",
      enc_txt,
      why);
    exit(1);
  }

  // ---- WHAT AN ABSENT `#exitN` VERDICT MEANS HERE, and why it is not the
  // ---- "absence is not evidence" mistake ----
  //
  // MEASURED on the `_box_inside` fixture: the unit has 4 exits, the box makes
  // 2 of them unreachable, and their asserts never become verification
  // conditions at all (`Generated 3 VCC(s)`). They were not discharged by the
  // simplifier and `--no-simplify` does not bring them back -- symex never
  // reaches the instruction.
  //
  // For THIS query an unreachable exit is a POSITIVE fact: no input the box
  // admits leaves through it, which is exactly what the certificate claims. It
  // is also the reading the verdict line always had, since an assert that is
  // never reached cannot fail.
  //
  // The reason that inference is safe HERE and nowhere else is the non-vacuity
  // witness above. Without it, "every exit unreachable" -- i.e. nothing
  // executed at all -- was indistinguishable from a certificate, and that is
  // the hole this whole change closes. With it, the all-absent case is caught
  // and reported VACUOUS before this point is reached. So the witness is what
  // converts absence from a silent false certificate into a named one; the
  // count is still printed rather than folded away, because a certificate
  // resting mostly on unreachable exits is a weaker artefact than one resting
  // on discharged ones and a reader should be able to see which they have.
  //
  // 'U' is NOT absence and is NOT folded in: the solver was asked and could not
  // answer.
  size_t refuted = 0, holds = 0, unknown = 0, unreachable = 0;
  for (const auto &k : path_cov_certify_exit_keys)
    switch (verdict_of(k))
    {
    case 'F':
      ++refuted;
      break;
    case 'P':
      ++holds;
      break;
    case '?':
      ++unreachable;
      break;
    default:
      ++unknown;
      break;
    }

  if (refuted > 0)
    log_status(
      "--path-cov-certify: RESULT: REFUTED — {} of {} exit assert(s) were "
      "refuted, so an input the box admits leaves this path. The witness input "
      "is the value the box gets shrunk with; see the SHRINK / PUNCH "
      "suggestion "
      "above. Non-vacuity WAS witnessed, so this is a genuine refutation and "
      "not an empty box",
      refuted,
      path_cov_certify_exit_keys.size());
  else if (unknown > 0)
    log_error(
      "--path-cov-certify: RESULT: UNDECIDED — no exit assert was refuted, but "
      "{} of {} came back UNKNOWN from the solver. That is not a certificate: "
      "'nothing refuted it' and 'it was checked' are different statements, and "
      "only the second certifies anything",
      unknown,
      path_cov_certify_exit_keys.size());
  else
    log_status(
      "--path-cov-certify: RESULT: CERTIFIED — every input the box admits "
      "walks "
      "path enc={} ({} of {} exit assert(s) discharged by the solver, {} at "
      "exits the box makes unreachable), and NON-VACUITY was witnessed, so "
      "this "
      "is a statement about executions rather than about an empty box. "
      "BOUNDED: true under THIS exploration (tx/unwind bound, post-constructor "
      "entry state), never 'proven'",
      enc_txt,
      holds,
      path_cov_certify_exit_keys.size(),
      unreachable);

  log_status(
    "--path-cov-certify: the run's VERIFICATION SUCCESSFUL / FAILED line is "
    "NOT the result of this mode. The non-vacuity witness is REFUTED on every "
    "run that certifies, so a certified box prints VERIFICATION FAILED. Read "
    "the RESULT line above");
}

// ---- ONE bound record and ONE parser for every stage-2/3 region spec ----
//
// NOTE, so it is not mistaken for done: --path-cov-certify still carries its
// own inline copy of this parse and of the three structural gates below.
// Unifying them is a separate change, deliberately not folded in here, because
// it is a span replacement inside a branch that three regressions pin by
// MESSAGE and a blind edit there would look like a fix and read as a
// regression. Until it happens, a fix to one copy does not reach the other --
// and FOUR of the five documented false-certificate routes live in exactly this
// code.
struct path_cov_boundt
{
  std::string name, lo, hi;
  std::vector<std::string> holes;
};

// Read `j[key]` (an array of {name, lo, hi, holes?}) into `out`. The key is a
// parameter so certify can keep "box" and stage 3 use "region" without a second
// parser existing.
//
// `lo`/`hi`/`holes` are decimal STRINGS, never JSON numbers: Solidity inputs
// are up to 256 bits and a JSON number would be silently truncated to a double
// on the way in -- a region quietly covering the wrong values is the one
// outcome these queries exist to prevent. A missing field throws (`.at`), which
// the caller turns into a fatal, named parse failure; defaulting it would
// produce a plausible full report answering a different question.
static void parse_bounds(
  const nlohmann::json &j,
  const char *key,
  std::vector<path_cov_boundt> &out)
{
  for (const auto &b : j.value(key, nlohmann::json::array()))
  {
    path_cov_boundt cb;
    cb.name = b.at("name").get<std::string>();
    cb.lo = b.at("lo").get<std::string>();
    cb.hi = b.at("hi").get<std::string>();
    for (const auto &h : b.value("holes", nlohmann::json::array()))
      cb.holes.push_back(h.get<std::string>());
    out.push_back(cb);
  }
}

// ---- Routes 1-3: the THREE ways a region is empty before any type is known --
//
// Returns the reason, or "" when this bound is structurally fine. `seen_names`
// carries across the whole spec and is mutated, which is what makes route 2
// (the same coordinate bounded twice) visible at all.
//
// Why a refusal and not a warning: an unsatisfiable entry assumption means
// nothing executes, so every assertion downstream of it holds FOR WANT OF AN
// EXECUTION. The run then prints a certificate next to a region that contains
// no input. That is a false certificate, not a weak one, and there is nothing
// to reinterpret afterwards -- which is why the gate sits before the query is
// formed rather than where its answer is read.
static std::string path_cov_structural_refusal(
  const path_cov_boundt &b,
  std::set<std::string> &seen_names)
{
  if (string2integer(b.hi) < string2integer(b.lo))
    return "the box is EMPTY on this coordinate (lo=" + b.lo + " > hi=" + b.hi +
           "), so the entry assumption is unsatisfiable and every exit assert "
           "would hold for want of an execution";
  // Closes the obvious hole in the test above: bounding one name twice can
  // intersect to nothing while each bound is individually fine, and a per-bound
  // test would wave both through.
  if (!seen_names.insert(b.name).second)
    return "the coordinate is bounded TWICE in this spec; two bounds on one "
           "name can intersect to an empty box while each is individually "
           "well-formed, which the emptiness test above would not see";
  if (!b.holes.empty())
  {
    // A PUNCHED interval has a SECOND way of being empty and `lo <= hi` cannot
    // see it: `[5,5] \ {5}` passes that test and admits no input at all.
    // Counting only the DISTINCT holes INSIDE [lo, hi] is what makes this exact
    // -- a hole outside the interval removes nothing.
    const BigInt lo = string2integer(b.lo), hi = string2integer(b.hi);
    std::set<std::string> inside;
    for (const auto &h : b.holes)
    {
      const BigInt hv = string2integer(h);
      if (hv >= lo && hv <= hi)
        inside.insert(integer2string(hv));
    }
    if (BigInt((int64_t)inside.size()) >= (hi - lo + 1))
      return "the PUNCHED box is EMPTY on this coordinate: [" + b.lo + ", " +
             b.hi + "] holds " + integer2string(hi - lo + 1) +
             " value(s) and the holes remove all of them, so the entry "
             "assumption is unsatisfiable. `lo <= hi` does NOT catch this -- "
             "the interval is well-formed and the punching is what empties it";
  }
  return std::string();
}

// Does a decimal from a spec fit an unsigned bit-vector coordinate?
//
// Every constant in these queries is built with constant_int2tc ON THE
// COORDINATE'S OWN TYPE, so a decimal above the type's maximum WRAPS and the
// query is emitted about a different number than the one written down.
//
// ---- BOOL IS NOT AN 8-BIT TYPE HERE, WHATEVER get_width() SAYS ----
//
// `bool_type2t::get_width()` returns 8 (src/irep2/irep2_type.cpp — "for the
// byte representing memory model"). That is the right answer to the question it
// answers and must not be changed. It is the WRONG answer to this one: the
// generic `2^width - 1` below would make the admissible range of a bool
// [0, 255], and `state.flag in [0, 200]` would then pass every type gate in the
// pipeline and be emitted as a constraint nobody could have meant. The value
// domain of a bool is {0, 1}, so that is what is checked and that is the
// maximum reported.
static bool path_cov_fits_type(
  const type2tc &t,
  const std::string &dec,
  std::string &tmax_out)
{
  if (is_bool_type(t))
  {
    tmax_out = "1";
    const BigInt v = string2integer(dec);
    return v >= 0 && v <= 1;
  }
  BigInt tmax = 1;
  for (unsigned w = 0; w < t->get_width(); ++w)
    tmax *= 2;
  tmax -= 1;
  tmax_out = integer2string(tmax);
  const BigInt v = string2integer(dec);
  return v >= 0 && v <= tmax;
}

// ---- Route 4: every lo / hi / hole must fit the coordinate's own type ----
static std::string
path_cov_out_of_type_refusal(const path_cov_boundt &b, const type2tc &bt)
{
  std::vector<std::pair<std::string, std::string>> vals = {
    {"lo", b.lo}, {"hi", b.hi}};
  for (const auto &h : b.holes)
    vals.push_back({"hole", h});
  for (const auto &wv : vals)
  {
    std::string tmax;
    if (path_cov_fits_type(bt, wv.second, tmax))
      continue;
    return "the " + wv.first + " value " + wv.second +
           " does not fit the coordinate's own type (admissible range [0, " +
           tmax + "])";
  }
  return std::string();
}

// ---- S5: the ENTRY CONSTRAINT of a BOOL coordinate ----
//
// A bool's domain has two points, so `lo`, `hi` and the holes together name a
// SUBSET S of {0,1} — and every subset of a two-point set is expressible
// EXACTLY as a disjunction of equalities. Returns `OR over v in S of (c == v)`,
// or a nil expression when S is empty (which the caller must refuse: an
// unsatisfiable entry assumption certifies vacuously, the oldest false-
// certificate route in this file).
//
// WHY NOT `lo <= c && c <= hi`: because `>=` / `<=` on a bool operand reaches
// the `assert(is_signedbv_type(...))` arms of smt_conv (2494 / 2525 / 2556 /
// 2587) and SIGABRTs. And why not `constant_int2tc(bool_type, v)`: whether that
// is well formed at the SMT layer was never established, and it does not need
// to be — `gen_true_expr()` / `gen_false_expr()` are the constants of that type
// this file already builds, so the question is sidestepped rather than
// answered.
//
// `holes_applied` is the number of holes that REMOVED A VALUE THE INTERVAL
// ACTUALLY HELD. It is deliberately not `holes.size()`: the caller reports it,
// and a counter that reads the spec instead of the formula is the defect this
// file already fixed once on the certify side. On a two-point domain a hole
// outside [lo, hi] removes nothing, and saying it removed something would be
// the same lie in the other direction.
//
// |S| == 2 STILL RETURNS A GUARD (`c == false || c == true`). It is trivially
// true, and emitting it anyway is the point: the caller increments its
// emitted-bounds counter next to the insertion, so a silent nil here would make
// that counter describe a conjunct that never existed.
static expr2tc path_cov_bool_domain_guard(
  const expr2tc &c,
  const std::string &lo_dec,
  const std::string &hi_dec,
  const std::vector<std::string> &holes,
  size_t &holes_applied)
{
  const BigInt lo = string2integer(lo_dec), hi = string2integer(hi_dec);
  std::set<BigInt> hset;
  for (const auto &h : holes)
    hset.insert(string2integer(h));
  expr2tc g;
  holes_applied = 0;
  for (int64_t v = 0; v <= 1; ++v)
  {
    const BigInt bv((int64_t)v);
    if (bv < lo || bv > hi)
      continue;
    if (hset.count(bv) != 0)
    {
      ++holes_applied;
      continue;
    }
    const expr2tc eq =
      equality2tc(c, v != 0 ? gen_true_expr() : gen_false_expr());
    g = is_nil_expr(g) ? eq : or2tc(g, eq);
  }
  return g;
}

// ---- The contract instance object of ONE named contract, by EXACT id ----
//
// resolve_coord picks the object by SUBSTRING (`id.find(scope_contract)`), and
// that test is wrong here in two ways that are both completely silent. With no
// --contract (or with --coverage-whole-unit) `scope_contract` is EMPTY and the
// test degenerates to "any contract object in the program"; and with
// `--contract Escrow` the substring matches `sol:@_ESBMC_Object_EscrowSrc#`,
// which is exactly the shape of the EscrowSrc/EscrowDst benchmarks.
//
// For a post-state assertion, reading the wrong object is the worst available
// outcome. The instrumented unit writes object X, the exit read looks at object
// Y that nothing touched, so `post == pre` HOLDS vacuously and `post != pre` is
// REFUTED -- a full, plausible, entirely green ladder about a contract that was
// never measured, with no error anywhere.
//
// Exact equality is available because the frontend builds the id as
// `"sol:@" + "_ESBMC_Object_" + <C> + "#"`
// (solidity_convert_contract.cpp, get_static_contract_instance_name) --
// trailing
// '#', nothing after it.
static const symbolt *
path_cov_contract_object(contextt &ctx, const std::string &contract)
{
  if (contract.empty())
    return nullptr;
  const std::string want = "sol:@_ESBMC_Object_" + contract + "#";
  const symbolt *obj = nullptr;
  ctx.foreach_operand([&obj, &want](const symbolt &s) {
    if (obj == nullptr && s.id.as_string() == want)
      obj = &s;
  });
  return obj;
}

// Is this component of the contract object a USER state variable?
//
// The object carries ESBMC's own fields ($address, $balance, $code, $codehash,
// $mutex_<C>, _ESBMC_bind_cname, $dynamic_pool, padding). Same four-way filter
// bmc.cpp applies when it restores the whole object, deliberately reusing the
// stricter of the two forms in that file, so a candidate can never be emitted
// about `$balance`.
static bool path_cov_user_state_name(const std::string &n)
{
  return !(
    n.empty() || n[0] == '$' || n.rfind("_ESBMC", 0) == 0 ||
    n.rfind("anon_pad", 0) == 0);
}

// Walk a dotted field path (`taker`, `timelocks.deployedAt`) down from `e`.
//
// This is what makes a STRUCT ARGUMENT generalisable at all. An aggregate has
// no interval, so coord_expressible refuses it -- correctly -- and a unit whose
// only real argument is a struct therefore had NOTHING to generalise over.
// Measured across all five EscrowSrc units: every one reported zero
// coordinates, with its actual argument sitting right there as a struct.
//
// It is NOT a new coordinate KIND and does not touch definition 6. A region is
// a product of per-coordinate sets, and a struct's scalar fields are exactly
// such coordinates; what was missing was the RESOLUTION, not the
// representation.
//
// A segment that names no component fails the whole resolution. Returning the
// parent aggregate instead would hand the caller a coordinate it did not ask
// for, and on the certify side that is a bound on the wrong quantity -- the one
// outcome that query exists to prevent.
static bool
walk_fields(const namespacet &ns, expr2tc &e, const std::string &path)
{
  size_t p = 0;
  while (p <= path.size())
  {
    const size_t q = path.find('.', p);
    const std::string field =
      path.substr(p, q == std::string::npos ? q : q - p);
    if (field.empty())
      return false;
    const typet st = ns.follow(migrate_type_back(e->type));
    if (st.id() != "struct")
      return false;
    bool hit = false;
    for (const auto &comp : to_struct_type(st).components())
      if (comp.get_name() == field || comp.get("#base_name") == field)
      {
        e = member2tc(migrate_type(comp.type()), e, comp.get_name());
        hit = true;
        break;
      }
    if (!hit)
      return false;
    if (q == std::string::npos)
      return true;
    p = q + 1;
  }
  return false;
}

// |R_c| for a punched interval (Definition 5): how many values of [lo, hi]
// survive once this coordinate's holes are removed.
//
// Shared by the two places that must not disagree — the choice between a hole
// and a side cut, and the test for whether the result is empty. If they used
// separate arithmetic, a region could be chosen as the widest option and then
// printed as non-empty while holding nothing.
static BigInt path_cov_kept_in(
  const std::map<std::string, std::set<BigInt>> &holes,
  const std::string &c,
  const BigInt &lo,
  const BigInt &hi)
{
  if (hi < lo)
    return BigInt(0);
  BigInt n = hi - lo + 1;
  auto h = holes.find(c);
  if (h != holes.end())
    for (const BigInt &v : h->second)
      if (v >= lo && v <= hi)
        n -= 1;
  return n;
}

void goto_coveraget::report_outer_boxes()
{
  if (!path_cov_outer_box_mode)
    return;

  // ---- 1. Read the ladder verdicts ----
  //
  // A probe `assert(tr == pi -> temp_c <= v)` that HOLDS ('P') says every input
  // walking pi has c <= v, so the tightest holding v is the outer bound. A
  // refuted probe is not a failure — it is the ladder doing its job, and its
  // counterexample is a real input of pi with c > v.
  //
  // Nothing here is "proven unreachable": these verdicts are bounded exactly
  // like every other one in this pass, so an outer box is an outer box UNDER
  // THE DECLARED EXPLORATION. Stated because a box is the input to a soundness
  // argument downstream, and a bounded box silently promoted to an absolute one
  // is precisely how a certified region ends up too wide.
  struct bound_infot
  {
    bool have_u = false, have_l = false;
    BigInt u, l;
    // The TIGHTEST REFUTED probe on each side. `assert(temp_c <= v)` refuted
    // means some input of this path has c > v, so the true bound lies strictly
    // above v; together with the smallest holding probe it BRACKETS the bound.
    //
    // Reported because it is what makes the loop converge geometrically. A
    // batch of K probes gives resolution span/(K+1) in one round; refining the
    // next round's span to this bracket divides the resolution by (K+1) again,
    // so the precision is logarithmic in ROUNDS while each round stays one run.
    // Without the bracket a driver has no principled next span — and the
    // recorded rule it would otherwise use (take the span from the nearest
    // sibling counterexample) was measured NOT to work: a solver counterexample
    // can sit arbitrarily far from the boundary, and on the first contract
    // tried it sat at 2^256-1.
    bool have_ur = false, have_lr = false;
    BigInt ur, lr;
  };
  std::map<std::pair<uint64_t, std::string>, bound_infot> bounds;
  // Seed with the free bound: the coordinate's own type range (see the header).
  for (const auto &[enc, depth] : path_cov_outer_box_paths)
    for (const auto &[cname, r] : path_cov_outer_box_type_range)
    {
      bound_infot &b = bounds[{enc, cname}];
      b.have_l = b.have_u = true;
      b.l = string2integer(r.first);
      b.u = string2integer(r.second);
    }
  size_t decided = 0, undecided = 0;
  {
    std::lock_guard lock(claim_outcome_mutex);
    for (const auto &p : path_cov_outer_box_probes)
    {
      auto it = claim_outcome.find(p.key.first + "\t" + p.key.second);
      if (it == claim_outcome.end())
      {
        ++undecided;
        continue;
      }
      ++decided;
      const BigInt v = string2integer(p.value);
      bound_infot &b = bounds[{p.enc, p.coord}];
      if (it->second == 'F')
      {
        // Refuted: keep the TIGHTEST one on each side, so the true bound is
        // bracketed rather than merely upper-bounded.
        if (p.upper)
        {
          if (!b.have_ur || v > b.ur)
          {
            b.ur = v;
            b.have_ur = true;
          }
        }
        else if (!b.have_lr || v < b.lr)
        {
          b.lr = v;
          b.have_lr = true;
        }
        continue;
      }
      if (it->second != 'P')
        continue; // unknown: establishes neither side
      if (p.upper)
      {
        if (!b.have_u || v < b.u)
        {
          b.u = v;
          b.have_u = true;
        }
      }
      else if (!b.have_l || v > b.l)
      {
        b.l = v;
        b.have_l = true;
      }
    }
  }

  // Refused coordinates, named BEFORE any box is printed. A refused coordinate
  // emits no probe, so it appears in no box below — and an absence reads as
  // "not asked about", while the truth is "asked about and refused". Those are
  // different facts and only one of them is a gap in the coordinate set.
  if (!path_cov_refused_coords.empty())
  {
    std::string refused;
    for (const auto &[cn, why] : path_cov_refused_coords)
      refused += (refused.empty() ? "" : "; ") + cn + " (" + why + ")";
    log_warning(
      "--path-cov-outer-box: {} coordinate(s) were REFUSED and appear in NO "
      "box "
      "below: {}. Their absence is a refusal, not a measurement — reading it "
      "as "
      "\"bounded by the whole type\" would attribute a measured bound to a "
      "coordinate nothing was measured on, and would widen every region that "
      "quoted it",
      path_cov_refused_coords.size(),
      refused);
  }

  std::string pin_note;
  for (const auto &[pn, pv] : path_cov_outer_box_pins)
    pin_note += (pin_note.empty() ? "" : ", ") + pn + " == " + pv;
  if (!pin_note.empty())
    log_status(
      "--path-cov-outer-box: every box and every region below is measured "
      "under "
      "the PIN {} — they describe that SLICE of the input space, not the whole "
      "domain. Any test rendered from one must carry the pin as a `require` "
      "too, "
      "or it claims something about inputs that were never examined",
      pin_note);

  log_status(
    "--path-cov-outer-box: {} of {} ladder probe(s) reached the solver. A "
    "probe "
    "that HOLDS is an outer bound; a refuted probe is the ladder working, and "
    "its counterexample is a genuine input of that path beyond the bound",
    decided,
    decided + undecided);

  // Coordinates, in the order the spec listed them.
  std::vector<std::string> coords;
  for (const auto &p : path_cov_outer_box_probes)
    if (std::find(coords.begin(), coords.end(), p.coord) == coords.end())
      coords.push_back(p.coord);

  auto show = [&](uint64_t enc) {
    std::string s;
    for (const auto &c : coords)
    {
      auto it = bounds.find({enc, c});
      s += (s.empty() ? "" : ", ") + c + " in ";
      if (it == bounds.end() || (!it->second.have_l && !it->second.have_u))
        s += "(unbounded within the probed span)";
      else
      {
        s += "[";
        s += it->second.have_l ? integer2string(it->second.l)
                               : std::string("<span lo");
        s += ", ";
        s += it->second.have_u ? integer2string(it->second.u)
                               : std::string(">span hi");
        s += "]";
        // An inverted interval is EMPTY, and printed bare it reads as a
        // measured range. Say so at the interval itself, not only in a trailing
        // note: the number pair is what gets quoted.
        if (
          it->second.have_l && it->second.have_u && it->second.l > it->second.u)
          s += " (EMPTY: lo > hi)";
      }
    }
    return s;
  };

  for (const auto &[enc, depth] : path_cov_outer_box_paths)
  {
    log_status(
      "--path-cov-outer-box: path enc={} depth={} OUTER box (D_path is "
      "CONTAINED "
      "in it): {}",
      enc,
      depth,
      show(enc));
    // The bracket: where the true bound still is. This is the next round's
    // span, and refining to it divides the resolution by (probes+1) again —
    // which is how a batch method reaches logarithmic precision without ever
    // becoming an adaptive query-per-step search.
    std::string br;
    for (const auto &c : coords)
    {
      auto it = bounds.find({enc, c});
      if (it == bounds.end())
        continue;
      if (it->second.have_ur && it->second.have_u)
        br += (br.empty() ? "" : ", ") + c + " upper in (" +
              integer2string(it->second.ur) + ", " +
              integer2string(it->second.u) + "]";
      if (it->second.have_lr && it->second.have_l)
        br += (br.empty() ? "" : ", ") + c + " lower in [" +
              integer2string(it->second.l) + ", " +
              integer2string(it->second.lr) + ")";
    }
    if (!br.empty())
      log_status(
        "--path-cov-outer-box: path enc={} BRACKET (refine the next batch's "
        "span "
        "to this): {}",
        enc,
        br);
  }

  // ---- 2. Subtract the siblings, greedily, one cut per sibling ----
  //
  // Zero queries. Path domains partition the input space, so an input in this
  // path's outer box and in NO sibling's outer box must walk this path. In one
  // dimension that difference is a union of segments; in more it is an L shape,
  // which cannot be written as one `require` per parameter. So each
  // intersecting sibling is removed by ONE cut along ONE coordinate — the
  // result is a box, and any point left in it is outside that sibling by
  // construction.
  //
  // A cut is legal only if this path's own counterexample survives it: the CE
  // is a known member of D_path, so a cut that drops it has certainly cut into
  // the real domain. Among legal cuts, take the one that loses the least width
  // on the coordinate it cuts. Shrinking only ever makes the region SMALLER, so
  // the heuristic cannot break correctness — it only decides how much is kept,
  // which is why it is a free parameter and is reported rather than argued for.
  //
  // ---- WHY A THIRD KIND OF CUT: THE HOLE (Definition 5) ----
  //
  // The two side cuts cannot express "everything except v", and the cost of
  // that is not resolution, it is DETERMINISM. When a sibling occupies a single
  // value v strictly inside this box, both side cuts are legal and the one that
  // gets taken is decided by which side this path's own counterexample happens
  // to sit on — a value the solver chose. MEASURED on one address coordinate:
  // supplying 2^160-1 as the sibling counterexample yields `[256, 2^160-1]`,
  // supplying 0 yields `[0, 254]`. Both are correct subsets of the true domain
  // `[0,254] U [256, 2^160-1]`; they differ by a factor of 5.7e45.
  //
  // Punching v out instead gives `[0, 2^160-1] \ {v}` in BOTH cases. Same
  // legality rule (this path's CE must survive, so CE != v), same only-ever-
  // narrower invariant, and the answer stops depending on a value nobody chose.
  //
  // Restricted, deliberately, to a sibling whose outer box on the coordinate is
  // a SINGLE POINT. A sibling spanning several values could be punched out
  // value by value, and Definition 5 allows it — but how many points are worth
  // punching before a side cut is better is a policy knob with a yield question
  // behind it, and nothing has measured it. A multi-point sibling therefore
  // still gets a side cut, and that is a stated limit rather than an oversight.
  for (const auto &[enc, depth] : path_cov_outer_box_paths)
  {
    // Materialise this path's box; skip a coordinate with no bound at all.
    std::map<std::string, std::pair<BigInt, BigInt>> box;
    // Definition 5's H, per coordinate: the values removed from [lo, hi].
    std::map<std::string, std::set<BigInt>> holes;
    for (const auto &c : coords)
    {
      auto it = bounds.find({enc, c});
      if (it != bounds.end() && it->second.have_l && it->second.have_u)
        box[c] = {it->second.l, it->second.u};
    }
    if (box.empty())
    {
      log_status(
        "--path-cov-outer-box: path enc={} has no fully bounded coordinate; no "
        "certified region is computed for it (the ladder span did not bracket "
        "its domain — widen the span or add probes)",
        enc);
      continue;
    }

    auto ce_of = [&](uint64_t e, const std::string &c, BigInt &out) {
      auto it = path_cov_outer_box_ce.find({e, c});
      if (it == path_cov_outer_box_ce.end())
        return false;
      out = string2integer(it->second);
      return true;
    };

    size_t degenerate = 0;
    for (const auto &[senc, sdepth] : path_cov_outer_box_paths)
    {
      if (senc == enc)
        continue;
      // Does the sibling's box still intersect ours on every coordinate?
      bool intersects = true;
      std::map<std::string, std::pair<BigInt, BigInt>> sbox;
      for (const auto &c : coords)
      {
        auto it = bounds.find({senc, c});
        if (it == bounds.end() || !it->second.have_l || !it->second.have_u)
          continue; // unbounded there: cannot be used to separate on it
        sbox[c] = {it->second.l, it->second.u};
        auto ob = box.find(c);
        if (ob == box.end())
          continue;
        if (it->second.u < ob->second.first || it->second.l > ob->second.second)
          intersects = false;
      }
      if (!intersects)
        continue; // provably disjoint already: nothing to remove
      if (sbox.empty())
      {
        // The sibling has no coordinate bounded on BOTH sides, so there is no
        // cut that can be shown to exclude it — its domain may lie anywhere in
        // the probed span. Counting this as separated would silently keep a
        // region that provably contains foreign inputs, which is the one
        // outcome the subtraction must never produce. It is the same shape as
        // the missing-path warning above: an unmeasured sibling is not an
        // absent one.
        ++degenerate;
        continue;
      }

      // Best legal cut across coordinates. Candidates are scored by how many
      // values SURVIVE on the coordinate they touch — not by raw width —
      // because a hole and a side cut are no longer comparable by width alone:
      // a hole keeps the full interval minus one point.
      bool best = false;
      std::string best_c;
      std::pair<BigInt, BigInt> best_range;
      bool best_is_hole = false;
      BigInt best_hole;
      BigInt best_kept;
      auto kept_in =
        [&](const std::string &c, const BigInt &lo, const BigInt &hi) {
          return path_cov_kept_in(holes, c, lo, hi);
        };
      for (const auto &[c, sr] : sbox)
      {
        auto ob = box.find(c);
        if (ob == box.end())
          continue;
        BigInt ce;
        if (!ce_of(enc, c, ce))
          continue; // no CE for this coordinate: cannot check legality
        // THE HOLE, tried first so it wins ties: the sibling occupies exactly
        // one value here, that value is inside our interval, and our own
        // counterexample is not it. Removing it excludes the whole sibling
        // while keeping a known member of the domain.
        if (
          sr.first == sr.second && sr.first >= ob->second.first &&
          sr.first <= ob->second.second && ce != sr.first)
        {
          const BigInt k = kept_in(c, ob->second.first, ob->second.second) - 1;
          if (!best || k > best_kept)
          {
            best = true;
            best_is_hole = true;
            best_c = c;
            best_hole = sr.first;
            best_kept = k;
          }
        }
        // Keep the part strictly below the sibling, or strictly above it.
        if (sr.first > ob->second.first && ce < sr.first)
        {
          std::pair<BigInt, BigInt> r{ob->second.first, sr.first - 1};
          BigInt k = kept_in(c, r.first, r.second);
          if (!best || k > best_kept)
          {
            best = true;
            best_is_hole = false;
            best_c = c;
            best_range = r;
            best_kept = k;
          }
        }
        if (sr.second < ob->second.second && ce > sr.second)
        {
          std::pair<BigInt, BigInt> r{sr.second + 1, ob->second.second};
          BigInt k = kept_in(c, r.first, r.second);
          if (!best || k > best_kept)
          {
            best = true;
            best_is_hole = false;
            best_c = c;
            best_range = r;
            best_kept = k;
          }
        }
      }
      if (!best)
      {
        // No coordinate separates them while keeping the CE. This is the
        // documented degenerate case: the certified region collapses towards
        // the CE point and the path falls back to a concrete test. Reported,
        // never papered over by keeping a region that provably contains foreign
        // inputs.
        ++degenerate;
        continue;
      }
      if (best_is_hole)
        holes[best_c].insert(best_hole);
      else
      {
        box[best_c] = best_range;
        // A hole outside the surviving interval removes nothing and would print
        // as a constraint on values the region no longer contains — a reader
        // would take it as evidence about the domain when it is evidence about
        // an interval that has been cut away since.
        auto h = holes.find(best_c);
        if (h != holes.end())
        {
          std::set<BigInt> keep;
          for (const BigInt &v : h->second)
            if (v >= best_range.first && v <= best_range.second)
              keep.insert(v);
          h->second.swap(keep);
        }
      }
    }

    std::string s;
    std::string empty_on;
    size_t holes_punched = 0;
    for (const auto &[c, r] : box)
    {
      s += (s.empty() ? "" : ", ") + c + " in [" + integer2string(r.first) +
           ", " + integer2string(r.second) + "]";
      // Definition 5's punched interval, printed WITH the interval rather than
      // in a trailing note: `[0, 2^160-1]` and `[0, 2^160-1] \ {255}` are
      // different regions and the numbers are what gets quoted.
      auto h = holes.find(c);
      if (h != holes.end() && !h->second.empty())
      {
        std::string hs;
        for (const BigInt &v : h->second)
          hs += (hs.empty() ? "" : ", ") + integer2string(v);
        s += " \\ {" + hs + "}";
        holes_punched += h->second.size();
      }
      // EMPTY has a second route once the interval can be punched: a
      // well-formed [lo, hi] whose every value has been removed. Both routes
      // land in the same note, because both mean the region holds no input.
      if (
        r.first > r.second ||
        path_cov_kept_in(holes, c, r.first, r.second) <= 0)
        empty_on += (empty_on.empty() ? "" : ", ") + c;
    }

    // AN EMPTY REGION IS NOT A REGION, and the heading is the dangerous part.
    // The subtraction can invert an interval — measured: with the environment
    // pinned, the ABI-gate revert path's domain is empty in that slice and the
    // cut duly produced lo > hi. Printed under "CERTIFIED region" with no
    // further comment, that reads to any reader as a region that was certified.
    // Nothing here certifies anything (this whole stage is zero-query), and an
    // empty box additionally certifies VACUOUSLY if it reaches the query.
    //
    // The driver already refuses these, which is exactly why the line KEEPS its
    // parseable shape instead of being replaced: the driver finds the region by
    // this prefix, and suppressing the line would drop it into its generic "no
    // fully bounded region was measured" branch — trading a precise diagnosis
    // for a vaguer one in the name of safety. Two defences, not one moved.
    //
    // What changes is that the emptiness travels WITH the numbers rather than
    // after them. A caveat at the end of the line is read after the interval
    // has already been read.
    std::string empty_note;
    if (!empty_on.empty())
      empty_note =
        " — EMPTY, NOT CERTIFIED: no value survives on " + empty_on +
        " (either lo > hi, or the holes remove every value the interval holds "
        "— "
        "a punched interval can be empty while its endpoints look well-formed)"
        ", so this box contains no input at all. The subtraction removed "
        "everything, which under a pin usually means the pin excluded this "
        "path "
        "from the slice; the honest statement is that exclusion. Do NOT hand "
        "this box to the certification query: an unsatisfiable assumption "
        "answers SUCCESSFUL for want of any execution";
    // S2: the unit itself is disqualified, so EVERY region below is a candidate
    // that must not be handed to the certification query -- which now refuses
    // this unit outright, so a driver that ignored this line would simply get a
    // named non-zero exit there instead. Printed with the region rather than
    // once at the top, because the region line is what gets quoted.
    std::string obstacle_note;
    if (!path_cov_outer_box_obstacle.empty())
      obstacle_note =
        " — NAMED OBSTACLE: " + path_cov_outer_box_obstacle +
        ". The containment above is still true, but this region must NOT be "
        "certified or turned into a test: the model admits an execution the "
        "chain does not have, so a test built from a counterexample in it can "
        "be RED on the UNMODIFIED contract";
    std::string caveat;
    if (degenerate > 0)
      caveat =
        " — WARNING: " + std::to_string(degenerate) +
        " sibling(s) could not be separated by any single coordinate cut that "
        "keeps this path's counterexample, so the region above STILL OVERLAPS "
        "them and must be certified before use (or the path falls back to its "
        "concrete counterexample test)";
    log_status(
      "--path-cov-outer-box: path enc={} CERTIFIED region after subtracting "
      "sibling outer boxes (zero queries): {}{}{}{}",
      enc,
      s,
      empty_note,
      obstacle_note,
      caveat);
    // Printed as its own line, and printed as a COUNT, because the property a
    // regression has to pin is "the subtraction punched rather than took a
    // side" — which is invisible in the region text of a path that happened to
    // need no hole, and which is exactly the property that stops the answer
    // depending on which counterexample the solver returned.
    if (holes_punched > 0)
      log_status(
        "--path-cov-outer-box: path enc={} — {} of the cut(s) above are HOLES "
        "(Definition 5), not side cuts: a sibling occupying a single value was "
        "removed by excluding that value. This is the part of the region that "
        "does NOT depend on which counterexample the solver returned for the "
        "sibling; a side cut there would have kept only the side holding this "
        "path's own counterexample",
        enc,
        holes_punched);
  }

  log_status(
    "--path-cov-outer-box: these regions are CANDIDATES, not certificates. The "
    "subtraction is sound only if path enumeration is complete for this unit; "
    "run --path-cov-certify on each region to confirm it, which is the same "
    "confirmation the method already prescribes when any path is undecided");
}

void goto_coveraget::report_path_cov_assertions()
{
  if (!path_cov_assert_mode)
    return;

  // ---- Refused variables FIRST, before any verdict ----
  //
  // A refused variable emits no candidate, so it appears in no row below -- and
  // an absence reads as "not asked about" while the truth is "asked about and
  // refused". In THIS mode the misreading is worse than in the outer box: a
  // state variable with no row reads as one that needed no assertion, i.e. as
  // one that does not change.
  if (!path_cov_refused_coords.empty())
  {
    std::string refused;
    for (const auto &rc : path_cov_refused_coords)
      refused +=
        (refused.empty() ? "" : "; ") + rc.first + " (" + rc.second + ")";
    log_warning(
      "--path-cov-assert: {} state variable(s) carry NO candidate and appear "
      "in NO row below: {}. Their absence is a REFUSAL, not a measurement -- "
      "read as \"unchanged\" it would be a claim about a variable nothing was "
      "asserted on",
      path_cov_refused_coords.size(),
      refused);
  }

  // ---- NON-VACUITY FIRST, and it is a HARD FAILURE ----
  //
  // Read before any candidate is printed, because if it did not hold there is
  // no table -- every row below would be a statement about a region that admits
  // no execution, and each row would read exactly like a certified one.
  {
    const std::string nv = path_cov_assert_nonvacuous_key.first.empty()
                             ? std::string()
                             : path_cov_assert_nonvacuous_key.first + "\t" +
                                 path_cov_assert_nonvacuous_key.second;
    char nvv = '?';
    if (!nv.empty())
    {
      std::lock_guard<std::mutex> lock(claim_outcome_mutex);
      auto it = claim_outcome.find(nv);
      if (it != claim_outcome.end())
        nvv = it->second;
    }
    if (nvv != 'F')
    {
      // ---- THE TRUNCATION GATE, IN FRONT OF THE VACUITY VERDICT ----
      //
      // Same gate as report_path_cov_certify()'s, and this is the side the
      // defect was MEASURED on: `--path-cov-assert` is the ONLY one of the
      // three sub-modes that forces `--no-simplify`
      // (esbmc_parseoptions.cpp:4223, against :4200-4203 for the other two),
      // and that force is precisely what lets a library loop be entered,
      // truncated at 4, and its executions assumed away. So this arm sees the
      // aqua signature -- the witness holding while six MUTUALLY CONTRADICTORY
      // rungs all hold beside it -- and printing VACUOUS there is the wrong
      // answer, not a conservative one.
      const std::string enc_txt =
        path_cov_assert_candidates.empty()
          ? std::string("<unknown>")
          : std::to_string(path_cov_assert_candidates.front().enc);
      const std::string why =
        nvv == 'P'
          ? "the antecedent held on every execution, i.e. no admitted "
            "input reaches this path"
          : (nvv == '?' ? "the witness claim never reached the solver"
                        : "the solver returned unknown for the witness "
                          "claim");
      const std::string truncated = path_cov_truncated_loops();
      if (!truncated.empty())
      {
        path_cov_report_truncated("--path-cov-assert", enc_txt, why, truncated);
        exit(1);
      }

      log_error(
        "--path-cov-assert: THE REGION IS VACUOUS -- no execution it admits "
        "walks path enc={} of this unit ({}). Every candidate below would hold "
        "FOR WANT OF AN EXECUTION, and the table would be indistinguishable "
        "from a fully certified ladder. The four structural gates on the "
        "region "
        "are SYNTACTIC (lo>hi, a name bounded twice, holes emptying the "
        "interval, a decimal outside the type) and none of them can see this: "
        "contract state is NOT havoc'd at this transaction bound, so a region "
        "naming an entry state the constructor never produces is well-formed, "
        "in-type, non-empty and admits nothing. No verdict table is printed",
        enc_txt,
        why);
      exit(1);
    }
    log_status(
      "--path-cov-assert: region NON-VACUITY witnessed -- at least one input "
      "the region admits does walk path enc={}, so the verdicts below are "
      "statements about executions rather than about an empty region",
      path_cov_assert_candidates.empty()
        ? 0
        : path_cov_assert_candidates.front().enc);
  }

  log_status(
    "--path-cov-assert: the run's VERIFICATION SUCCESSFUL / FAILED line is NOT "
    "the result of this mode. A REFUTED candidate is the ladder WORKING: it "
    "means there is an input in the region walking this path whose post-state "
    "violates the candidate, and that input is the counterexample. The result "
    "of this run is the per-candidate table below");

  size_t holds = 0, refuted = 0, nv_unknown = 0, nv_unreached = 0;
  {
    std::lock_guard<std::mutex> lock(claim_outcome_mutex);
    // Emission order, never completion order: under --parallel-solving the
    // claims finish in an arbitrary order, and a table whose ROW ORDER depended
    // on that would make every pinned line flaky for a reason unrelated to what
    // it pins.
    for (const auto &c : path_cov_assert_candidates)
    {
      const std::string sig = c.key.first + "\t" + c.key.second;
      auto it = claim_outcome.find(sig);
      std::string verdict;
      // The third state is EXPLICIT and has two named causes. Collapsing them
      // would hide the difference between "the solver could not answer" and
      // "the claim never reached the solver", which are opposite problems.
      if (it == claim_outcome.end())
      {
        ++nv_unreached;
        verdict = "NO VERDICT (never reached the solver)";
      }
      else if (it->second == 'P')
      {
        // Never "proven": path_cov_can_prove_unreachable() is false for every
        // coverage configuration, so this is bounded-holds.
        ++holds;
        verdict = "HOLDS";
      }
      else if (it->second == 'F')
      {
        ++refuted;
        verdict = "REFUTED";
      }
      else
      {
        ++nv_unknown;
        verdict = "NO VERDICT (solver unknown)";
      }

      // The witness is REPORTED, never demanded. audit_certify_witness makes a
      // witness-less refutation a hard failure and that rule does not transfer:
      // there a refutation is a defect-shaped event whose witness shrinks the
      // box, here it is an expected outcome and there is no box to shrink.
      std::string witness;
      if (it != claim_outcome.end() && it->second == 'F')
      {
        auto ce = path_ce.find(sig);
        if (ce != path_ce.end() && !ce->second.inputs.empty())
        {
          std::string vals;
          for (const auto &iv : ce->second.inputs)
            vals += (vals.empty() ? "" : ", ") + iv.first + "=" + iv.second;
          witness = "  [witness: " + vals + "]";
        }
        else
          witness =
            "  [no witness input recorded -- pass --cov-report-json if the "
            "refuting input is wanted; the verdict itself is unaffected]";
      }
      log_status(
        "--path-cov-assert: {}: {}  {}{}", c.var, c.text, verdict, witness);
    }
  }

  // All four counts, every time, zeros included: a category that stops
  // occurring is noticed, a category that silently disappears is not.
  log_status(
    "--path-cov-assert: ladder summary -- {} candidate(s): {} HOLDS, {} "
    "REFUTED, {} no verdict (solver unknown), {} no verdict (never reached the "
    "solver). HOLDS is BOUNDED-holds: true for every input of the region under "
    "THIS exploration (tx/unwind bound, post-constructor entry state), never "
    "\"proven\"",
    path_cov_assert_candidates.size(),
    holds,
    refuted,
    nv_unknown,
    nv_unreached);
}

bool goto_coveraget::focus_selects_unit(
  const std::string &unit_id,
  const std::string &focus)
{
  if (focus.empty())
    return true; // no narrowing
  // A caller that already has the fully mangled id may pass it verbatim. This
  // is a whole-value comparison on purpose: an id contains '@' and '#' but no
  // separator this option uses, so it can never be mistaken for a list.
  if (unit_id == focus)
    return true;
  // `sol:@C@<C>@F@<fn>#<node-id>` -- extract the <fn> segment, bounded by the
  // '#' so `pubx` cannot match a focus of `pub`, and ask the SHARED parser
  // whether the focus names it.
  //
  // --focus-function names a SET (`--focus-function a,b`), and the membership
  // test lives in util/config/focus_function.h rather than here because the
  // frontend's dispatcher filter has to answer the identical question about the
  // identical value. Two copies of it would be a detector keyed on a condition
  // its own branch does not state: a unit the dispatcher can enter but this
  // test skips carries no claim at all, which reads as an honest zero rather
  // than as a hole. Exact per-name equality is what the frontend applies to the
  // source-level name, and it is also what makes every OVERLOAD selected here:
  // overloads share <fn> and differ only in <node-id>, and the dispatcher
  // offers all of them.
  const std::string tag = "@F@";
  const size_t f = unit_id.find(tag);
  if (f == std::string::npos)
    return false;
  const size_t b = f + tag.size();
  const size_t h = unit_id.find('#', b);
  if (h == std::string::npos)
    return false;
  return focus_function_selects(focus, unit_id.substr(b, h - b));
}

void goto_coveraget::audit_entry_liveness(const std::string &focus_function)
{
  // Rebuilt every call: report_coverage can run more than once (k-induction
  // phases), and a stale set would label paths of a unit that WAS entered this
  // time.
  units_not_entered.clear();
  if (all_claims.empty())
    return;

  // Is this unit the one --focus-function names?
  //
  // Goes through focus_selects_unit(), the SAME matcher
  // solidity_path_coverage() narrows instrumentation with. Two independent
  // copies of this test would be a detector keyed on a condition its own branch
  // does not state: a unit the narrowing considered focused but this test did
  // not would be filed "excluded by --focus-function" — informational — when it
  // is in fact the focused unit having never been entered, which is the hard
  // failure below.
  auto is_focused = [&focus_function](const std::string &unit) {
    if (focus_function.empty())
      return true; // no narrowing: every unit is expected to be entered
    return focus_selects_unit(unit, focus_function);
  };

  // Group the enumerated paths by unit. The claim comment is
  // "<function-id>:path:<enc>", so the unit is the part before ":path:".
  struct tally
  {
    size_t instrumented = 0; // claims actually put to the solver this run
    size_t decided = 0;      // of those, how many came back with a verdict
  };
  std::map<std::string, tally> per_unit;
  size_t total_instrumented = 0, total_decided = 0;

  {
    std::lock_guard lock(claim_outcome_mutex);
    for (const auto &key : all_claims)
    {
      const std::string &comment = key.first;
      const size_t p = comment.rfind(":path:");
      const std::string unit =
        p == std::string::npos ? comment : comment.substr(0, p);

      // A path deliberately skipped because an earlier round already witnessed
      // it is not evidence of anything — it was never instrumented this run.
      auto sid = path_stable_id.find(key);
      if (sid != path_stable_id.end() && path_covered_ids.count(sid->second))
        continue;

      tally &t = per_unit[unit];
      ++t.instrumented;
      ++total_instrumented;
      if (claim_outcome.count(key.first + "\t" + key.second))
      {
        ++t.decided;
        ++total_decided;
      }
    }
  }

  // First-class output: how much of the solve stage actually happened. Reported
  // whether or not anything is wrong, because "0 of N" is exactly the number
  // that was missing when a fully empty run read as a merely undecided one.
  log_status(
    "--solidity-path-coverage: {} of {} instrumented path claim(s) reached the "
    "solver across {} unit(s)",
    total_decided,
    total_instrumented,
    per_unit.size());

  std::vector<std::string> dead, dead_by_design;
  for (const auto &[unit, t] : per_unit)
    if (t.instrumented > 0 && t.decided == 0)
    {
      // Recorded for BOTH branches, WITH the cause. Whether or not the absence
      // is a defect, a path of this unit must be reported as "the unit was
      // never entered" rather than by a per-path verdict that means nothing
      // when nothing ran — and the cause has to travel with it, because the
      // planned entry-liveness witness needs exactly this split to avoid
      // aborting on every --focus-function run.
      const bool focused = is_focused(unit);
      units_not_entered[unit] =
        focused ? "harness never entered it (no --focus-function narrowing "
                  "explains this)"
                : "excluded by --focus-function '" + focus_function + "'";
      (focused ? dead : dead_by_design).push_back(unit);
    }

  if (!dead_by_design.empty())
    log_status(
      "--solidity-path-coverage: {} unit(s) were not entered because "
      "--focus-function narrowed the dispatcher to '{}'. That is the intended "
      "behaviour of per-method runs (their paths are meant to be witnessed by "
      "the run that focuses on them and unioned via the covered set), so it is "
      "reported, not treated as a failure",
      dead_by_design.size(),
      focus_function);

  if (dead.empty())
    return;

  std::string names;
  for (const auto &d : dead)
    names += (names.empty() ? "" : "; ") + d;

  // THE AUDIT'S PREMISE DOES NOT HOLD ON A RUN THAT DIED, and applying it there
  // would be the third time this check has accused a correct run of a defect it
  // does not have (the first was --focus-function; the second was the certify
  // audit on --result-only). "A unit with instrumented claims should have been
  // entered" is true of a run that reached the end of its job loop. A run that
  // stopped at claim 1 of 1822 has legitimately entered almost nothing.
  //
  // And the cost of getting it wrong here is not a false accusation but a lost
  // artefact: this audit runs BEFORE any figure is printed and before the JSON
  // is written (bmc.cpp:1134 vs :1336), so aborting would destroy the partial
  // report on its way out -- the exact deliverable the partial path exists to
  // save.
  if (!path_cov_partial_reason.empty())
  {
    log_warning(
      "--solidity-path-coverage: {} unit(s) had claims instrumented but none "
      "decided: {}. NOT treated as a defect, because this run did not conclude "
      "({}). On a complete run this is a hard failure; here it means only that "
      "the run stopped before reaching those units, and their paths are filed "
      "'unit-not-entered' with that caveat rather than being used as evidence "
      "of anything",
      dead.size(),
      names,
      path_cov_partial_reason);
    return;
  }

  if (total_decided == 0)
    log_error(
      "--solidity-path-coverage: INTERNAL DEFECT — NOT ONE of the {} "
      "instrumented path claim(s) reached the solver. The harness never "
      "entered any unit, so this run establishes nothing whatsoever; every "
      "path would otherwise be reported 'U', which reads exactly like an "
      "honest solver timeout. This is the extreme case of the per-unit check "
      "below and is a tool failure, not a result.",
      total_instrumented);
  else
    log_error(
      "--solidity-path-coverage: INTERNAL DEFECT — {} unit(s) had claims "
      "instrumented but NONE of them reached the solver, i.e. the harness "
      "never entered them: {}. Their paths would be reported 'U', which is "
      "indistinguishable from an honest solver timeout, so the results for "
      "those units are vacuous rather than merely incomplete.",
      dead.size(),
      names);
  abort();
}

std::string goto_coveraget::get_filename_from_path(std::string path)
{
  if (path.find_last_of('/') != std::string::npos)
    return path.substr(path.find_last_of('/') + 1);

  return path;
}

/*
  replace the old_condition of all assertions
  to the new condition(guard)
*/
void goto_coveraget::replace_all_asserts_to_guard(
  const expr2tc &guard,
  bool is_instrumentation)
{
  std::unordered_set<std::string> location_pool = {};
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      Forall_goto_program_instructions (it, goto_program)
      {
        std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->is_assert())
          replace_assert_to_guard(guard, it, is_instrumentation);
      }
    }
}

/*
  replace the old_condition of a specific assertion
  to the new condition(guard)
*/
void goto_coveraget::replace_assert_to_guard(
  const expr2tc &guard,
  goto_programt::instructiont::targett &it,
  bool is_instrumentation)
{
  const expr2tc old_guard = it->guard;
  it->guard = guard;
  if (is_instrumentation)
    it->location.property("instrumented assertion");
  else
    it->location.property("replaced assertion");
  // Keep an existing comment (e.g. the "msg" of __ESBMC_assert(cond, "msg"),
  // or the modeled "assertion ..." text of C library asserts) as the claim
  // identifier; asserts lowered without a comment get the stringified guard.
  if (it->location.comment().empty())
    it->location.comment(from_expr(ns, "", old_guard));
  it->location.user_provided(true);
}

/*
  convert assert(cond) to assume(cond)
  preserving the original condition as a path constraint
*/
void goto_coveraget::replace_assert_to_assume(
  goto_programt::instructiont::targett &it)
{
  const expr2tc guard = it->guard;
  it->make_assumption(guard);
  it->location.property("replaced assertion");
  it->location.user_provided(true);
}

/*
  convert all assertions to assumptions
*/
void goto_coveraget::replace_all_asserts_to_assume()
{
  std::unordered_set<std::string> location_pool = {};
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      Forall_goto_program_instructions (it, goto_program)
      {
        std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->is_assert())
          replace_assert_to_assume(it);
      }
    }
}

/*
Algo:
- convert all assertions to false and enable multi-property
*/
void goto_coveraget::assertion_coverage()
{
  replace_all_asserts_to_guard(gen_false_expr(), true);
  total_assert = get_total_instrument();
  total_assert_ins = get_total_assert_instance();
  all_claims = get_total_cond_assert();
}

/*
Branch coverage applies to any control structure that can alter the flow of
execution, including:
- if-else
- switch-case
- Loops (for, while, do-while)
- try-catch-finally (not in c)
- Early exits (return, break, continue)
The goal of branch coverage is to ensure that all possible execution paths in
the program are tested.

The CBMC extends it to the entry of the function. So we will do the same.


Algo:
  1. convert assertions to true
  2. add false assertion add the beginning of the function and the branch()
*/
void goto_coveraget::branch_function_coverage()
{
  log_progress("Adding false assertions...");
  total_func_branch = 0;

  std::unordered_set<std::string> location_pool = {};
  // cmdline.arg[0]
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  std::unordered_set<int> catch_tgt_list;
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      auto in_verifying_files = [&](const goto_programt::instructiont &ins) {
        return location_pool.count(
                 get_filename_from_path(ins.location.file().as_string())) != 0;
      };

      // Add a false assert at the first instruction belonging to one of the
      // files under verification, to check if the function is entered.
      auto entry = std::find_if(
        goto_program.instructions.begin(),
        goto_program.instructions.end(),
        in_verifying_files);
      if (entry != goto_program.instructions.end())
        insert_assert(
          goto_program,
          entry,
          gen_false_expr(),
          "function entry: " + id2string(f_it->first));

      Forall_goto_program_instructions (it, goto_program)
      {
        if (!in_verifying_files(*it))
          continue;

        if (it->location.property().as_string() == "skipped")
          // this stands for the auxiliary condition/branch we added.
          continue;

        // convert assertions to true (or assume)
        if (
          it->is_assert() &&
          it->location.property().as_string() != "replaced assertion" &&
          it->location.property().as_string() != "instrumented assertion")
        {
          if (cov_assume_asserts)
            replace_assert_to_assume(it);
          else
            replace_assert_to_guard(gen_true_expr(), it, false);
        }

        // e.g. IF !(a > 1) THEN GOTO 3
        else if (it->is_goto() && !is_true(it->guard))
        {
          if (it->is_target())
            target_num = it->target_number;
          // assert(!(a > 1));
          // assert(a > 1);
          insert_assert(goto_program, it, it->guard);
          insert_assert(goto_program, it, gen_not_expr(it->guard));
        }
      }
    }

  // fix for branch coverage with kind/incr
  // It seems in kind/incr, the goto_functions used during the BMC is simplified
  // and incomplete
  total_func_branch = get_total_instrument();
  all_claims = get_total_cond_assert();

  // avoid Assertion `call_stack.back().goto_state_map.size() == 0' failed
  goto_functions.update();
}

// Walk an ASSIGN rhs / RETURN operand for short-circuit operators that
// the frontend did NOT lower to control flow. goto_sideeffects.cpp:160
// rewrites `||`/`&&` into an if-then-else GOTO chain ONLY when an
// operand has a side effect; with side-effect-free operands the `or`/
// `and` stays a flat boolean expression in one instruction and carries
// no GOTO guard, so the `it->is_goto()` arm below never sees it
// (ESBMC: "No branch detected" where solc instruments the operator as
// a 2-arm decision). For each such operator, emit the same 2-arm
// decision the GOTO arm produces, keyed on the short-circuit operand
// (side_1 — the operand that decides whether the rest is evaluated),
// recursing both sides to reach nested operators.
// Max folded short-circuit/ternary operands treated as decisions at ONE site.
// Phase 1 (runtime snapshots into tr/cnt) and Phase 2 (offline enumeration of
// the 2^K combinations) MUST apply this identically: if Phase 1 snapshots K
// operands that Phase 2 does not enumerate, the emitted path carries a depth
// that is short by K, so `cnt != depth` holds on EVERY real execution and the
// path becomes permanently uncoverable — and is then reported as PASSED, i.e.
// a false proof of unreachability. 2^12 = 4096 combinations per site is already
// far beyond any real Solidity expression; sites above it are left out of the
// decision set entirely (and reported) rather than half-instrumented.
static constexpr size_t SC_DECISION_MAX = 12;

static void collect_short_circuit_decisions(
  const expr2tc &e,
  const std::function<void(const expr2tc &)> &emit)
{
  if (is_nil_expr(e))
    return;
  if (is_or2t(e))
    emit(to_or2t(e).side_1);
  else if (is_and2t(e))
    emit(to_and2t(e).side_1);
  else if (is_if2t(e))
    // Solidity ternary `cond ? a : b`: lowered to a flat if2t SELECT
    // when both arms are side-effect-free.  solc-coverage instruments
    // the ternary's `cond` as a 2-arm decision; mirror that by emitting
    // the cond expression as a probe keyed on the same location.
    emit(to_if2t(e).cond);
  for (size_t i = 0; i < e->get_num_sub_exprs(); ++i)
  {
    const expr2tc *sub = e->get_sub_expr(i);
    if (sub != nullptr)
      collect_short_circuit_decisions(*sub, emit);
  }
}

bool goto_coveraget::edge_reaches_error_revert(
  goto_programt::const_targett it,
  goto_programt::const_targett end) const
{
  // Bounded straight-line walk. Stop at anything that changes control flow or
  // merges another edge in; only an unbroken run of straight-line instructions
  // that reaches an error call proves THIS edge reverts.
  for (size_t steps = 0; it != end && steps < 256; ++it, ++steps)
  {
    // A downstream join (some other edge targets this instruction) means we can
    // no longer attribute a later terminator to this edge alone.
    if (steps > 0 && it->is_target())
      return false;
    // Control-flow / terminating instructions break the straight-line run.
    if (
      it->is_goto() || it->is_return() || it->is_end_function() ||
      it->is_throw() || it->is_catch())
      return false;
    // A lowered `revert CustomError(...)` is a call to a `#sol_error` function.
    if (it->is_function_call() && is_code_function_call2t(it->code))
    {
      const expr2tc &fn = to_code_function_call2t(it->code).function;
      if (is_symbol2t(fn))
      {
        const symbolt *s = ns.lookup(to_symbol2t(fn).thename);
        if (s && !s->get_type().get("#sol_error").as_string().empty())
          return true;
      }
      // A non-error call is straight-line; keep walking.
    }
    // ASSIGN / DECL / DEAD / SKIP / LOCATION / OTHER / ATOMIC / ASSUME are
    // straight-line: keep walking.
  }
  return false;
}

void goto_coveraget::branch_coverage()
{
  log_progress("Adding false assertions...");
  total_branch = 0;
  // all_claims is the no-skip static universe, rebuilt every call
  // (Item 2c). covered_set/outpath start clean unless a path is given.
  all_claims.clear();
  covered_set.clear();
  covered_set_outpath.clear();

  // Cross-run covered-set (Item 2): load the persisted edge keys. A
  // missing/unreadable/empty file is treated as "nothing covered yet"
  // (first run). The path is still recorded so the run-end report
  // (bmc.cpp) merge-writes the accumulated set back.
  if (!covered_set_path.empty())
  {
    covered_set_outpath = covered_set_path;
    std::ifstream in(covered_set_path);
    if (in)
    {
      try
      {
        nlohmann::json j;
        in >> j;
        for (const auto &e : j.value("covered", nlohmann::json::array()))
          covered_set.emplace(
            e.at("cond").get<std::string>(), e.at("loc").get<std::string>());
      }
      catch (const std::exception &ex)
      {
        log_warning(
          "coverage-covered-set: ignoring unparseable {} ({})",
          covered_set_path,
          ex.what());
        covered_set.clear();
      }
    }
  }

  std::unordered_set<std::string> location_pool = {};
  // cmdline.arg[0]
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  std::unordered_set<int> catch_tgt_list;
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      Forall_goto_program_instructions (it, goto_program)
      {
        std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        // skip if it's not the verifying files
        // probably a library
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->location.property().as_string() == "skipped")
          // this stands for the auxiliary condition/branch we added.
          continue;

        // Emit one 2-arm decision (assert(cond) + assert(!cond)) for
        // `cond` at the current instruction, under the exact scoping,
        // edge-key, static-universe and cross-run covered-set rules the
        // GOTO-guard path uses, so a folded short-circuit operator and
        // a control-flow guard are counted identically.
        // `cond_reverts`/`neg_reverts`: the covered edge behind the
        // assert(cond) / assert(!cond) probe reverts (require failure / revert
        // CustomError). Stamp `sol_revert_edge` on that probe so the Foundry
        // generator emits vm.expectRevert(). Only set for GOTO decisions.
        auto emit_decision = [&](
                               const expr2tc &cond,
                               bool cond_reverts = false,
                               bool neg_reverts = false) {
          // Per-contract scoping (--contract C, Solidity): only instrument
          // decisions lexically declared inside contract C. The frontend
          // stamps each statement location with "sol_decl_contract" (its
          // declaring ContractDefinition, invariant across inheritance
          // merge-by-copy). Skipping the assert pair for C-foreign
          // decisions auto-scopes BOTH the denominator
          // (get_total_cond_assert counts instrumented asserts only) and
          // the numerator (reached_claims can only hit instrumented
          // asserts), so the percentage stays correct by construction.
          if (
            !scope_contract.empty() &&
            it->location.get("sol_decl_contract").as_string() != scope_contract)
            return;

          // Item 5-d: dependency exclusion. Drop the decision BEFORE the
          // all_claims.insert below, so an excluded contract's decisions
          // leave BOTH the denominator (static universe) and the
          // numerator (no assert => reached_claims can never hit it) —
          // exactly the "OZ in no denominator, no numerator" property.
          // Default mode never reaches here for foreign code (scope
          // filter above already skipped it), so this is a no-op there.
          if (
            !exclude_contracts.empty() &&
            exclude_contracts.count(
              it->location.get("sol_decl_contract").as_string()))
            return;

          // Edge keys (guard_str, location.as_string()) — exactly the
          // identity get_total_cond_assert() and the numerator
          // (bmc.cpp claim_sig) use, so universe / denominator /
          // numerator stay key-aligned. as_string() excludes custom
          // irep fields, so inheritance/modifier copies fold to one.
          const expr2tc neg = gen_not_expr(cond);
          const std::string loc = it->location.as_string();
          const std::pair<std::string, std::string> k_g(
            from_expr(ns, "", cond), loc);
          const std::pair<std::string, std::string> k_ng(
            from_expr(ns, "", neg), loc);

          // Static universe (Item 2c): every in-scope edge counts in
          // the denominator regardless of the covered-set skip below,
          // so skipping can never inflate coverage.
          all_claims.insert(k_g);
          all_claims.insert(k_ng);

          // Item 2b: an edge already witnessed P_SATISFIABLE in a prior
          // run (covered_set) is not re-instrumented — fewer SMT
          // obligations on re-runs. Sound: an instrumented assert is a
          // property obligation, not a path constraint, so omitting it
          // removes one observation only and perturbs no other branch;
          // the cross-run cover is monotone-∃ (a real witness stays
          // valid). Only true P_SATISFIABLE is ever written back.
          if (!covered_set.count(k_g))
          {
            insert_assert(goto_program, it, cond);
            if (cond_reverts)
              std::prev(it)->location.set("sol_revert_edge", true);
          }
          if (!covered_set.count(k_ng))
          {
            insert_assert(goto_program, it, neg);
            if (neg_reverts)
              std::prev(it)->location.set("sol_revert_edge", true);
          }
        };

        // convert assertions to true (or assume)
        if (
          it->is_assert() &&
          it->location.property().as_string() != "replaced assertion" &&
          it->location.property().as_string() != "instrumented assertion")
        {
          if (cov_assume_asserts)
            replace_assert_to_assume(it);
          else
            replace_assert_to_guard(gen_true_expr(), it, false);
        }

        // e.g. IF !(a > 1) THEN GOTO 3
        else if (
          it->is_goto() && !is_true(it->guard) &&
          !guard_folds_to_constant(it->guard))
        {
          if (it->is_target())
            target_num = it->target_number;
          // Revert fidelity: classify which edge reverts BEFORE instrumenting
          // (target/fall-through still point at the original successors). A
          // probe assert(P) fails when P is false, so assert(it->guard) covers
          // the FALL-THROUGH edge and assert(!it->guard) the GOTO-taken edge.
          const bool taken_reverts = edge_reaches_error_revert(
            it->get_target(), goto_program.instructions.end());
          const bool fall_reverts = edge_reaches_error_revert(
            std::next(it), goto_program.instructions.end());
          // Only tag when exactly one edge reverts (both/neither -> no tag).
          emit_decision(
            it->guard,
            /*cond_reverts=*/fall_reverts && !taken_reverts,
            /*neg_reverts=*/taken_reverts && !fall_reverts);
        }

        // Pure short-circuit ||/&& folded into an ASSIGN rhs / RETURN
        // operand (no GOTO — see collect_short_circuit_decisions above).
        // solc instruments every such operator as a 2-arm decision;
        // without this ESBMC reports "No branch detected" for e.g.
        // `return a == 0 || b == 1;`.
        else if (it->is_assign())
          collect_short_circuit_decisions(
            to_code_assign2t(it->code).source, emit_decision);

        else if (it->is_return())
          collect_short_circuit_decisions(
            to_code_return2t(it->code).operand, emit_decision);
      }
    }

  // Denominator = the no-skip static universe built in the loop above:
  // every in-scope decision edge keyed by (condition,
  // location.as_string()) in a std::set, so inheritance/modifier
  // physical copies fold to one source identity and override/sibling
  // decisions stay distinct (different source line). This decouples the
  // denominator from what was actually instrumented (Item 2c): when the
  // covered-set skip omits an assert, all_claims is unaffected, so
  // coverage % can never be spuriously inflated. The numerator
  // (reached_claims, matched against all_claims) uses the same key.
  // When no covered-set is given this is identical to the previous
  // get_total_cond_assert() result (same keys, same dedup), so the
  // no-path path is behaviour-preserving. Other coverage modes
  // (assertion/k-path/branch-function) keep get_total_cond_assert() /
  // get_total_instrument() by design.
  total_branch = static_cast<size_t>(all_claims.size());
  // Signal-safe snapshot for the timeout/term handlers ("data even on
  // UNKNOWN"). Set here, at instrumentation time, before any solve can
  // be killed. covered_set_outpath is set during option parsing (well
  // before this), so covered_set_mode is final here.
  total_branch_atomic.store(total_branch, std::memory_order_relaxed);
  covered_set_mode.store(
    !covered_set_outpath.empty(), std::memory_order_relaxed);
  live_reached.store(0, std::memory_order_relaxed);
  covered_run.store(0, std::memory_order_relaxed);
  branch_cov_active.store(true, std::memory_order_relaxed);

  // avoid Assertion `call_stack.back().goto_state_map.size() == 0' failed
  goto_functions.update();
}

// Post-simplification depth of an expression tree, capped early once the
// caller's threshold is exceeded. Used to gate emission of the structural
// witness (issue #4325).
static size_t expr_depth(const expr2tc &e, size_t cap)
{
  if (is_nil_expr(e))
    return 0;
  size_t n = e->get_num_sub_exprs();
  if (n == 0)
    return 1;
  size_t d = 0;
  for (size_t i = 0; i < n; ++i)
  {
    const expr2tc *sub = e->get_sub_expr(i);
    if (sub == nullptr)
      continue;
    d = std::max(d, expr_depth(*sub, cap));
    if (d > cap)
      return d + 1;
  }
  return 1 + d;
}

// True when the conjunction of `atoms` ((guard, polarity) pairs) is
// unsatisfiable because some scalar variable is pinned to an empty integer
// range — e.g. `x==1 && x==2` or `x<0 && x>10`. This catches mutually
// exclusive comparisons on one variable, which the same-atom check (a guard
// and its negation) and simplify() both miss.
//
// Sound by construction: an atom only counts when it is exactly
// `var <rel> int-const` (after peeling negations and normalising operand
// order), and bounds are tracked over unbounded integers, so the admissible
// set is over-approximated and a satisfiable conjunction is never reported
// unsat. Like the same-atom check it treats a guard as a stable value across
// the prefix window, so a variable reassigned between two branches is not
// specially handled here.
static bool
kpath_atoms_comparison_unsat(const std::vector<std::pair<expr2tc, bool>> &atoms)
{
  enum relt
  {
    R_EQ,
    R_NE,
    R_LT,
    R_GT,
    R_LE,
    R_GE
  };

  // Normalise `g` to (symbol-name, rel, constant), folding Boolean
  // negations into `pol`. Returns false unless `g` is a comparison between
  // a scalar symbol and an integer constant (either operand order).
  auto normalise =
    [](expr2tc g, bool pol, irep_idt &var, relt &rel, BigInt &c) -> bool {
    while (is_not2t(g))
    {
      pol = !pol;
      g = to_not2t(g).value;
    }

    relt base;
    expr2tc s1, s2;
    if (is_equality2t(g))
    {
      base = R_EQ;
      s1 = to_equality2t(g).side_1;
      s2 = to_equality2t(g).side_2;
    }
    else if (is_notequal2t(g))
    {
      base = R_NE;
      s1 = to_notequal2t(g).side_1;
      s2 = to_notequal2t(g).side_2;
    }
    else if (is_lessthan2t(g))
    {
      base = R_LT;
      s1 = to_lessthan2t(g).side_1;
      s2 = to_lessthan2t(g).side_2;
    }
    else if (is_greaterthan2t(g))
    {
      base = R_GT;
      s1 = to_greaterthan2t(g).side_1;
      s2 = to_greaterthan2t(g).side_2;
    }
    else if (is_lessthanequal2t(g))
    {
      base = R_LE;
      s1 = to_lessthanequal2t(g).side_1;
      s2 = to_lessthanequal2t(g).side_2;
    }
    else if (is_greaterthanequal2t(g))
    {
      base = R_GE;
      s1 = to_greaterthanequal2t(g).side_1;
      s2 = to_greaterthanequal2t(g).side_2;
    }
    else
      return false;

    // Require one symbol and one integer-constant operand; put the symbol
    // on the left, mirroring the relation if the operands were swapped.
    bool swapped = false;
    if (is_constant_int2t(s1) && is_symbol2t(s2))
    {
      std::swap(s1, s2);
      swapped = true;
    }
    else if (!(is_symbol2t(s1) && is_constant_int2t(s2)))
      return false;

    var = to_symbol2t(s1).get_symbol_name();
    c = to_constant_int2t(s2).value;

    // Swapping operands mirrors the relation (c < x  <=>  x > c).
    if (swapped)
    {
      switch (base)
      {
      case R_LT:
        base = R_GT;
        break;
      case R_GT:
        base = R_LT;
        break;
      case R_LE:
        base = R_GE;
        break;
      case R_GE:
        base = R_LE;
        break;
      default:
        // EQ / NE are symmetric
        break;
      }
    }

    // Negative polarity takes the complement relation.
    if (!pol)
    {
      switch (base)
      {
      case R_EQ:
        base = R_NE;
        break;
      case R_NE:
        base = R_EQ;
        break;
      case R_LT:
        base = R_GE;
        break;
      case R_GE:
        base = R_LT;
        break;
      case R_GT:
        base = R_LE;
        break;
      case R_LE:
        base = R_GT;
        break;
      }
    }

    rel = base;
    return true;
  };

  struct boundst
  {
    bool has_lo = false, has_hi = false;
    BigInt lo, hi;
    std::set<BigInt> excluded;
  };
  std::map<irep_idt, boundst> vars;

  auto raise_lo = [](boundst &b, const BigInt &v) {
    if (!b.has_lo || v > b.lo)
    {
      b.lo = v;
      b.has_lo = true;
    }
  };
  auto lower_hi = [](boundst &b, const BigInt &v) {
    if (!b.has_hi || v < b.hi)
    {
      b.hi = v;
      b.has_hi = true;
    }
  };

  for (const auto &[g, pol] : atoms)
  {
    irep_idt var;
    relt rel;
    BigInt c;
    if (!normalise(g, pol, var, rel, c))
      continue;

    boundst &b = vars[var];
    switch (rel)
    {
    case R_EQ:
      raise_lo(b, c);
      lower_hi(b, c);
      break;
    case R_NE:
      b.excluded.insert(c);
      break;
    case R_LT:
      lower_hi(b, c - 1);
      break;
    case R_LE:
      lower_hi(b, c);
      break;
    case R_GT:
      raise_lo(b, c + 1);
      break;
    case R_GE:
      raise_lo(b, c);
      break;
    }
  }

  for (const auto &[name, b] : vars)
  {
    (void)name;
    if (b.has_lo && b.has_hi)
    {
      if (b.lo > b.hi)
        // empty integer range, e.g. x<0 && x>10
        return true;
      // Single admissible point that is also excluded, e.g. x==1 && x!=1.
      if (b.lo == b.hi && b.excluded.count(b.lo))
        return true;
    }
  }
  return false;
}

/*
k-path coverage (Phase 1 — see GitHub issue #4325).

For each branching `IF g GOTO L`, emit one coverage goal per combination of
the last (n-1) prior branch directions × the two outcomes of the current
branch. Each goal is `assert(!witness)` where `witness = d_1 ∧ … ∧ d_k`
with each d_i either a prior branch guard or its negation; multi_property
marks a goal as reached when the assertion is falsifiable, i.e. when the
corresponding path is feasible. This mirrors the existing branch_coverage
inversion convention.

Bounded by the textual order of branches within a function (cheap and
deterministic). Joins make this an *over-approximation* of true path
coverage — some witnesses may be infeasible and stay uncovered, which is
correct under the spanning-set scoring proposed in #4325.

Goal count per branch: 2^min(prefix_size+1, n). Capped per function by
`k_path_max_goals`; on overflow the instrumentation aborts with an
actionable error rather than silently truncating (decision locked in #4325).
*/
void goto_coveraget::k_path_coverage()
{
  log_progress("Adding k-path coverage assertions (n={})...", k_path_n);
  total_kpath = 0;
  total_kpath_spanning = 0;
  k_path_spanning_redundant.clear();
  k_path_spanning_sett spanning;

  // Defense-in-depth: parseoptions rejects N==0 and N>30 at the CLI, but
  // re-check here in case the method is invoked via another code path.
  // 30 keeps `1 << pdepth` well below the size_t shift limit and below
  // any reasonable goal cap.
  static constexpr size_t K_PATH_N_MAX = 30;
  if (k_path_n == 0 || k_path_n > K_PATH_N_MAX)
  {
    log_error(
      "--k-path-coverage requires 1 <= N <= {} (got {})",
      K_PATH_N_MAX,
      k_path_n);
    abort();
  }

  std::unordered_set<std::string> location_pool = {};
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available || f_it->first == "__ESBMC_main")
      continue;

    goto_programt &goto_program = f_it->second.body;
    if (filter(f_it->first, goto_program))
      continue;

    // Sliding window of the last (n-1) prior branch guards in textual order.
    // Reset per function: each function is its own k-path scope (#4325).
    std::deque<expr2tc> prefix;
    size_t function_goals = 0;

    Forall_goto_program_instructions (it, goto_program)
    {
      std::string cur_filename =
        get_filename_from_path(it->location.file().as_string());
      if (location_pool.count(cur_filename) == 0)
        continue;

      if (it->location.property().as_string() == "skipped")
        continue;

      // Mirror branch_coverage: neutralise existing assertions so they don't
      // confuse multi_property_check.
      if (
        it->is_assert() &&
        it->location.property().as_string() != "replaced assertion" &&
        it->location.property().as_string() != "instrumented assertion")
      {
        if (cov_assume_asserts)
          replace_assert_to_assume(it);
        else
          replace_assert_to_guard(gen_true_expr(), it, false);
        continue;
      }

      // Conditional forward branch. Backward unconditional gotos (loop
      // back-edges) carry guard=true and are skipped here; iteration
      // semantics are picked up later by ESBMC's --unwind unrolling.
      if (it->is_goto() && !is_true(it->guard))
      {
        if (it->is_target())
          target_num = it->target_number;

        const expr2tc current_guard = it->guard;
        // pdepth is bounded by k_path_n - 1 <= K_PATH_N_MAX - 1 = 29
        // (enforced above), so the shift below cannot overflow. Assert as
        // a tripwire — silent overflow would be unsound.
        const size_t pdepth = std::min(prefix.size(), k_path_n - 1);
        assert(pdepth < 30 && "pdepth bounded by parseoptions cap");
        const size_t pcombos = size_t(1) << pdepth;
        const size_t branch_goals = 2 * pcombos;

        if (
          branch_goals > k_path_max_goals ||
          function_goals > k_path_max_goals - branch_goals)
        {
          log_error(
            "k-path coverage: per-function goal count would exceed "
            "--k-path-max-goals={} in '{}'. Lower --k-path-coverage=N "
            "(currently {}) or raise --k-path-max-goals.",
            k_path_max_goals,
            id2string(f_it->first),
            k_path_n);
          abort();
        }

        // The deque is trimmed to ≤ (n-1) entries at the bottom of every
        // branch iteration, so its current contents are exactly the active
        // prefix.
        std::vector<expr2tc> active(prefix.begin(), prefix.end());

        for (size_t mask = 0; mask < pcombos; ++mask)
        {
          // Build the prefix witness for this direction mask, while
          // tracking (stored-guard, polarity) pairs so we can drop mask
          // combinations that are unsat by construction.
          //
          // ESBMC's `simplify` recognises 2-term `p ∧ ¬p` but does not
          // fold chained forms like `p ∧ q ∧ ¬p` to FALSE — it would
          // instrument a tautological `assert(¬(p ∧ q ∧ ¬p))` that can
          // never be falsified, permanently inflating the denominator.
          // Catching this at construction time is sound (we only drop
          // witnesses we can prove unsat by syntactic structure) and
          // preserves the single-term behaviour of the simplifier.
          //
          // Phase-1 limitation: this only catches *syntactic* same-atom
          // contradictions (same stored guard with opposing polarities).
          // Semantically contradictory pairs across different stored
          // expressions — e.g. `(x == 1) ∧ (x == 2)` from successive
          // switch-case branches — require comparison-domain reasoning,
          // handled per-goal below by kpath_atoms_comparison_unsat.
          expr2tc pwit;
          std::vector<std::pair<expr2tc, bool>> atoms;
          atoms.reserve(pdepth);
          bool contradictory = false;
          for (size_t i = 0; i < pdepth; ++i)
          {
            const bool pol = (mask & (size_t(1) << i)) != 0;
            for (const auto &[h, p] : atoms)
            {
              if (h == active[i] && p != pol)
              {
                contradictory = true;
                break;
              }
            }
            if (contradictory)
              break;
            atoms.emplace_back(active[i], pol);
            expr2tc d = pol ? active[i] : gen_not_expr(active[i]);
            pwit = is_nil_expr(pwit) ? d : gen_and_expr(pwit, d);
          }
          if (contradictory)
            continue;

          // Emit one goal per current direction. Skip the direction if
          // it would contradict an atom already in the prefix.
          const expr2tc current_neg = gen_not_expr(current_guard);
          for (size_t cd = 0; cd < 2; ++cd)
          {
            const bool cdir_pol = (cd == 0);
            bool cdir_conflict = false;
            for (const auto &[h, p] : atoms)
            {
              if (h == current_guard && p != cdir_pol)
              {
                cdir_conflict = true;
                break;
              }
            }
            if (cdir_conflict)
              continue;

            const expr2tc &cdir = cdir_pol ? current_guard : current_neg;
            expr2tc full = is_nil_expr(pwit) ? cdir : gen_and_expr(pwit, cdir);
            simplify(full);

            if (is_false(full))
              continue;
            if (is_true(full))
              continue;

            if (expr_depth(full, k_path_witness_depth) > k_path_witness_depth)
            {
              // Phase 1: drop witnesses past the depth cap. The hashed
              // ghost-flag fallback for deep prefixes is Phase 2 (#4325).
              continue;
            }

            // Drop goals whose conjunction pins a variable to an empty
            // integer range (e.g. x==1 && x==2). simplify() does not do this
            // cross-term reasoning, so otherwise the tautological
            // assert(!witness) can never be falsified and permanently
            // inflates the spanning denominator. Built after the cheaper
            // checks above so goals they skip don't pay for the vector.
            std::vector<std::pair<expr2tc, bool>> goal_atoms = atoms;
            goal_atoms.emplace_back(current_guard, cdir_pol);
            if (kpath_atoms_comparison_unsat(goal_atoms))
              continue;

            expr2tc neg_full = gen_not_expr(full);
            simplify(neg_full);

            std::string idf = from_expr(ns, "", full);
            insert_assert(goto_program, it, neg_full, idf);

            // Record the goal's full atom multiset (prefix + current
            // direction) so the spanning-set analysis can drop subsumed
            // emissions from the coverage denominator.
            spanning.add_goal(
              std::move(goal_atoms), idf, it->location.as_string());

            ++function_goals;
          }
        }

        prefix.push_back(current_guard);
        if (prefix.size() > k_path_n - 1)
          prefix.pop_front();
      }
    }
  }

  total_kpath = get_total_instrument();
  all_claims = get_total_cond_assert();

  // Soundness invariant: each insert_assert call above paired with
  // exactly one spanning.add_goal call, so the number of goals tracked
  // in the spanning analysis must equal the number of instrumented
  // assertions counted in the goto programs. A divergence means the
  // emission path diverged from the spanning bookkeeping (e.g. a future
  // edit added an insert_assert without the matching add_goal, or vice
  // versa) and the spanning-set denominator would be silently wrong.
  // ESBMC is a verifier — we abort rather than report an unsound
  // coverage percentage.
  if (spanning.total() != static_cast<size_t>(total_kpath))
  {
    log_error(
      "k-path coverage: internal invariant violated — spanning.total()={} "
      "but get_total_instrument()={}. Each instrumented assertion must "
      "have a matching spanning.add_goal entry. Aborting rather than "
      "report an unsound coverage percentage.",
      spanning.total(),
      total_kpath);
    abort();
  }

  // Compute the spanning-set after every goal has been collected. The
  // resulting size is the Phase-2 denominator; redundant_claims feeds the
  // JSON `feasibility` field.
  //
  // Secondary invariant: the simplifier never collapses two semantically
  // distinct witnesses to the same idf string, so spanning_size_ is
  // bounded above by all_claims.size() + |redundant|, which is what
  // allows the bmc.cpp coverage cap to make sense. Any future change
  // that reuses an idf across distinct witnesses or alters from_expr()
  // formatting must preserve this 1:1 mapping or the percentage will
  // silently deflate.
  spanning.finalize();
  total_kpath_spanning = spanning.spanning_size();
  for (const auto &claim : all_claims)
    if (spanning.is_redundant(claim.first, claim.second))
      k_path_spanning_redundant.insert(claim);

  goto_functions.update();
}

// Count complete paths in a goto program WITHOUT instrumenting it, using the
// same traversal rules as the enumerating DFS: conditional GOTOs fan out 2-way,
// each loop head gets its own back-edge budget, folded short-circuit operands
// in ASSIGN/RETURN fan out 2^K, and RETURN / END_FUNCTION / a `#sol_error` call
// are terminators.
//
// Sole purpose is measurement: it is run on a snapshot of each unit's body
// taken BEFORE internal calls are expanded, so the ratio against the real
// enumeration quantifies how much expansion multiplies paths — a cost that was
// declared when expansion was adopted and until now had no number attached.
//
// Duplicated traversal logic is a drift hazard, so the caller checks the one
// case where the two must agree exactly: a unit into which nothing was expanded
// must produce the same count from both. A mismatch there is a hard failure, so
// this measurement is itself measured rather than self-certified.
static size_t count_paths_no_instrument(
  const goto_programt &p,
  const namespacet &ns,
  size_t unwind,
  size_t cap,
  bool &hit_cap)
{
  hit_cap = false;
  if (p.instructions.empty())
    return 0;

  auto is_err_call = [&ns](goto_programt::const_targett i) {
    if (!i->is_function_call() || !is_code_function_call2t(i->code))
      return false;
    const expr2tc &fn = to_code_function_call2t(i->code).function;
    if (!is_symbol2t(fn))
      return false;
    const symbolt *s = ns.lookup(to_symbol2t(fn).thename);
    return s && !s->get_type().get("#sol_error").as_string().empty();
  };

  using becntt = std::map<unsigned, unsigned>;
  std::vector<std::pair<goto_programt::const_targett, becntt>> stack;
  stack.push_back({p.instructions.begin(), becntt{}});
  size_t paths = 0, pushes = 0;
  const size_t push_cap = 50 * cap + 100000;

  while (!stack.empty())
  {
    auto [pc, becnt] = stack.back();
    stack.pop_back();
    while (true)
    {
      if (pc == p.instructions.end() || pc->is_end_function())
      {
        if (pc != p.instructions.end())
          ++paths;
        break;
      }
      if (is_err_call(pc))
      {
        ++paths;
        break;
      }
      if (pc->is_return() && is_code_return2t(pc->code))
      {
        size_t rk = 0;
        if (pc->location.property().as_string() != "skipped")
          collect_short_circuit_decisions(
            to_code_return2t(pc->code).operand, [&](const expr2tc &) { ++rk; });
        paths += (rk > 0 && rk <= SC_DECISION_MAX) ? (size_t(1) << rk) : 1;
        break;
      }
      if (pc->is_goto())
      {
        const bool back = pc->is_backwards_goto();
        if (is_true(pc->guard))
        {
          if (back)
          {
            const unsigned key = pc->get_target()->target_number;
            if (becnt[key] >= unwind)
              break;
            ++becnt[key];
          }
          pc = pc->get_target();
          continue;
        }
        bool take = true;
        becntt becnt_taken = becnt;
        if (back)
        {
          const unsigned key = pc->get_target()->target_number;
          if (becnt_taken[key] >= unwind)
            take = false;
          else
            ++becnt_taken[key];
        }
        if (take)
        {
          if (++pushes > push_cap)
          {
            hit_cap = true;
            break;
          }
          stack.push_back({pc->get_target(), becnt_taken});
        }
        pc = std::next(pc);
        continue;
      }
      if (pc->is_assign() && pc->location.property().as_string() != "skipped")
      {
        size_t k = 0;
        collect_short_circuit_decisions(
          to_code_assign2t(pc->code).source, [&](const expr2tc &) { ++k; });
        if (k > 0 && k <= SC_DECISION_MAX)
        {
          for (uint64_t m = 0; m < (uint64_t(1) << k); ++m)
          {
            if (++pushes > push_cap)
            {
              hit_cap = true;
              break;
            }
            stack.push_back({std::next(pc), becnt});
          }
          break;
        }
      }
      pc = std::next(pc);
    }
    if (paths > cap)
    {
      hit_cap = true;
      break;
    }
    if (hit_cap)
      break;
  }
  return paths;
}

// ---------------------------------------------------------------------------
// Selective inliner for Solidity complete-path coverage.
//
// The methodology fixes two separate things that are easy to conflate:
//   * what is a UNIT (does it get a path set of its own)? -> a public/external
//     function, because that is what an external caller can invoke;
//   * does a CALL expand into the caller's path identity? -> an internal call
//     expands, an external call does not (its success/failure is the branch
//     point instead).
// The two answers are independent: a `public` function that is also called
// internally is BOTH a unit of its own (entered from outside, free arguments)
// AND expanded into its internal caller's paths (entered with computed
// arguments). Both descriptions are needed and they describe different input
// spaces.
//
// Expansion must be PHYSICAL — splice the callee's body into the caller —
// rather than a cross-function walk of the enumerator, because `tr`/`cnt` are
// per-function ghost symbols: a callee left as a call updates ITS OWN
// accumulator, so its decisions are invisible to the caller's path number.
// Splicing first and instrumenting second makes the caller's own accumulator
// record the whole expanded decision sequence, with no shared-state
// bookkeeping.
//
// goto_inlinet's own inliners cannot be used directly: `full` inlining drags in
// every c2goto library model, and its recursion handling rewrites the recursive
// call to a SKIP — deleting a real behaviour instead of bounding it. So the
// call-selection policy and the recursion bound live here, and only the tested
// mechanical pieces (parameter_assignments / replace_return) are reused.
class sol_path_inlinet : public goto_inlinet
{
public:
  sol_path_inlinet(
    goto_functionst &_goto_functions,
    optionst &_options,
    const namespacet &_ns)
    : goto_inlinet(_goto_functions, _options, _ns), sol_ns(_ns)
  {
  }

  // Kept for the R0 event probe below: this is the only place that still has
  // the callee's identity in hand.
  const namespacet &sol_ns;

  // Splice `f`'s body over the FUNCTION_CALL at `target`. On return `target`
  // names the instruction FOLLOWING the spliced region, so a single pass
  // expands exactly the calls that were present when the pass began — that is
  // what makes "one pass == one level of call depth" true, and hence what
  // bounds recursion.
  void expand_here(
    goto_programt &dest,
    goto_programt::targett &target,
    const goto_functiont &f)
  {
    // Copy what we need out of the call before the instruction is overwritten.
    const code_function_call2t &call = to_code_function_call2t(target->code);
    const expr2tc lhs = call.ret;
    const std::vector<expr2tc> args = call.operands;

    // ---- R0 EVENT RUNG, capture-point probe ----
    //
    // THIS is the moment: the callee is still named, the call is still a
    // FUNCTION_CALL, and its position in the caller is `target`. One line
    // below, `target->type = LOCATION` and the code is cleared, after which no
    // consumer can tell an emit from any other erased call -- MEASURED, three
    // censuses downstream all read zero.
    //
    // The discriminator is a flag the front end sets on the event's SYMBOL at
    // declaration, because that is the last point where the AST still says
    // EventDefinition; by here every event looks like an ordinary function
    // with an empty body.
    if (is_symbol2t(call.function))
    {
      const irep_idt fid = to_symbol2t(call.function).thename;
      const symbolt *cs = sol_ns.get_context().find_symbol(fid);
      if (cs != nullptr && cs->get_type().get_bool("sol_event"))
      {
        // CARRY THE IDENTITY ONTO THE SURVIVING INSTRUCTION. `target` is not
        // deleted below -- it becomes a LOCATION and stays in place, in
        // program order -- so a stamp written here is exactly where the path
        // walk will step over it. This is the pattern that WORKS on this side
        // of the pipeline (`sol_path_inlined`, three lines further down,
        // stamps GOTO instruction locations the same way); the front-end
        // equivalent was measured not to survive goto conversion.
        target->location.set("sol_emit_name", id2string(fid));
        log_status(
          "--solidity-path-coverage: EXPANDING AN EMIT: {} at {}",
          id2string(fid),
          target->location.as_string());
      }
    }

    goto_programt tmp2;
    tmp2.copy_from(f.body);
    assert(tmp2.instructions.back().is_end_function());
    tmp2.instructions.back().type = LOCATION;
    replace_return(tmp2, lhs);

    goto_programt tmp;
    parameter_assignments(
      tmp2.instructions.front().location, to_code_type(f.type), args, tmp);
    // parameter_assignments emits each formal as an OTHER instruction carrying
    // a code_decl. goto_symext::symex_other has no case for code_decl and
    // aborts the run ("unexpected statement: code_decl") — measured. The
    // existing inliners get away with it only because they run before the pass
    // that normalises this; we run after goto conversion, so emit the real
    // instruction kind.
    Forall_goto_program_instructions (pit, tmp)
      if (
        pit->type == OTHER && !is_nil_expr(pit->code) &&
        is_code_decl2t(pit->code))
        pit->type = DECL;
    tmp.destructive_append(tmp2);

    // Flag every spliced instruction. The exit classifier asks "did this path
    // run the function epilogue?" via a symbol name that an inlined callee's
    // own epilogue also matches; without the flag, a caller's bare revert edge
    // that happened to walk through an inlined callee's epilogue would be
    // reported `normal` instead of `undetermined`.
    Forall_goto_program_instructions (iit, tmp)
      iit->location.set("sol_path_inlined", true);

    target->type = LOCATION;
    target->code = expr2tc();
    goto_programt::targett next_target(target);
    ++next_target;
    dest.instructions.splice(next_target, tmp.instructions);
    target = next_target;
  }
};

// ---- Can a box bound be EXPRESSED on this coordinate? ----
//
// A WHITELIST, deliberately, and the direction is the whole point. Resolving a
// coordinate name succeeds for plenty of things this stage cannot bound: a
// `string` state variable, a contract/interface handle, a struct. Every bound
// is built as `coord <= constant` / `coord >= constant` with the constant
// carrying the coordinate's own type, so a non-integer coordinate produces a
// malformed comparison that the SMT layer then dies on — measured as
// "Projecting from non-tuple based AST" and as a "Tuple AST mismatch"
// assertion, both SIGABRT. An abort turns a recordable refusal into a core
// dump, which unattended is the difference between a datum and a lost run.
//
// So the test is "is it one of the kinds we can bound?" and NOT "is it one of
// the kinds we know to be broken". Unrecognised types are unbounded in number:
// three projects fell over on three DIFFERENT ones (mapping, string, calldata
// struct) with the identical failure shape, and a blacklist would have caught
// at most the one that was written down. A new bounded type is one line here;
// a new unbounded type costs a named refusal, not a crash.
//
// Unsigned bit-vectors and BOOL. Address, bytesN and every Solidity integer
// lower to an unsigned bit-vector, so that arm covers the coordinates that have
// ever been measured.
//
// ---- S5: `bool` IS EXPRESSIBLE, BUT NOT AS AN INTERVAL ----
//
// It used to be refused here, and the stated reason was right about the shape
// and wrong about the conclusion: a two-point domain has no interval to
// measure, so the honest form is an EQUALITY coordinate. That form exists —
// `{0,1}` has only four subsets, so lo/hi/holes collapse to an allowed set
// `S ⊆ {0,1}` and the constraint is `OR over v in S of (c == v)`, which is
// exact rather than an approximation. Refusing it cost the class a
// flag-setting function is entirely about.
//
// WHAT EVERY CALLER OWES IN RETURN, because this predicate no longer implies
// it: a bool coordinate may NOT have `>=` / `<=` / `>` / `<` built on it. Those
// arms of smt_conv fall through to `assert(is_signedbv_type(...))`
// (src/solvers/smt/smt_conv.cpp 2494 / 2525 / 2556 / 2587) — SIGABRT, which is
// exactly the class of failure this whitelist exists to prevent. Nor may a
// `constant_int2tc` be built on a bool type; `gen_true_expr()` /
// `gen_false_expr()` are the only constants of that type this file makes.
//
// Stage 3's per-variable ladder therefore takes `coord_expressible` as its
// EQUALITY gate and derives its interval gate as `equality_ok &&
// !is_bool_type(vt)`. Widening this test without that split un-gates the
// ordering rungs and crashes; the two changes are one change.
//
// One further trap, recorded because the type system actively misleads here:
// `bool_type2t::get_width()` returns 8 (irep2_type.cpp — "the byte representing
// memory model"). That is correct for its own purpose and must NOT be changed,
// but it means a naive `2^width - 1` type-range check admits `[0, 255]` for a
// bool. `path_cov_fits_type` and certify's copy of it both special-case bool to
// `[0, 1]` for that reason.
static bool coord_expressible(const type2tc &t, std::string &why)
{
  if (is_unsignedbv_type(t))
    return true;
  if (is_bool_type(t))
    return true;
  // SIGNED IS REFUSED, and it was NOT before. Letting it in was my own hole and
  // it is a live false certificate, reproduced on a must-flip pair that differs
  // in one token:
  //
  //   uint256 a, box a in [0, 2^256-1]  ->  VERIFICATION FAILED   (correct: the
  //                                        box is the whole type and holds
  //                                        inputs of both paths)
  //   int256  a, the SAME decimal box   ->  VERIFICATION SUCCESSFUL
  //
  // Because the bound is built with constant_int2tc on the coordinate's own
  // type and the comparison is signedness-aware: on a signed 256-bit type the
  // decimal 2^256-1 is all ones, i.e. -1 under bvsle, so `a >= 0 && a <= -1` is
  // UNSATISFIABLE and every exit assert holds for want of an execution.
  //
  // Note what this defeats: the empty-box guard added hours earlier compares
  // the spec's DECIMAL lo and hi, and 0 <= 2^256-1 decimally, so it never
  // fires. The box is empty in the SOLVER and non-empty in the SPEC -- exactly
  // the gap between the two readings that the guard was written without
  // considering.
  //
  // Refusing is the fail-closed direction and costs nothing measured: every
  // coordinate any real contract has produced so far is uint256, address or
  // bytesN, all unsigned. Supporting signed properly is not a whitelist entry,
  // it is bound VALIDATION -- the spec's decimal bounds have to be checked
  // against the coordinate's own signed range ([-2^255, 2^255-1] for int256)
  // and refused when out of it. That is a separate change with its own
  // criteria; it must not be smuggled in by widening this test back.
  if (is_signedbv_type(t))
    why =
      "it resolves to a SIGNED bit-vector. A bound is built as a constant "
      "of the coordinate's own type and compared with a signedness-aware "
      "predicate, so a decimal bound above the signed maximum wraps to a "
      "negative value and can make the assumption UNSATISFIABLE -- which "
      "certifies vacuously while the printed box still looks non-empty "
      "(reproduced: the same box that is refuted on uint256 is 'certified' "
      "on int256). Supporting this needs the bounds validated against the "
      "signed range, not a wider type test";
  else if (is_array_type(t))
    why =
      "it resolves to an ARRAY — the frontend lowers strings, bytes, "
      "mappings and dynamic arrays to arrays, and a scalar interval is not "
      "expressible on one";
  else if (is_struct_type(t) || is_union_type(t))
    why =
      "it resolves to an AGGREGATE (struct / contract instance) — bounding "
      "it would need a coordinate per field, which is a different "
      "coordinate kind, not a wider interval";
  else if (is_pointer_type(t))
    why =
      "it resolves to a POINTER (a contract or interface handle) — the "
      "value is an address in the model's own allocator, not an input a "
      "test can set";
  else
    why =
      "it does not resolve to a bit-vector, which is the only kind this "
      "stage can put a bound on";
  return false;
}

/*
Solidity complete-path coverage (entry->exit path coverage for test gen).

The UNIT is a public/external function — what an external caller can actually
invoke. Internal/private helpers are not units: an internal call is EXPANDED
into its caller (see sol_path_inlinet above), so the callee's decisions are part
of the caller's path identity, and the helper gets no path set of its own. A
`public` function that is also called internally is therefore described twice,
by design: once as a unit entered from outside with free arguments, and once as
expanded decisions inside each internal caller's paths, where its arguments are
computed. Those are different input spaces and need different tests.

Consequence for reading the totals: the same source line can appear in more than
one unit's path set, so the path total is NOT a count of distinct code paths in
the contract — it is the sum over units of that unit's complete paths.

For each unit:
  Phase 1: one integer path-number accumulator `tr`. At function entry
           `tr = 1`; before every decision `tr = tr*2 + guard_value`. A single
           scalar records the whole decision sequence in order and survives
           loop unrolling (symex re-runs the update each iteration), so it
           handles loops without per-occurrence ghost symbols. The guard VALUE
           (not the direction) is accumulated; the path condition supplies the
           direction, so no CFG edge-splitting is needed.
  Phase 2: bounded DFS of complete entry->exit decision sequences. Each path's
           number enc mirrors tr (start 1; enc*2+1 for the guard-true/taken
           successor, enc*2+0 for guard-false/fallthrough). At END_FUNCTION
           emit `assert(tr != enc)`, falsified exactly on that path (enc is
           unique, so all path asserts can sit before the single END_FUNCTION).
           enc goes into the claim comment for a unique claim_sig
           (bmc.cpp:2000 is otherwise unsound). Loops: a back-edge is followed
           at most path_cov_unwind times per path, so paths are enumerated up
           to that many iterations, aligned with the symex --unwind bound;
           `assert(tr != enc)` fires per distinct iteration count.

Revert exits (require/revert in public/external functions) lower to a real
branch that END_FUNCTION captures, so they are enumerated as distinct paths.
*/
void goto_coveraget::solidity_path_coverage()
{
  log_progress("Adding Solidity complete-path coverage assertions...");
  if (cov_context == nullptr)
  {
    log_error(
      "--solidity-path-coverage: no context available to create ghost "
      "snapshot symbols (dispatch must set cov_context). Aborting rather "
      "than silently producing no coverage.");
    abort();
  }

  std::unordered_set<std::string> location_pool = {};
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  // Cross-run covered-set: paths already witnessed (CE obtained) in an earlier
  // round are NOT re-instrumented this round, so each escalation round
  // instruments a strictly smaller set and spends its solver budget only on
  // paths still lacking a CE. Sound: an instrumented assert is a property
  // obligation, not a path constraint, so omitting one removes an observation
  // and perturbs no other path; the cross-run cover is monotone (a real
  // witness stays valid). The denominator (all_claims) is built below WITHOUT
  // the skip, so skipping can never inflate the reported coverage.
  all_claims.clear();
  covered_set.clear();
  covered_set_outpath.clear();
  path_covered_ids.clear();
  path_covered_payload.clear();
  path_stable_id.clear();
  path_covered_outpath.clear();
  // Filename hardcoded relative to CWD, exactly like cov-report.json, and
  // deliberately never READ back in: a journal that a later run loaded would
  // accumulate, and "what this run witnessed" would stop being answerable from
  // it. It is truncated and rewritten from scratch by every run that asks for
  // the payload.
  path_ce_journal_path = emit_ce_journal ? "cov-ce-journal.json" : "";
  // Cleared per instrumentation, like every other static here: a stale reason
  // would stamp a perfectly complete run PARTIAL, which is the mirror of the
  // failure this whole mechanism exists to prevent and just as damaging.
  path_cov_partial_reason.clear();
  claims_in_solve_loop.clear();
  revert_paths.clear();
  rollback_revert_paths.clear();
  undetermined_exit_paths.clear();
  normal_exit_paths.clear();
  named_obstacle_paths.clear();
  truncation_weakened.clear();
  path_decision_depth.clear();
  degraded_call_sites.clear();
  {
    std::lock_guard lock(claim_outcome_mutex);
    claim_outcome.clear();
    path_ce.clear();
    path_ce_all.clear();
  }

  // Fingerprint of everything that can change what a path IS. The stable path
  // key below survives re-numbering; it cannot survive a change of identity, so
  // that is what this covers, and a mismatch throws the cache away rather than
  // trying to translate it.
  //
  // PATH_ID_SCHEMA_VERSION: bump when the shape of the stable key changes.
  // DECISION_SET_VERSION:   bump when the SET of things counted as a decision
  //                         changes (source ifs, short-circuit operands,
  //                         ternaries, the ABI non-payable gate, ...). Adding a
  //                         decision kind changes every path's identity even
  //                         though the source is untouched.
  {
    static constexpr int PATH_ID_SCHEMA_VERSION = 1;
    static constexpr int DECISION_SET_VERSION = 4;
    uint64_t h = fnv1a("path-cov-fingerprint");
    h = fnv1a("schema=" + std::to_string(PATH_ID_SCHEMA_VERSION), h);
    h = fnv1a("decisions=" + std::to_string(DECISION_SET_VERSION), h);
    h = fnv1a("sc_max=" + std::to_string(SC_DECISION_MAX), h);
    // Loop unwinding, call/recursion depth and external-call re-entry depth are
    // all bounded by this one number today; each is listed so that if they ever
    // separate, the omission is visible here rather than silent.
    h = fnv1a("loop_bound=" + std::to_string(path_cov_unwind), h);
    h = fnv1a("call_depth=" + std::to_string(path_cov_unwind), h);
    h = fnv1a("reentry_depth=" + std::to_string(path_cov_unwind), h);
    h = fnv1a("goal_cap=" + std::to_string(path_cov_max_goals), h);
    h = fnv1a("contract=" + scope_contract, h);
    // ---- --focus-function IS DELIBERATELY *NOT* IN THIS FINGERPRINT ----
    //
    // It narrows which units are instrumented, so a focused run writes a
    // covered set describing a SUBSET of the paths a whole-contract run
    // describes. The question the fingerprint answers is narrower than that:
    // "does an id written then still designate the same path now?" -- and for
    // this axis it does, by construction.
    //
    // A stable id is `hex64(fnv1a("exit:" + <exit location>, idh))` where `idh`
    // folds, per decision, the decision's SOURCE SITE, its operand index, its
    // polarity and its per-site occurrence count, seeded with
    // `fnv1a("unit:" + <unit id>)`. Every one of those inputs comes from the
    // unit's own body -- and the unit's own body does not depend on the focus,
    // because the EXPANSION loop is not narrowed (see the header on
    // `focus_function`; that is exactly why it is not). So the same path has
    // the same id under `--focus-function f`, under a different focus, and
    // under no focus at all. An old id either matches exactly or does not
    // match; it can never designate a DIFFERENT path, which is the failure this
    // guard exists to prevent.
    //
    // Nothing is lost on write-back either: write_path_covered_set_atomic seeds
    // its id set from the LOADED ids and only inserts, and carries the loaded
    // payloads forward before this run's overwrite them. A focused run
    // therefore UNIONS with the other units' entries rather than replacing
    // them.
    //
    // Adding the field would be actively harmful rather than merely
    // conservative: every focus would get its own fingerprint, so each
    // per-method run would DISCARD the accumulated file and re-solve from
    // scratch -- destroying the union the per-method sweep exists to build,
    // which is the documented workflow ("their paths are meant to be witnessed
    // by the run that focuses on them and unioned via the covered set").
    //
    // The residual is stated rather than hidden, and it is NOT introduced here:
    // the resulting file is an UNATTRIBUTABLE union -- a flat array of ids with
    // no record of which configuration witnessed each -- so a percentage
    // computed from it belongs to no single invocation. That was already true
    // before instrumentation was narrowed, because the focus was already
    // outside this fingerprint and already produced unionable files. Source
    // content, not mtime or path: the same bytes must produce the same
    // fingerprint on another machine.
    std::vector<std::string> srcs;
    srcs.push_back(filename);
    for (const auto &inc : config.ansi_c.include_files)
      srcs.push_back(inc);
    std::sort(srcs.begin(), srcs.end());
    for (const auto &s : srcs)
    {
      std::ifstream f(s, std::ios::binary);
      if (!f)
      {
        // Unreadable source => we cannot prove the cache still applies. Make
        // the fingerprint depend on that fact so it can never accidentally
        // match.
        h = fnv1a("unreadable:" + s, h);
        continue;
      }
      std::string body(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      h = fnv1a("src:" + get_filename_from_path(s) + ":" + body, h);
    }
    path_cov_fingerprint = hex64(h);
  }

  // Cross-run covered set: content-addressed ids, guarded fail-closed.
  if (!covered_set_path.empty())
  {
    path_covered_outpath = covered_set_path;
    std::ifstream in(covered_set_path);
    if (in)
    {
      try
      {
        nlohmann::json j;
        in >> j;
        const std::string fp = j.value("fingerprint", std::string());
        const int ver = j.value("version", 0);
        // FINGERPRINT FIRST, VERSION SECOND, and the order is deliberate: the
        // fingerprint is the stronger discriminator (it covers the source
        // bytes), and its message is what the fail-closed regression pins. A
        // file that fails both should be reported for the reason that makes the
        // entries meaningless, not for the reason that makes them incomplete.
        if (fp != path_cov_fingerprint)
        {
          // Fail closed. No migration: a file written under a different source,
          // decision set or bound describes paths that no longer exist, and any
          // attempt to carry entries over is exactly the code path in which a
          // wrong "already covered" would hide.
          log_status(
            "--coverage-covered-set: discarding {} — it was written for a "
            "different program/instrumentation (fingerprint {} != {}). "
            "Recomputing from scratch; no entries are carried over",
            covered_set_path,
            fp.empty() ? "<none>" : fp,
            path_cov_fingerprint);
        }
        else if (ver != PATH_COVERED_SET_VERSION)
        {
          // Also fail closed, and for a reason the fingerprint cannot see. A
          // version <= 2 file has the RIGHT ids for the RIGHT program and no
          // `payloads` object at all. Loading it would skip every listed path
          // as "already witnessed" and then report each one as an `F` whose
          // counterexample values are "in the report of the round that
          // witnessed it" — a round whose report may no longer exist. The
          // resulting F can never produce a test and nothing marks it, which is
          // strictly worse than re-solving the path.
          log_status(
            "--coverage-covered-set: discarding {} — it is schema version {} "
            "and this build reads version {}. Versions <= 2 persist path IDS "
            "with NO counterexample payload, so carrying them over would mark "
            "paths covered while permanently reporting them as payload-less. "
            "Recomputing from scratch; no entries are carried over",
            covered_set_path,
            ver,
            PATH_COVERED_SET_VERSION);
        }
        else
        {
          for (const auto &e : j.value("covered", nlohmann::json::array()))
            path_covered_ids.insert(e.get<std::string>());
          const nlohmann::json pl =
            j.value("payloads", nlohmann::json::object());
          for (auto it = pl.begin(); it != pl.end(); ++it)
            path_covered_payload[it.key()] = path_ce_from_json(it.value());
          // Printed even when the two agree, and ESPECIALLY when they do not: a
          // covered id with no payload is a path that will be skipped this
          // round and reported without inputs, which is the failure mode this
          // whole schema change exists to close. It must be visible on load,
          // not discovered later in the report.
          log_status(
            "--coverage-covered-set: loaded {} — {} witnessed path(s), {} of "
            "them carrying a CE payload",
            covered_set_path,
            path_covered_ids.size(),
            path_covered_payload.size());
        }
      }
      catch (const std::exception &ex)
      {
        log_warning(
          "coverage-covered-set: ignoring unparseable {} ({})",
          covered_set_path,
          ex.what());
        path_covered_ids.clear();
        path_covered_payload.clear();
      }
    }
  }

  // ---- Stage-2 certification query spec (--path-cov-certify) ----
  //
  // {"unit": "<fn name or full unit id>", "enc": N, "depth": D,
  //  "box": [{"name": "a", "lo": "0", "hi": "10", "holes": ["4"]}, ...]}
  //
  // `lo`/`hi`/`holes` are decimal STRINGS, not JSON numbers: Solidity inputs
  // are up to 256 bits and a JSON number would be silently truncated to a
  // double on the way in — a certified box quietly covering the wrong region is
  // the one outcome this query exists to prevent.
  //
  // `holes` is Definition 5's punched interval, and it is optional: absent
  // means the plain closed interval, byte for byte the query that was emitted
  // before it existed. See path_cov_certify_holes in the header for why a
  // closed interval alone makes the yield depend on an arbitrary solver choice.
  struct certify_boundt
  {
    std::string name, lo, hi;
    std::vector<std::string> holes;
  };
  // ---- Stage-2 outer-box batch spec (--path-cov-outer-box) ----
  struct outer_coordt
  {
    std::string name, lo, hi;
    // Explicit probe values, when the driver wants to choose them itself. The
    // first round on a 256-bit input cannot use a linear ladder — any span wide
    // enough to contain the boundary makes the resolution useless — so the
    // driver bootstraps with a GEOMETRIC ladder (0, 1, 2, 4, ... 2^k), gets a
    // bracket within a factor of two whatever the magnitude, and only then
    // switches to linear inside it. Which ladder to use is a policy decision
    // and therefore the driver's; the tool just measures the values it is
    // given.
    std::vector<std::string> values;
    // Also lay the uniform ladder over [lo, hi]. Set when the spec supplies
    // both forms: the explicit values keep an exactly-known point, the ladder
    // still measures the range, and the probe set is their union.
    bool subdivide = false;
  };
  bool outer_on = false;
  std::string outer_unit;
  size_t outer_probes = 8;
  std::vector<outer_coordt> outer_coords;
  // Coordinates PINNED to a value for this batch. The measured box is then a
  // statement about the SLICE through those values, not about the whole domain
  // — which is the point: when a guard ties two coordinates together
  // (`bal >= amt`) the domain is a diagonal, no box contains it tightly, and
  // the subtraction cannot separate anything. Pinning all but one coordinate
  // turns the problem back into one dimension, where the interval is exact.
  //
  // The pin is part of the answer and is printed with it. A region measured
  // under `bal == 0` that got rendered as `require(amt >= 1)` alone would be a
  // claim about inputs that were never examined.
  std::vector<std::pair<std::string, std::string>> outer_pins;
  path_cov_outer_box_mode = false;
  path_cov_outer_box_probes.clear();
  path_cov_outer_box_paths.clear();
  path_cov_outer_box_ce.clear();
  path_cov_outer_box_type_range.clear();
  path_cov_outer_box_pins.clear();
  path_cov_outer_box_obstacle.clear();
  path_cov_refused_coords.clear();
  if (!path_cov_outer_box_path.empty())
  {
    std::ifstream oin(path_cov_outer_box_path);
    if (!oin)
    {
      log_error(
        "--path-cov-outer-box: cannot open '{}'", path_cov_outer_box_path);
      abort();
    }
    try
    {
      nlohmann::json j;
      oin >> j;
      outer_unit = j.at("unit").get<std::string>();
      outer_probes = j.value("probes", (size_t)8);
      for (const auto &p : j.value("pin", nlohmann::json::array()))
        outer_pins.emplace_back(
          p.at("name").get<std::string>(), p.at("value").get<std::string>());
      for (const auto &c : j.at("coords"))
      {
        outer_coordt oc;
        oc.name = c.at("name").get<std::string>();
        // `values` and `lo`/`hi` are no longer exclusive, and the union is the
        // point. A ladder subdividing a span measures a bound only to its own
        // resolution, so a sibling whose real projection is a SINGLE POINT
        // comes back as an interval -- and the punched cut, which needs that
        // point exactly, then never fires. Measured end to end: level 0
        // resolved the sibling to `to == 255`, the refine round reported `[230,
        // 256]` for the same path, and the region fell back to a side cut.
        // Carrying the exact candidates alongside the ladder costs two probes
        // and keeps the point.
        if (c.contains("values"))
          for (const auto &v : c.at("values"))
            oc.values.push_back(v.get<std::string>());
        if (c.contains("lo") && c.contains("hi"))
        {
          oc.lo = c.at("lo").get<std::string>();
          oc.hi = c.at("hi").get<std::string>();
          oc.subdivide = true;
        }
        else if (oc.values.empty())
          // Neither form given: say so here rather than emitting a coordinate
          // with no probe at all, which reads downstream as "measured, and
          // nothing bounded it".
          throw std::runtime_error(
            "coordinate '" + oc.name +
            "' has neither \"values\" nor \"lo\"/\"hi\"");
        outer_coords.push_back(oc);
      }
      for (const auto &p : j.at("paths"))
      {
        const uint64_t e = p.at("enc").get<uint64_t>();
        path_cov_outer_box_paths.emplace_back(e, p.at("depth").get<uint64_t>());
        // Bound to a NAMED object first: `.items()` on the temporary returned
        // by `value()` iterates a destroyed object.
        const nlohmann::json ce_obj = p.value("ce", nlohmann::json::object());
        for (auto it = ce_obj.begin(); it != ce_obj.end(); ++it)
          path_cov_outer_box_ce[{e, it.key()}] = it.value().get<std::string>();
      }
    }
    catch (const std::exception &ex)
    {
      log_error(
        "--path-cov-outer-box: cannot parse '{}' ({}). Expected "
        "{{\"unit\":..., "
        "\"probes\":K, "
        "\"coords\":[{{\"name\":..,\"lo\":\"..\",\"hi\":\"..\"}}], "
        "\"paths\":[{{\"enc\":N,\"depth\":D,\"ce\":{{\"a\":\"..\"}}}}]}}",
        path_cov_outer_box_path,
        ex.what());
      abort();
    }
    outer_on = true;
    path_cov_outer_box_mode = true;
    path_cov_outer_box_pins = outer_pins;
    log_status(
      "--path-cov-outer-box: OUTER-BOX BATCH for unit '{}' — {} path(s), {} "
      "coordinate(s), {} probe(s) per direction. One fixed assumption per path "
      "(`tr == enc`), a whole ladder of candidate bounds as assertions, ONE "
      "run. This measures a box CONTAINING each path's domain; the certified "
      "region is then the difference against the siblings' boxes, at zero "
      "further queries. Resolution is (hi-lo)/(probes+1) per coordinate — a "
      "non-adaptive batch cannot give logarithmic precision; refine by running "
      "another batch on a narrower span",
      outer_unit,
      path_cov_outer_box_paths.size(),
      outer_coords.size(),
      outer_probes);
  }

  bool certify_on = false;
  // Did the certification query ever find its unit? See the check after the
  // enumeration loop -- a query matching NO unit emits no assume and no assert,
  // so everything holds vacuously and the run prints SUCCESSFUL.
  size_t certify_units_matched = 0;
  path_cov_certify_mode = false;
  path_cov_certify_box_names.clear();
  path_cov_certify_box.clear();
  path_cov_certify_ce.clear();
  path_cov_certify_holes.clear();
  path_cov_certify_nonvacuous_key = {};
  path_cov_certify_exit_keys.clear();
  std::string certify_unit;
  uint64_t certify_enc = 0, certify_depth = 0;
  std::vector<certify_boundt> certify_box;
  if (!path_cov_certify_path.empty())
  {
    std::ifstream cin_f(path_cov_certify_path);
    if (!cin_f)
    {
      log_error("--path-cov-certify: cannot open '{}'", path_cov_certify_path);
      abort();
    }
    try
    {
      nlohmann::json j;
      cin_f >> j;
      certify_unit = j.at("unit").get<std::string>();
      certify_enc = j.at("enc").get<uint64_t>();
      certify_depth = j.at("depth").get<uint64_t>();
      for (const auto &b : j.value("box", nlohmann::json::array()))
      {
        certify_boundt cb;
        cb.name = b.at("name").get<std::string>();
        cb.lo = b.at("lo").get<std::string>();
        cb.hi = b.at("hi").get<std::string>();
        for (const auto &h : b.value("holes", nlohmann::json::array()))
          cb.holes.push_back(h.get<std::string>());
        certify_box.push_back(cb);
      }
      const nlohmann::json cce = j.value("ce", nlohmann::json::object());
      for (auto it = cce.begin(); it != cce.end(); ++it)
        path_cov_certify_ce[it.key()] = it.value().get<std::string>();
    }
    catch (const std::exception &ex)
    {
      // Deliberately fatal rather than "ignore and fall back to enumeration":
      // a malformed spec that silently produced an ordinary coverage run would
      // print a full, plausible report that answers a different question.
      log_error(
        "--path-cov-certify: cannot parse '{}' ({}). Expected "
        "{{\"unit\":..., \"enc\":N, \"depth\":D, \"box\":[{{\"name\":..., "
        "\"lo\":\"..\", \"hi\":\"..\"}}]}}",
        path_cov_certify_path,
        ex.what());
      abort();
    }
    certify_on = true;
    path_cov_certify_mode = true;
    size_t certify_holes_total = 0;
    for (const auto &b : certify_box)
    {
      path_cov_certify_box_names.push_back(b.name);
      path_cov_certify_box.push_back({b.name, b.lo, b.hi});
      if (!b.holes.empty())
      {
        path_cov_certify_holes[b.name] = b.holes;
        certify_holes_total += b.holes.size();
      }
    }
    if (certify_holes_total > 0)
      log_status(
        "--path-cov-certify: the box is a PUNCHED interval (Definition 5) — {} "
        "value(s) are removed across {} coordinate(s), so the assumption is "
        "`lo <= c <= hi && c != h ...`. A hole says the region omits exactly "
        "those points, which a closed interval cannot say: without it the "
        "subtraction has to keep whichever SIDE of an excluded value happens "
        "to "
        "hold its own counterexample, and that side is chosen by the solver, "
        "not "
        "by the method",
        certify_holes_total,
        path_cov_certify_holes.size());
    log_status(
      "--path-cov-certify: CERTIFICATION QUERY for unit '{}' path enc={} "
      "depth={} over {} bounded input(s). The per-path identity asserts are "
      "NOT "
      "emitted in this mode and NO [Coverage] block is printed — a certified "
      "box makes these claims HOLD, which the coverage counters would report "
      "as "
      "uncovered. THE RESULT OF THIS RUN IS THE `RESULT:` LINE, not the "
      "VERIFICATION SUCCESSFUL / FAILED verdict: a non-vacuity witness is "
      "emitted at this path's own exit and is REFUTED on every run that "
      "certifies, so a CERTIFIED box prints VERIFICATION FAILED. RESULT is one "
      "of CERTIFIED (every input in the box walks this path), REFUTED (the "
      "counterexample is an input inside the box that leaves it), VACUOUS "
      "(the box admits no execution that walks this path at all, so the run "
      "establishes nothing), UNDECIDED (nothing was refuted but the solver "
      "could not answer) or UNDECIDED-TRUNCATED (a loop was cut at the unwind "
      "bound while unwinding assertions were disabled, so the executions that "
      "would have witnessed this path may simply have been assumed away -- "
      "VACUOUS is then not a statement this run is entitled to make). THIS "
      "LIST IS THE COMPLETE SET and a driver must branch on all of it: an "
      "unrecognised token is a tool that knows something the driver does not, "
      "and reading it as \"no verdict line\" silently falls back to the "
      "VERIFICATION SUCCESSFUL / FAILED line this very message says not to use",
      certify_unit,
      certify_enc,
      certify_depth,
      certify_box.size());
  }

  // ---- STAGE 3: post-state assertion synthesis spec (--path-cov-assert) ----
  //
  // {"unit": ..., "enc": N, "depth": D,
  //  "region": [{"name","lo","hi","holes"?}, ...],
  //  "vars":   [{"name","abs_lo"?,"abs_hi"?,
  //              "delta_dir"?,"delta_lo"?,"delta_hi"?}, ...]}
  //
  // `region` is byte for byte the shape certify parses under "box" and goes
  // through the SAME parser. `vars` is optional; omitting it emits the equality
  // and sign rungs for every eligible state variable, which is the default.
  struct assert_vart
  {
    std::string name;
    bool has_abs = false;
    std::string abs_lo, abs_hi;
    bool has_delta = false;
    std::string delta_dir, delta_lo, delta_hi;
  };
  bool assert_on = false;
  // Route 5, mirrored from certify: a spec matching NO unit emits no assume and
  // no assert, so nothing is checked and the run prints VERIFICATION SUCCESSFUL
  // with exit 0.
  size_t assert_units_matched = 0;
  std::string assert_unit;
  uint64_t assert_enc = 0, assert_depth = 0;
  std::vector<path_cov_boundt> assert_region;
  std::vector<assert_vart> assert_vars;
  // "vars was written down" is NOT "vars named something". Two entry conditions
  // into one symptom (an empty ladder); they need two messages.
  bool assert_vars_present = false;
  path_cov_assert_mode = false;
  path_cov_assert_candidates.clear();
  if (!path_cov_assert_path.empty())
  {
    std::ifstream ain(path_cov_assert_path);
    if (!ain)
    {
      log_error("--path-cov-assert: cannot open '{}'", path_cov_assert_path);
      abort();
    }
    try
    {
      nlohmann::json j;
      ain >> j;
      assert_unit = j.at("unit").get<std::string>();
      assert_enc = j.at("enc").get<uint64_t>();
      assert_depth = j.at("depth").get<uint64_t>();
      parse_bounds(j, "region", assert_region);
      assert_vars_present = j.contains("vars");
      for (const auto &v : j.value("vars", nlohmann::json::array()))
      {
        assert_vart av;
        av.name = v.at("name").get<std::string>();
        if (v.contains("abs_lo") || v.contains("abs_hi"))
        {
          // Half an interval is not an interval. Refused rather than completed
          // with a type bound, because the spec would then be answered about
          // something it did not say.
          av.abs_lo = v.at("abs_lo").get<std::string>();
          av.abs_hi = v.at("abs_hi").get<std::string>();
          av.has_abs = true;
        }
        if (
          v.contains("delta_lo") || v.contains("delta_hi") ||
          v.contains("delta_dir"))
        {
          // A DIRECTION IS MANDATORY, and defaulting it is a false-certificate
          // route. Candidate variables are unsigned, so `post - pre` WRAPS on a
          // decrease: a spec meaning "decreases by 1..10" defaulted to `inc`
          // would be answered about the wrapped difference.
          av.delta_dir = v.at("delta_dir").get<std::string>();
          av.delta_lo = v.at("delta_lo").get<std::string>();
          av.delta_hi = v.at("delta_hi").get<std::string>();
          if (av.delta_dir != "inc" && av.delta_dir != "dec")
            throw std::runtime_error(
              "variable '" + av.name +
              "': delta_dir must be \"inc\" or \"dec\"");
          av.has_delta = true;
        }
        assert_vars.push_back(av);
      }
    }
    catch (const std::exception &ex)
    {
      // Fatal rather than "ignore and fall back to enumeration": a malformed
      // spec that silently produced an ordinary coverage run would print a
      // full, plausible report answering a different question.
      log_error(
        "--path-cov-assert: cannot parse '{}' ({}). Expected "
        "{{\"unit\":..., \"enc\":N, \"depth\":D, "
        "\"region\":[{{\"name\":...,\"lo\":\"..\",\"hi\":\"..\"}}], "
        "\"vars\":[{{\"name\":...,\"abs_lo\":\"..\",\"abs_hi\":\"..\","
        "\"delta_dir\":\"inc|dec\",\"delta_lo\":\"..\",\"delta_hi\":\"..\"}}]}"
        "}",
        path_cov_assert_path,
        ex.what());
      abort();
    }
    // ---- N1, entry condition (a): `vars` is present and names NOTHING ----
    //
    // Zero candidates emitted, nothing checked, VERIFICATION SUCCESSFUL with
    // exit 0 -- indistinguishable from a ladder that passed. Refused here
    // because it is knowable here; entry condition (b) has its own gate after
    // the unit loop and its own message. Closing one would produce a run that
    // looks exactly like a fix.
    if (assert_vars_present && assert_vars.empty())
    {
      log_error(
        "--path-cov-assert: the spec contains an EMPTY \"vars\" array, so not "
        "one candidate assertion would be emitted and the run would print "
        "VERIFICATION SUCCESSFUL for a ladder it never built. Omit \"vars\" "
        "entirely to get the default ladder over every eligible state "
        "variable, or name the variables to assert about");
      exit(1);
    }
    assert_on = true;
    path_cov_assert_mode = true;
    log_status(
      "--path-cov-assert: POST-STATE ASSERTION LADDER for unit '{}' path "
      "enc={} depth={} over {} region bound(s) and {} explicit variable "
      "spec(s). The region is ASSUMED at entry -- it is exactly the `require` "
      "a "
      "generated test would carry -- and each candidate is asserted at THIS "
      "path's own exit under `tr != enc || cnt != depth`, so it is vacuous on "
      "every other path. One fixed assumption, a whole ladder of assertions, "
      "ONE run. NO [Coverage] block is printed and the run's VERIFICATION "
      "SUCCESSFUL/FAILED line is NOT the result: a REFUTED candidate is the "
      "ladder working. The result is the per-candidate table printed after "
      "solving",
      assert_unit,
      assert_enc,
      assert_depth,
      assert_region.size(),
      assert_vars.size());
  }

  // Keep the counterexample payload alive through slicing, WITHOUT switching
  // slicing off. The symex slicer works backwards from the claim, and a path
  // claim's guard is `tr != enc || cnt != depth` — it mentions nothing but the
  // ghost accumulators. So every contract-state write and every environment
  // read is unreachable from the claim and gets sliced, leaving the report's
  // `inputs`/`env`/`final_state` empty. (Call arguments survive on their own:
  // the decisions that build `tr` are guards over them.)
  //
  // ESBMC already has a per-symbol exemption for exactly this — no_slice_names
  // (`--no-slice-name`), consulted by symex_slicet::get_symbols. Register the
  // three symbol families the harvest in bmc.cpp reads, and nothing else:
  //   (a) the contract instance object — a Solidity `this->x = v` is an update
  //       of THIS symbol, so without it every final_state is empty;
  //   (b) contract-scope stores with no `@F@` in the id — mappings and dynamic
  //       arrays are NOT fields of the contract object, the frontend lowers
  //       them to contract-level globals; this is the same shape the harvest
  //       keys on;
  //   (c) the EVM environment, by the same msg_/tx_/block_ base-name test the
  //       harvest uses to classify a value as `env` rather than an argument.
  // Everything else — the c2goto keccak/sha256/ABI tables, the address
  // allocator, the dispatcher plumbing — is still sliced away.
  if (protect_ce_symbols)
  {
    size_t n_obj = 0, n_store = 0, n_env = 0;
    cov_context->foreach_operand([&](const symbolt &s) {
      const std::string id = s.id.as_string();
      const std::string base = s.name.as_string();
      if (id.rfind("sol:@_ESBMC_Object_", 0) == 0)
      {
        config.no_slice_names.insert(id);
        ++n_obj;
      }
      else if (
        id.rfind("sol:@C@", 0) == 0 && id.find("@F@") == std::string::npos)
      {
        config.no_slice_names.insert(id);
        ++n_store;
      }
      else if (
        base.rfind("msg_", 0) == 0 || base.rfind("tx_", 0) == 0 ||
        base.rfind("block_", 0) == 0)
      {
        config.no_slice_names.insert(id);
        ++n_env;
      }
    });
    log_status(
      "--solidity-path-coverage with --cov-report-json: exempting {} symbol(s) "
      "from slicing so each path's counterexample values survive into the "
      "report ({} contract object(s), {} contract-scope store(s), {} "
      "environment); slicing stays enabled for everything else",
      n_obj + n_store + n_env,
      n_obj,
      n_store,
      n_env);
  }

  // A function is a UNIT iff it is a public/external entry. The frontend
  // creates `<function-id>#_sol_save_this` for exactly those (public /
  // external / receive / fallback) and for nothing else, so the symbol's
  // existence is the test — no new frontend signal is needed.
  auto is_external_entry = [&](const irep_idt &fid) {
    return ns.lookup(irep_idt(fid.as_string() + "#_sol_save_this")) != nullptr;
  };

  // Is any of this body's instructions in the user source? c2goto library
  // models and the synthetic harness carry non-user locations.
  auto body_in_user_src = [&](const goto_programt &p) {
    forall_goto_program_instructions (uit, p)
      if (
        location_pool.count(
          get_filename_from_path(uit->location.file().as_string())))
        return true;
    return false;
  };

  // Is this instruction an INTERNAL call that must be expanded into the
  // caller's path identity?
  //
  // Everything a Solidity contract calls directly by symbol is internal in the
  // goto model. Measured on a `this.f(a)` self-call: the frontend lowers it to
  // the very same direct FUNCTION_CALL as a plain `f(a)` and models no
  // success/failure edge for it, so at this layer an external SELF-call is
  // indistinguishable from an internal one — and there is nothing to branch on
  // even if it were distinguished. A genuine external call (`addr.call{...}`,
  // an interface method on another address) does not appear as a direct call
  // to a user function at all: it goes through the `_ESBMC_Nondet_Extcall_*`
  // model, which this predicate excludes, so it is left unexpanded and its
  // success/failure stays a decision, as the methodology requires.
  auto expandable_callee =
    [&](goto_programt::const_targett i) -> const goto_functiont * {
    if (!i->is_function_call() || !is_code_function_call2t(i->code))
      return nullptr;
    const expr2tc &callee = to_code_function_call2t(i->code).function;
    if (!is_symbol2t(callee))
      return nullptr;
    const irep_idt cid = to_symbol2t(callee).thename;
    const std::string cids = cid.as_string();
    // Harness plumbing: the dispatcher and the external-call model are the
    // boundary of the unit, never part of it.
    if (
      cids.find("_ESBMC_Main") != std::string::npos ||
      cids.find("_ESBMC_Nondet_Extcall") != std::string::npos)
      return nullptr;
    // A lowered `revert E()` is a STATEMENT, not a call: its whole body is
    // ASSUME(false) and the enumerator detects it by name to place the path's
    // assert upstream of that assume. Expanding it would destroy that marker.
    const symbolt *csym = ns.lookup(cid);
    if (
      csym != nullptr &&
      !csym->get_type().get("#sol_error").as_string().empty())
      return nullptr;
    auto m_it = goto_functions.function_map.find(cid);
    if (m_it == goto_functions.function_map.end())
      return nullptr;
    if (!m_it->second.body_available || m_it->second.body.hide)
      return nullptr;
    if (!body_in_user_src(m_it->second.body))
      return nullptr;
    // ---- --focus-function MUST NOT REACH THIS PREDICATE ----
    //
    // A unit body has a DOUBLE IDENTITY: (a) an externally-callable entry with
    // its own ABI value gate, and (b) a physically inlined copy inside another
    // unit's path when it is called internally. --focus-function suppresses (a)
    // for non-focused units -- that is the narrowing, and it is done in the
    // ENUMERATION loop. Suppressing (b) as well, by refusing to expand a call
    // whose callee is a non-focused unit, is a different change and a wrong
    // one: the focused unit then loses the callee's decisions from its own path
    // identity and every `enc` silently means something else.
    //
    // MEASURED as a must-flip pair on
    // regression/esbmc-solidity/solidity_path_cov_focus_function_keeps_callee_-
    // decisions (`--focus-function caller`, where `caller` internally calls the
    // PUBLIC `pub` and the private `helper`):
    //
    //     correct            expanded 2 calls, 5 paths, 5.00x, enc
    //     15/14/13/12/2 (b) suppressed     expanded 1 call,  3 paths, 3.00x,
    //     enc 7/6/2
    //
    // The broken side looks BETTER -- fewer paths, 100% coverage instead of
    // 80%, a faster run -- and it is silent: refusing the callee here also
    // hides it from the residual-unit-call scan below, which is the detector
    // that exists to name an unexpanded call to a gated unit, so not even the
    // NAMED OBSTACLE warning fires. Nothing marks it. Do not add a focus test
    // here.
    return &m_it->second;
  };

  // Expand internal calls into every unit, so a callee's decisions become part
  // of its caller's path identity. Bounded by path_cov_unwind passes: one pass
  // expands exactly the calls present when it started, so pass count == call
  // depth. Using the LOOP bound here is not a convenience — symex bounds
  // recursion with the same --unwind, so any other value would enumerate paths
  // the solver cannot reach (or miss ones it can).
  optionst inline_opts;
  sol_path_inlinet inliner(goto_functions, inline_opts, ns);
  size_t inlined_calls = 0, residual_calls = 0;
  std::set<std::string> residual_fns;
  std::set<std::string> residual_unit_fns;
  // Per unit: which UNIT callees it still calls unexpanded. This is not a
  // reporting convenience — it is what the containment below is keyed on, so
  // the marking reaches the right unit's paths instead of being a global count
  // nobody acts on.
  std::map<std::string, std::set<std::string>> residual_unit_callees_of;
  // Measurement (§ "path distribution"): a snapshot of every unit's body BEFORE
  // expansion, plus how many calls were expanded into it. Taken here because
  // this is the last moment the pre-expansion shape exists.
  std::map<std::string, goto_programt> pre_inline_body;
  std::map<std::string, size_t> expanded_into_unit;
  size_t degraded_units = 0, withdrawn_sites_total = 0;
  size_t degradation_exhausted_units = 0;

  // Identity of a CALL POINT for degradation: the callee plus the source
  // location of the call expression. Deliberately not the goto instruction —
  // after expansion the same source-level call exists as several physical
  // copies (one per copy of the enclosing callee), and "withdraw this call"
  // has to withdraw all of them or the unit's paths still multiply through the
  // copies that were left behind. A synthesised call with no source location
  // therefore folds together with its siblings, which errs towards withdrawing
  // more: the safe direction, since withdrawing costs assertion strength while
  // failing to withdraw costs the budget.
  auto callee_id_of = [](goto_programt::const_targett i) {
    const expr2tc &callee = to_code_function_call2t(i->code).function;
    return to_symbol2t(callee).thename.as_string();
  };
  auto call_point_key = [&callee_id_of](goto_programt::const_targett i) {
    return callee_id_of(i) + " at " + i->location.as_string();
  };

  // May this call point be withdrawn?
  //
  // NOT if the callee is itself a unit, and the reason is soundness rather than
  // taste. Withdrawing means leaving a direct call to the callee's own body —
  // and that body carries the synthesised ABI non-payable gate, which models an
  // EXTERNAL entry. An internal call never runs that gate on-chain, so the
  // model would admit "the callee reverted because the transaction carried
  // value" in the middle of a caller that on-chain proceeds normally: a
  // counterexample describing an execution that does not exist, i.e. a test
  // that is red on the unmodified contract. Physical expansion is exactly what
  // fixes that (the caller's copy is gate-free), so undoing it for a unit
  // callee re-opens the hole. Internal/private helpers have no gate and are
  // safe to withdraw.
  auto withdrawable = [&](goto_programt::const_targett i) {
    return !is_external_entry(irep_idt(callee_id_of(i)));
  };

  // Expand every expandable call into `b` EXCEPT those whose call-point key is
  // in `withdrawn`. One pass expands exactly the calls that were present when
  // it started, so the pass index is the call's depth from the unit entry; that
  // is both what bounds recursion and what `sites` records alongside each key.
  // Only withdrawable call points are offered as candidates in `sites`.
  auto expand_into = [&](
                       goto_programt &b,
                       const std::set<std::string> &withdrawn,
                       std::map<std::string, size_t> *sites) -> size_t {
    size_t n = 0;
    for (size_t round = 0; round < path_cov_unwind; ++round)
    {
      bool changed = false;
      for (auto it = b.instructions.begin(); it != b.instructions.end();)
      {
        const goto_functiont *callee = expandable_callee(it);
        if (callee == nullptr)
        {
          ++it;
          continue;
        }
        if (withdrawn.count(call_point_key(it)) != 0)
        {
          // Left as a plain call: this IS the degradation. The DFS walks over a
          // call as a straight-line instruction, so the callee contributes no
          // decisions to this unit's path identity — while symex still executes
          // it, so the model is unchanged and only the recording is coarser.
          ++it;
          continue;
        }
        if (sites != nullptr && withdrawable(it))
          sites->emplace(call_point_key(it), round);
        inliner.expand_here(b, it, *callee);
        ++n;
        changed = true;
      }
      if (!changed)
        break;
      remove_no_op(b);
      b.update();
    }
    return n;
  };

  // Complete-path count of `b` under the same rules the DFS uses, capped so a
  // runaway unit costs bounded work. Anything past the cap comes back as
  // cap + 1, which is all a budget comparison needs.
  auto count_paths_of = [&](const goto_programt &b, size_t cap) -> size_t {
    bool hit = false;
    const size_t n =
      count_paths_no_instrument(b, ns, path_cov_unwind, cap, hit);
    return hit ? cap + 1 : n;
  };

  // The budget a unit's enumeration has to fit into. One goal is held back for
  // the ABI non-payable gate, which is synthesised AFTER this point and adds
  // exactly one path: without the reservation a unit could be brought to
  // exactly the cap here and then handed straight to the truncation backstop by
  // the gate, which is precisely the ordering this is meant to prevent.
  const size_t unit_budget =
    path_cov_max_goals > 1 ? path_cov_max_goals - 1 : 1;

  // Why each unit ended up where it did. Recorded rather than re-derived,
  // because the truncation backstop's report has to distinguish three different
  // situations that all look identical at the cap: a policy that was not
  // aggressive enough, a unit that had nothing left to give up, and the
  // estimator disagreeing with the enumeration. Only the last is a defect.
  enum class budget_statet
  {
    fits,          // the pre-enumeration count was already inside the budget
    no_candidates, // over budget, with no withdrawable call point at all
    degraded_fits, // withdrawals brought it inside the budget
    degraded_over  // everything withdrawable is gone and it is still over
  };
  std::map<std::string, budget_statet> budget_state;
  // The path count the budget decision was actually taken on, per unit. Kept so
  // that when the cap fires anyway the report can print the pre-enumeration
  // number NEXT TO the enumerated one: the two are produced by different code
  // (a flat counter vs the enumerating DFS) and a gap between them is a
  // COUNTING-UNIT mismatch to be reconciled, not automatically a bug. Printing
  // one number and calling it a failure would send someone hunting for a defect
  // that may not exist.
  std::map<std::string, size_t> estimated_paths;

  Forall_goto_functions (e_it, goto_functions)
  {
    if (!e_it->second.body_available || e_it->first == "__ESBMC_main")
      continue;
    goto_programt &b = e_it->second.body;
    if (filter(e_it->first, b))
      continue;
    if (!is_external_entry(e_it->first))
      continue;
    if (
      !scope_contract.empty() &&
      contract_of(e_it->first.as_string()) != scope_contract)
      continue;
    if (!body_in_user_src(b))
      continue;

    const std::string uname = e_it->first.as_string();
    pre_inline_body[uname].copy_from(b);

    // Expand everything first and measure. Almost every unit fits, and for
    // those this is the only expansion performed and `b` is already the body we
    // want — the degradation machinery below costs nothing.
    std::map<std::string, size_t> sites;
    size_t expanded_here = expand_into(b, {}, &sites);
    std::set<std::string> withdrawn;
    std::vector<std::string> withdrawn_order;

    const size_t full_paths = count_paths_of(b, unit_budget);
    estimated_paths[uname] = full_paths;
    if (full_paths <= unit_budget)
      budget_state[uname] = budget_statet::fits;
    else if (sites.empty())
    {
      // Over budget with nothing to give up: either the unit has no internal
      // calls at all, or every call it has is to another UNIT and withdrawing
      // those would re-open the ABI-gate hole (see `withdrawable`). Say so
      // here, before the enumeration runs, so the truncation that follows reads
      // as the declared consequence of a budget rather than as a surprise.
      budget_state[uname] = budget_statet::no_candidates;
      log_warning(
        "--solidity-path-coverage: unit '{}' is over the per-unit budget ({}) "
        "and degradation has NOTHING it may withdraw — the unit's own source "
        "decisions already exceed the budget, or its only internal calls are "
        "to "
        "public/external functions, which must stay expanded (their bodies "
        "carry the ABI value gate, which an internal call does not run "
        "on-chain). The goal cap will therefore truncate this unit; raise "
        "--path-cov-max-goals to enumerate it",
        uname,
        path_cov_max_goals);
    }
    else
    {
      // ---- DEGRADATION (fires BEFORE the truncation backstop) ----
      //
      // Measured motivation: on a real benchmark one contract enumerated 120166
      // paths across 39 units, 12 of which were over the cap on their own. At
      // that size the choice is not "enumerate or not" but "give up resolution
      // deliberately, or have it taken away by a cap". Those are not the same:
      // the cap DROPS paths that exist in the model, so nothing downstream can
      // subtract their inputs from a surviving path's certified region;
      // withdrawing a call point instead makes the surviving classes coarser
      // while they still cover the whole input space.
      //
      // Which call points: measured, not estimated. Each candidate is expanded
      // once with itself withheld and the resulting path count recorded, so
      // "cuts the most paths" is a number rather than a guess. Ties break
      // towards the call FURTHEST from the entry, where the decisions are least
      // directly controlled by the unit's own arguments and the assertion
      // strength given up is worth least.
      //
      // The ranking is measured once and then applied greedily. Re-ranking
      // after every withdrawal would be quadratic for no soundness benefit:
      // soundness does not depend on the choice at all, only the strength of
      // what survives does, and the result is reported per unit either way.
      std::vector<std::tuple<size_t, size_t, std::string>> ranked;
      ranked.reserve(sites.size());
      for (const auto &[key, depth] : sites)
      {
        std::set<std::string> trial{key};
        goto_programt probe;
        probe.copy_from(pre_inline_body[uname]);
        expand_into(probe, trial, nullptr);
        // Sorting ascending on (paths left, SIZE_MAX - depth) puts the biggest
        // cut first and, among equal cuts, the deepest call first.
        ranked.emplace_back(
          count_paths_of(probe, unit_budget), SIZE_MAX - depth, key);
      }
      std::sort(ranked.begin(), ranked.end());

      bool fits = false;
      for (const auto &r : ranked)
      {
        withdrawn.insert(std::get<2>(r));
        withdrawn_order.push_back(std::get<2>(r));
        b = pre_inline_body[uname];
        expanded_here = expand_into(b, withdrawn, nullptr);
        estimated_paths[uname] = count_paths_of(b, unit_budget);
        if (estimated_paths[uname] <= unit_budget)
        {
          fits = true;
          break;
        }
      }

      budget_state[uname] =
        fits ? budget_statet::degraded_fits : budget_statet::degraded_over;
      ++degraded_units;
      withdrawn_sites_total += withdrawn_order.size();
      degraded_call_sites[uname] = withdrawn_order;
      std::string names;
      for (const auto &w : withdrawn_order)
        names += (names.empty() ? "" : "; ") + w;
      log_warning(
        "--solidity-path-coverage: DEGRADED unit '{}' — fully expanded it "
        "enumerates more paths than the per-unit budget ({}), so {} call "
        "point(s) were WITHDRAWN from its path identity and are now treated as "
        "black boxes: {}. The callees still EXECUTE (the call is still there), "
        "they just stop contributing decisions, so the path classes get "
        "coarser "
        "while still partitioning the input space — sound, with weaker "
        "assertions, and weaker exactly at the call points named here. This is "
        "tried BEFORE the goal cap on purpose: the cap would instead DROP "
        "paths "
        "that exist in the model",
        uname,
        path_cov_max_goals,
        withdrawn_order.size(),
        names);
      if (!fits)
      {
        ++degradation_exhausted_units;
        log_warning(
          "--solidity-path-coverage: unit '{}' is STILL over the budget ({}) "
          "with every one of its {} call point(s) withdrawn — its own source "
          "decisions alone exceed it. Degradation has nothing left to give up, "
          "so the goal cap will act as the backstop and that unit's truncation "
          "is expected rather than a policy failure",
          uname,
          path_cov_max_goals,
          withdrawn_order.size());
      }
    }

    // Anything still callable after the last pass is recursion deeper than the
    // bound. Its decisions are NOT in this unit's path identity, so the paths
    // through it are merged rather than enumerated. That is a silent loss of
    // resolution unless it is reported, so report it. Call points withdrawn by
    // degradation are excluded here: they are also unexpanded, but deliberately
    // and with their own report, and merging the two would make a budget
    // decision read as a depth-bound overflow.
    forall_goto_program_instructions (it, b)
    {
      const goto_functiont *callee = expandable_callee(it);
      if (callee == nullptr)
        continue;
      if (withdrawn.count(call_point_key(it)) != 0)
        continue;
      ++residual_calls;
      const std::string cid = callee_id_of(it);
      residual_fns.insert(cid);
      // A residual call to a UNIT is not a loss of resolution — it is the SAME
      // hole that degradation refuses to open, arrived at from the other side.
      //
      // A unit's body is DOUBLE-IDENTITY: the copy expanded into a caller
      // models an internal call (no ABI value gate), while the unit's own body
      // models an external entry (gated). That is precisely what physical
      // expansion buys. Leaving a call to the unit's own body unexpanded makes
      // an INTERNAL call use the EXTERNAL-entry body, so the model admits "the
      // callee reverted because the transaction carried value" inside a caller
      // that on-chain proceeds. That execution does not exist on chain, and a
      // test built from a counterexample containing it is RED on the unmodified
      // contract.
      //
      // So this is contained, not merely named. Naming it while letting the
      // paths through to a downstream emitter would leave exactly the failure
      // the whole obstacle mechanism exists to prevent.
      if (is_external_entry(irep_idt(cid)))
      {
        residual_unit_fns.insert(cid);
        residual_unit_callees_of[uname].insert(cid);
      }
    }
    inlined_calls += expanded_here;
    expanded_into_unit[uname] = expanded_here;
  }
  if (degraded_units > 0)
    log_status(
      "--solidity-path-coverage: degradation summary — {} unit(s) had {} call "
      "point(s) withdrawn to fit the per-unit budget ({}); {} of those unit(s) "
      "could not be made to fit even with every call point withdrawn. "
      "Degradation and truncation are separate mechanisms with separate "
      "reports: this one costs assertion strength at named places and keeps "
      "the enumeration complete, the goal cap instead drops paths",
      degraded_units,
      withdrawn_sites_total,
      path_cov_max_goals,
      degradation_exhausted_units);
  if (inlined_calls > 0)
    log_status(
      "--solidity-path-coverage: expanded {} internal call(s) into their "
      "calling unit (call depth bound = {}), so a callee's decisions are part "
      "of its caller's path identity",
      inlined_calls,
      path_cov_unwind);
  if (residual_calls > 0)
  {
    std::string names;
    for (const auto &n : residual_fns)
      names += (names.empty() ? "" : ", ") + n;
    log_warning(
      "--solidity-path-coverage: {} call site(s) are deeper than the call "
      "depth bound ({}) and were NOT expanded ({}); paths through them are "
      "MERGED rather than enumerated. Raise --unwind to enumerate them",
      residual_calls,
      path_cov_unwind,
      names);
    if (!residual_unit_fns.empty())
    {
      std::string unames;
      for (const auto &n : residual_unit_fns)
        unames += (unames.empty() ? "" : ", ") + n;
      log_warning(
        "--solidity-path-coverage: {} of those unexpanded callee(s) are "
        "themselves public/external UNITS ({}). That is not only coarser: "
        "their "
        "bodies carry the synthesised ABI value gate, which models an EXTERNAL "
        "entry, while the call reaching them here is INTERNAL and never runs "
        "that gate on-chain. So the model can admit an execution in which the "
        "callee reverts for carrying value inside a caller that on-chain "
        "proceeds. Every path of every unit containing such a call is "
        "therefore "
        "a NAMED OBSTACLE (same containment as a branch-free assume — same "
        "failure, different route), not merely a coarser one. Raise --unwind "
        "so "
        "these are expanded: an expanded copy is gate-free, which is exactly "
        "what makes both entry kinds correct at once",
        residual_unit_fns.size(),
        unames);
    }
  }

  size_t ghost_counter = 0;
  // Folded short-circuit sites left out of the decision set for exceeding
  // SC_DECISION_MAX operands; reported so the incompleteness is visible.
  size_t sc_sites_over_cap = 0;
  size_t total_paths = 0;
  size_t dropped_paths = 0;
  size_t skipped_paths = 0;
  // In-scope user functions that are NOT units (internal/private helpers).
  size_t non_unit_functions = 0;
  size_t units_enumerated = 0;
  // --focus-function: units that ARE units and ARE in scope but are not the
  // focused one, so this run enumerates and instruments nothing for them.
  size_t units_skipped_by_focus = 0;
  // Every unit id the loop below considered, focused or not. Kept ONLY so the
  // no-match failure can print what was actually available. "focus 'x' matched
  // nothing" sends a reader to check the spelling; the candidate list lets them
  // see instead whether the name is present under a different contract, or
  // absent entirely, without having to guess which.
  std::vector<std::string> focus_candidates;
  // Units disqualified wholesale: the model and the EVM disagree there, so no
  // path of the unit may become a test. TWO causes reach this, counted apart
  // because they are different defects needing the same containment — both are
  // "the model admits an execution that does not exist on chain":
  //   (a) a source decision lowered to a control-flow-free assume;
  //   (b) an unexpanded call to a UNIT, which routes an internal call through
  //       the external-entry body and its ABI value gate.
  // Truncation used to be counted here too and no longer is — it weakens
  // assertions without making them wrong (see `truncation_weakened`).
  size_t obstacle_units = 0;
  size_t obstacle_paths_assume = 0;
  size_t obstacle_units_residual = 0;
  size_t obstacle_paths_residual = 0;
  // Paths whose certified region is narrowed by their unit hitting the cap.
  size_t truncation_weakened_paths = 0;
  // Path-count distribution (see the measurement note on
  // count_paths_no_instrument). `pre_expansion_total` sums each unit's paths as
  // its own body had them before internal calls were expanded, so the ratio
  // against the real total is the measured cost of expansion. `units_at_cap`
  // must be reported: a nonzero value means the TAIL of the distribution is
  // truncated and the distribution must not be presented as complete.
  size_t pre_expansion_total = 0;
  size_t units_at_cap = 0;
  size_t max_unit_paths = 0;
  std::string max_unit_name;

  // ---- Resolve a box / ladder coordinate name to what it denotes ----
  //
  // Three kinds, told apart by an EXPLICIT prefix rather than by guessing:
  //   msg.* / tx.* / block.*   the EVM environment
  //   state.<field>            a state variable AT FUNCTION ENTRY
  //   anything else            a parameter of this unit
  //
  // `state.` is the one that matters. Real path conditions are mostly guarded
  // by storage, and a box that can only bound parameters cannot say anything
  // about those paths: the region it certifies is a statement about the
  // parameter axes only, while the path taken still depends on state the box
  // never constrained. The visible symptom is a region that fails certification
  // with a counterexample that keeps moving — the escaping input differs in a
  // coordinate the box does not mention, so shrinking on the coordinates it
  // does mention never converges.
  //
  // SCOPE: `state.<field>` resolves against the contract INSTANCE OBJECT's
  // struct components, so it covers scalar state variables. A mapping or
  // dynamic array is NOT a field of that object — the frontend lowers those to
  // contract-scope globals (`sol:@C@<C>@<name>`) — so `state.balances` does not
  // resolve. A FOURTH shape, `state.<m>[<key>]`, names ONE SLOT of such a store
  // and IS resolved; see the branch below for what that buys and what it still
  // cannot say.
  //
  // Guessing between the kinds was rejected: a contract with a parameter and a
  // state variable of the same name would silently bound the wrong one, and
  // "silently bounds the wrong thing" is the failure this whole layer exists to
  // avoid.
  //
  // A `std::function` rather than a plain lambda ONLY so the slot branch can
  // resolve its own KEY through this same function. The alternative was a
  // second copy of the parameter/environment lookups inside that branch, which
  // is the one-fact-two-ledgers shape this file has already paid for elsewhere.
  std::function<bool(const symbolt *, const std::string &, expr2tc &)>
    resolve_coord;
  resolve_coord =
    [&](const symbolt *fsym, const std::string &name, expr2tc &out) -> bool {
    if (
      name.rfind("msg.", 0) == 0 || name.rfind("tx.", 0) == 0 ||
      name.rfind("block.", 0) == 0)
    {
      std::string env = name;
      for (auto &ch : env)
        if (ch == '.')
          ch = '_';
      const symbolt *s = ns.lookup(irep_idt("c:@" + env));
      if (s == nullptr)
        return false;
      out = symbol2tc(migrate_type(s->get_type()), s->id);
      return true;
    }
    if (name.rfind("state.", 0) == 0)
    {
      const std::string field = name.substr(6);

      // ---- `state.<m>[<key>]`: ONE SLOT of a mapping / dynamic array ----
      //
      // WHY THIS IS HERE AND NOT IN THREE PLACES. The certify query, the
      // outer-box ladder and the stage-3 assertion ladder all resolve their
      // coordinates through THIS function, so the single refusal
      //     "a mapping or a dynamic array does not resolve"
      // fired in all three at once.
      //
      // MEASURED, on notes/coverage/poc/P28_MapMin.sol `take` (whole contract,
      // --solidity-max-tx 2): the ONLY quantity separating path 15 from its
      // sibling is the slot `bal[k]`. The driver duly pinned it, stage 2
      // DROPPED the pin ("the certification query cannot express it") and then
      // reported the region as
      //     NOT CERTIFIED -- refuted with no single-coordinate cut available
      // -- a verdict about the coordinate set, not about the region. Every
      // downstream consumer read it as the latter.
      //
      // WHAT IT DOES NOT DO, stated because the gap is easy to misread as
      // closed: this names ONE slot at ONE key. Quantifying over keys
      // (`forall k. bal[k] <= total`) is not a coordinate in the sense of
      // Definition 6 -- a region is a product of per-coordinate SETS -- and is
      // not attempted. A single named slot IS such a coordinate: it is a
      // scalar with an interval, exactly like a scalar state variable.
      const size_t ob = field.find('[');
      if (
        ob != std::string::npos && field.size() > ob + 1 && field.back() == ']')
      {
        const std::string mname = field.substr(0, ob);
        const std::string kname = field.substr(ob + 1, field.size() - ob - 2);
        if (mname.empty() || kname.empty())
          return false;
        // WHICH CONTRACT: the unit's own, from its mangled id. Falling back to
        // scope_contract only when there is no unit symbol -- the substring
        // test resolve_coord uses for the contract OBJECT is wrong here for the
        // reason path_cov_contract_object records (`Escrow` matches
        // `EscrowSrc`), and reading the wrong contract's store would build a
        // bound on a quantity nothing wrote.
        const std::string own =
          fsym != nullptr ? contract_of(fsym->id.as_string()) : scope_contract;
        if (own.empty())
          return false;
        // The frontend lowers a mapping / dynamic array to a CONTRACT-SCOPE
        // GLOBAL `sol:@C@<C>@<name>#<id>` -- which is exactly why the struct
        // walk below cannot see it. Same scan, and the same `@F@` exclusion,
        // that the stage-3 ladder's SECOND SCAN uses to build `store_syms`.
        const std::string cpfx = "sol:@C@" + own + "@";
        const symbolt *store = nullptr;
        cov_context->foreach_operand([&](const symbolt &s) {
          if (store != nullptr)
            return;
          const std::string id = s.id.as_string();
          if (id.rfind(cpfx, 0) != 0 || id.find("@F@") != std::string::npos)
            return;
          std::string nm = id.substr(cpfx.size());
          const size_t hash = nm.find('#');
          if (hash != std::string::npos)
            nm = nm.substr(0, hash);
          if (nm == mname)
            store = &s;
        });
        if (store == nullptr)
          return false;
        const type2tc mt = migrate_type(store->get_type());
        if (!is_array_type(mt))
          return false;
        const type2tc et = to_array_type(mt).subtype;

        // ---- THE KEY: A LITERAL OR A NAME, AND BOTH ARE NEEDED ----
        //
        // Stage 2 takes its key from a COUNTEREXAMPLE, so the spec it writes is
        // `state.bal[0xFFFF...FFFF]`; a hand-written spec naturally says
        // `state.bal[k]` for a parameter, and `state.m[msg.sender]` is the
        // guard shape the paragraph above names as the commonest real one. A
        // name is resolved by recursing into THIS function, so all three cost
        // one branch rather than three lookups.
        //
        // HEX NEEDS ITS `0x`. Testing "is every character a hex digit" without
        // the prefix would read a PARAMETER named `abc` as the number 0xabc --
        // a perfectly well-formed bound on a slot nobody named, which is the
        // silent-wrong-quantity failure this whole layer exists to avoid.
        expr2tc kexpr;
        bool klit = false;
        BigInt kval;
        if (kname.rfind("0x", 0) == 0 || kname.rfind("0X", 0) == 0)
        {
          klit = kname.size() > 2;
          for (size_t i = 2; i < kname.size(); ++i)
            if (isxdigit((unsigned char)kname[i]) == 0)
              klit = false;
          if (klit)
            kval = BigInt(kname.c_str() + 2, 16);
        }
        else
        {
          klit = true;
          for (char c : kname)
            if (c < '0' || c > '9')
              klit = false;
          if (klit)
            kval = string2integer(kname);
        }
        if (klit)
          // uint256 because that is the widest key Solidity has; the SMT layer
          // resizes an index to the array's own domain width, so a narrower
          // key (an `address`) is not mis-indexed by carrying the wider type.
          kexpr = constant_int2tc(get_uint_type(256), kval);
        else if (!resolve_coord(fsym, kname, kexpr))
          return false;

        out = index2tc(et, symbol2tc(mt, store->id), kexpr);
        return true;
      }

      // The contract instance object. Same symbol family the counterexample
      // harvest reads `final_state` from, so a coordinate named here and a
      // value reported there refer to the same thing by construction.
      const symbolt *obj = nullptr;
      cov_context->foreach_operand([&](const symbolt &s) {
        if (obj != nullptr)
          return;
        const std::string id = s.id.as_string();
        if (
          id.rfind("sol:@_ESBMC_Object_", 0) == 0 &&
          (scope_contract.empty() ||
           id.find(scope_contract) != std::string::npos))
          obj = &s;
      });
      if (obj == nullptr)
        return false;
      const typet ostruct = ns.follow(obj->get_type());
      if (ostruct.id() != "struct")
        return false;
      // Same field walk as for struct parameters below, so a struct-typed state
      // variable is reachable field by field too (`state.cfg.limit`). The first
      // segment is the state variable itself.
      out = symbol2tc(migrate_type(ostruct), obj->id);
      return walk_fields(ns, out, field);
    }
    if (fsym == nullptr)
      return false;
    // ---- STRUCT PARAMETERS, BY FIELD ----
    //
    // `immutables.taker` rather than `immutables`. An aggregate has no interval
    // (coord_expressible refuses it, correctly), so a unit whose only argument
    // is a struct had NOTHING generalisable -- measured across all five
    // EscrowSrc units, every one of which reported zero coordinates while its
    // real argument sat right there as a struct.
    //
    // This is NOT a new coordinate KIND and does not touch definition 6: a
    // region is a product of per-coordinate sets, and the fields of a struct
    // are exactly such coordinates. What was missing was the resolution, not
    // the representation. Nested access works by the same walk, so
    // `immutables.timelocks.deployedAt` costs nothing extra.
    const size_t dot = name.find('.');
    const std::string base = name.substr(0, dot);
    for (const auto &arg : to_code_type(fsym->get_type()).arguments())
      if (arg.get_base_name() == base)
      {
        const symbolt *s = ns.lookup(arg.get_identifier());
        if (s == nullptr)
          return false;
        out = symbol2tc(migrate_type(s->get_type()), s->id);
        if (dot == std::string::npos)
          return true;
        return walk_fields(ns, out, name.substr(dot + 1));
      }
    return false;
  };

  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available || f_it->first == "__ESBMC_main")
      continue;
    goto_programt &goto_program = f_it->second.body;
    if (filter(f_it->first, goto_program))
      continue;

    // Only instrument functions living in the user source. c2goto library
    // models and the synthetic _ESBMC_Main dispatcher harness carry
    // non-user locations; enumerating their complete paths is both wrong
    // (not the unit under test) and explodes (thousands of paths). A
    // function is in scope iff at least one of its instructions is in
    // location_pool.
    bool in_user_src = false;
    forall_goto_program_instructions (uit, goto_program)
      if (
        location_pool.count(
          get_filename_from_path(uit->location.file().as_string())))
      {
        in_user_src = true;
        break;
      }
    if (!in_user_src)
      continue;

    // --contract scoping (codex #4): only enumerate functions declared in the
    // target contract; sibling contracts / their helpers are out of the unit
    // under test. Empty scope_contract => no scoping (whole-unit).
    if (
      !scope_contract.empty() &&
      contract_of(f_it->first.as_string()) != scope_contract)
      continue;

    // Skip the lowered custom-error functions (`error E();` becomes a
    // `#sol_error` function whose whole body is ASSUME(false)). They are the
    // lowering of a `revert E()` STATEMENT, not a unit under test: counting
    // their single degenerate path would inflate the denominator with a goal
    // that is uncoverable by construction (its assert sits downstream of the
    // ASSUME(false)) and so permanently reports as undecided.
    {
      const symbolt *fsym = ns.lookup(f_it->first);
      if (fsym && !fsym->get_type().get("#sol_error").as_string().empty())
        continue;
    }

    // The UNIT is a public/external function. An internal/private helper is not
    // something an external caller can invoke, so it has no path set of its own
    // — its decisions live in the units that call it, which the expansion above
    // has already spliced in. Enumerating it separately would report a path set
    // for an input space no test can address, and would double-count the same
    // source in the total.
    if (!is_external_entry(f_it->first))
    {
      ++non_unit_functions;
      continue;
    }

    // ---- --focus-function NARROWS INSTRUMENTATION, NOT ONLY DISPATCH ----
    //
    // Placed HERE, after the unit test, so `non_unit_functions` keeps counting
    // the same population it always did; and after the --contract scope test,
    // so `focus_candidates` lists what a focused run could actually have
    // chosen.
    //
    // The EXPANSION loop above is deliberately NOT narrowed (see the header on
    // `focus_function`): the focused unit's body — and hence every `enc`, every
    // depth and every stable path id — stays bit-identical to a whole-contract
    // run's. That is the whole reason this narrowing is safe, and it is why a
    // callee's decisions are still inside the focused unit's paths.
    //
    // A unit named by an active stage-2/3 spec is NEVER skipped, whatever the
    // focus says. Those three modes each narrow to their own unit at their own
    // branch further down; skipping their target here would leave them with no
    // assume and no assert, and the run would then die at their route-5 gate
    // blaming the unit NAME — pointing the reader at a spelling mistake that
    // does not exist. Same failure shape the outer-box/certify precedence
    // already produced once.
    {
      const std::string uid = f_it->first.as_string();
      focus_candidates.push_back(uid);
      auto spec_names = [&uid](const std::string &spec) {
        return uid == spec || uid.find("@F@" + spec + "#") != std::string::npos;
      };
      const bool stage_target = (outer_on && spec_names(outer_unit)) ||
                                (certify_on && spec_names(certify_unit)) ||
                                (assert_on && spec_names(assert_unit));
      if (
        !focus_function.empty() && !stage_target &&
        !focus_selects_unit(uid, focus_function))
      {
        ++units_skipped_by_focus;
        continue;
      }

      // ---- --path-cov-instrument-only: a SECOND, narrower filter ----
      //
      // Separate from the focus test above rather than folded into it, because
      // the two answer different questions and their counters must stay apart:
      // `units_skipped_by_focus` is "the harness cannot enter this unit", while
      // this one is "the harness CAN enter it and we chose not to measure it".
      // Collapsing them would make a run that deliberately narrowed its
      // denominator indistinguishable from one that narrowed its alphabet, and
      // the published path-coverage percentage means different things in the
      // two cases.
      //
      // Same matcher as the focus test (`focus_selects_unit`) and the same
      // stage-target exemption: a unit named by an active stage-2/3 spec is
      // never skipped, or that mode would run with no assume and no assert and
      // then blame the unit NAME at its route gate.
      if (
        !instrument_only.empty() && !stage_target &&
        !focus_selects_unit(uid, instrument_only))
      {
        // Named on stdout, one line per unit, rather than counted into
        // `units_skipped_by_focus`. The two skips are different facts and a
        // reader must be able to tell them apart: that counter means "the
        // harness cannot enter this unit", this line means "the harness CAN
        // enter it and this run chose not to measure it". A run that
        // deliberately narrowed its DENOMINATOR while widening its ALPHABET is
        // the whole point of the option, and folding it into the other counter
        // would make the published percentage unattributable.
        log_status(
          "--path-cov-instrument-only: '{}' is dispatchable (it is in the "
          "--focus-function set) but NOT instrumented, so it contributes no "
          "paths to the denominator. It can still run and establish state for "
          "the units that are instrumented",
          uid);
        continue;
      }
    }

    ++units_enumerated;
    bool gate_inserted = false;

    // Does this unit still CALL another unit's own (gated) body? See the
    // residual scan above for why that is a model/EVM divergence rather than a
    // loss of resolution. Containment is per unit, matching the assume case.
    // Per-path containment would in fact be sound here — a path that never
    // reaches the call site cannot execute the spurious value-reject edge — but
    // the DFS no longer carries per-site state bits (they were removed when
    // per-path containment was ruled out for the assume case), and marking too
    // much costs tests while marking too little ships a red one.
    const auto residual_units_here =
      residual_unit_callees_of.find(f_it->first.as_string());
    const bool unit_calls_gated_unit =
      residual_units_here != residual_unit_callees_of.end();
    std::string residual_unit_names;
    if (unit_calls_gated_unit)
    {
      for (const auto &n : residual_units_here->second)
        residual_unit_names += (residual_unit_names.empty() ? "" : ", ") + n;
      ++obstacle_units_residual;
    }

    // ABI-layer decision: a non-payable public/external function REVERTS when
    // it is called with value, before a single statement of its body runs. The
    // frontend does not model this (measured: a payable and a non-payable
    // function with identical bodies enumerate identically), which costs two
    // different things:
    //
    //  * a missing path — every non-payable entry has a real, testable
    //    "called with value -> revert" execution that was never enumerated;
    //  * WRONG counterexamples — `msg.value` is re-havoc'd per transaction and
    //    nothing constrained it to zero, so a reported path could carry a
    //    nonzero msg.value that on-chain would revert at entry. That test
    //    cannot replay, which is the one thing this pass must never emit.
    //
    // Synthesise the check here, at the front, so Phase 1 snapshots it and
    // Phase 2 enumerates it like any other decision:
    //
    //     IF msg_value == 0 THEN GOTO <original first instruction>
    //     _ESBMC_sol_mark_revert();      // makes exit_kind = revert
    //     GOTO <END_FUNCTION>
    //
    // `#sol_payable` is already stamped on the function type
    // (solidity_convert_modifier.cpp), so no frontend change is needed to know
    // where the gate applies.
    //
    // Placing it in the body is correct here ONLY because of the expansion
    // above. The gate lives in the external wrapper on-chain, not in the
    // function body, and goto has one body per function — so before expansion,
    // a `public` function that was also called internally had ONE body serving
    // two entry kinds, and the gate invented a revert on the internal one
    // (measured: `g() payable` internally calling `f() public` admitted an
    // execution where `f` took the value-reject edge, which cannot happen
    // on-chain). After expansion the internal caller holds its own gate-free
    // COPY of the callee, and this body is reachable only through the
    // dispatcher — the exact entry kind the gate models. Both entry kinds are
    // now right, with no precondition to weaken.
    {
      const symbolt *fsym = ns.lookup(f_it->first);
      const bool is_payable =
        fsym != nullptr && fsym->get_type().get_bool("#sol_payable");
      const symbolt *mv = ns.lookup(irep_idt("c:@msg_value"));
      const symbolt *mark = ns.lookup(irep_idt("c:@F@_ESBMC_sol_mark_revert"));
      if (
        !is_payable && mv != nullptr && mark != nullptr &&
        !goto_program.instructions.empty())
      {
        auto body_start = goto_program.instructions.begin();
        auto end_fn = std::prev(goto_program.instructions.end());
        if (end_fn->is_end_function())
        {
          const locationt loc = body_start->location;
          const irep_idt fn = body_start->location.get_function();
          expr2tc mv_expr = symbol2tc(migrate_type(mv->get_type()), mv->id);
          expr2tc zero = gen_zero(migrate_type(mv->get_type()));

          // Plain list insertion, NOT insert_swap. insert_swap moves the
          // instruction's CONTENT, so the iterator that named the original
          // first instruction ends up naming the newly inserted one — the
          // branch below then targets itself and the function acquires a
          // self-loop. (Measured before this was fixed: a two-path function
          // enumerated 64 paths and reported a truncated loop.) std::list
          // insert keeps `body_start` attached to the original instruction.
          goto_programt::instructiont brk;
          brk.type = GOTO;
          brk.guard = equality2tc(mv_expr, zero);
          brk.location = loc;
          // MARK IT AS SYNTHETIC AT THE SOURCE. This decision exists in the
          // path metric and in NO other metric, and its location is the one
          // copied from the unit's first body instruction just above — so any
          // consumer that projects walked decisions onto source lines would
          // credit itself with whatever real decision sits on that line. The
          // alternative (recognising it downstream by its condition text) is a
          // string match on a lowering detail, i.e. a check that stops firing
          // silently the day `msg_value` is renamed.
          brk.location.set("sol_abi_value_gate", true);
          brk.function = fn;
          auto it_brk = goto_program.instructions.insert(body_start, brk);

          goto_programt::instructiont call;
          call.type = FUNCTION_CALL;
          call.code = code_function_call2tc(
            expr2tc(),
            symbol2tc(migrate_type(mark->get_type()), mark->id),
            std::vector<expr2tc>());
          call.location = loc;
          call.function = fn;
          goto_program.instructions.insert(body_start, call);

          goto_programt::instructiont jmp;
          jmp.type = GOTO;
          jmp.guard = gen_true_expr();
          jmp.location = loc;
          jmp.function = fn;
          auto it_jmp = goto_program.instructions.insert(body_start, jmp);

          // Targets set after insertion: `body_start` still names the original
          // first instruction, `end_fn` the END_FUNCTION.
          it_brk->targets.clear();
          it_brk->targets.push_back(body_start);
          it_jmp->targets.clear();
          it_jmp->targets.push_back(end_fn);

          goto_program.compute_target_numbers();
          // Recorded for the expansion-ratio measurement: the gate contributes
          // exactly ONE extra path (its reject edge terminates immediately),
          // and it is absent from the pre-expansion snapshot, so it must be
          // discounted before the two counts are compared.
          gate_inserted = true;
        }
      }
    }

    // Loops are handled by the tr accumulator (survives unrolling) plus a
    // bounded DFS (each back-edge followed at most path_cov_unwind times).

    // Phase 1: one integer path-number accumulator `tr` per function.
    // tr starts at 1 (a leading sentinel bit so different-length prefixes
    // stay distinct) and each decision does `tr = tr*2 + guard_value`. A
    // single scalar records the whole decision sequence in order and — unlike
    // one bool per static decision — survives loop unrolling, because symex
    // re-runs the update on every iteration (this is the Slice-2 enabler).
    const type2tc utype = get_uint_type(64);
    symbolt sym;
    sym.set_type(unsignedbv_typet(64));
    sym.name = "__ESBMC_path_tr$" + i2string(ghost_counter++);
    sym.id = "path_cov::" + id2string(sym.name);
    sym.lvalue = true;
    sym.static_lifetime = false;
    sym.is_extern = false;
    symbolt *psym;
    cov_context->move(sym, psym);
    expr2tc tr = symbol2tc(migrate_type(psym->get_type()), psym->id);
    irep_idt tr_id = psym->id;

    // Companion decision-COUNT ghost `cnt` (starts 0, +1 per decision). The
    // exit assert checks tr==enc AND cnt==depth, so a feasible path with more
    // than 64 decisions — whose 64-bit tr WRAPS — can never spuriously match a
    // shorter emitted path's enc (its cnt = true length differs). Without this
    // a wrapped tr could fire another path's assert => a WRONG test (codex #1).
    symbolt csym;
    csym.set_type(unsignedbv_typet(64));
    csym.name = "__ESBMC_path_cnt$" + i2string(ghost_counter++);
    csym.id = "path_cov::" + id2string(csym.name);
    csym.lvalue = true;
    csym.static_lifetime = false;
    csym.is_extern = false;
    symbolt *pcsym;
    cov_context->move(csym, pcsym);
    expr2tc cnt = symbol2tc(migrate_type(pcsym->get_type()), pcsym->id);
    irep_idt cnt_id = pcsym->id;

    // Snapshot one decision: insert `tr = tr*2 + (uint64)val; cnt = cnt+1`
    // before `it` (leaving `it` unchanged), both marked "skipped" so they are
    // not coverage claims. For K decisions at the same site, call in order —
    // the first snapshotted becomes the higher-order bit, matching the DFS.
    auto snapshot = [&](goto_programt::targett &sit, const expr2tc &val) {
      goto_programt::instructiont a;
      a.type = ASSIGN;
      a.code = code_assign2tc(
        tr,
        add2tc(
          utype,
          mul2tc(utype, tr, constant_int2tc(utype, BigInt(2))),
          typecast2tc(utype, val)));
      a.location = sit->location;
      a.location.property("skipped");
      a.function = sit->location.get_function();
      goto_program.insert_swap(sit++, a);
      --sit;
      goto_programt::instructiont b;
      b.type = ASSIGN;
      b.code = code_assign2tc(
        cnt, add2tc(utype, cnt, constant_int2tc(utype, BigInt(1))));
      b.location = sit->location;
      b.location.property("skipped");
      b.function = sit->location.get_function();
      goto_program.insert_swap(sit++, b);
      --sit;
    };

    // ---- `tr` COMPLETENESS: every decision the DFS branches on must be
    // ---- accounted for in the runtime accumulator.
    //
    // This is the invariant the whole stage-3 certification query rests on.
    // The query is `assume(L <= x <= U); assert(tr == pi)`, and it is sound
    // ONLY because `tr` records the complete decision sequence of whatever
    // execution actually happens — including executions of paths that were
    // never enumerated (dropped at the goal cap). That property is what makes
    // certification immune to truncation, and until now it was an ARGUMENT in a
    // comment, not something the tool checked.
    //
    // The fatal direction is one-sided: a site the DFS fans out on but Phase 1
    // did NOT snapshot. Then every real execution carries fewer accumulated
    // decisions than the emitted path expects, `cnt != depth` holds always, and
    // the path is permanently uncoverable — while being reported PASSED, i.e. a
    // false proof. That is not hypothetical: it is exactly the short-circuit
    // cap mismatch measured earlier (26 operands => `Reached : 0`, reported
    // PASSED). This assertion would have caught it at instrumentation time.
    //
    // The other direction (snapshotted but never traversed) is benign — dead
    // code — so it is not an error.
    std::set<std::pair<std::string, unsigned>> phase1_decision_sites;
    std::set<std::pair<std::string, unsigned>> dfs_decision_sites;

    // At each decision: snapshot its value into tr. Conditional GOTOs (guard)
    // AND folded short-circuit &&/|| / ternary operands in ASSIGN/RETURN — the
    // latter carry no GOTO, so branch_coverage collects them via
    // collect_short_circuit_decisions; we mirror that (codex #2), snapshotting
    // each in collect order (matched by the DFS fan-out).
    Forall_goto_program_instructions (it, goto_program)
    {
      if (it->is_goto() && !is_true(it->guard))
      {
        phase1_decision_sites.emplace(it->location.as_string(), 0u);
        snapshot(it, it->guard);
      }
      else if (
        (it->is_assign() || it->is_return()) &&
        it->location.property().as_string() != "skipped")
      {
        const expr2tc &src = it->is_assign()
                               ? to_code_assign2t(it->code).source
                               : to_code_return2t(it->code).operand;
        std::vector<expr2tc> ops;
        collect_short_circuit_decisions(
          src, [&](const expr2tc &e) { ops.push_back(e); });
        // Same cap as the Phase-2 fan-out. Snapshotting operands the DFS will
        // not enumerate would desynchronise cnt from every emitted depth and
        // silently make the whole site's paths uncoverable.
        if (ops.size() > SC_DECISION_MAX)
        {
          ++sc_sites_over_cap;
          continue;
        }
        for (unsigned j = 0; j < ops.size(); ++j)
        {
          phase1_decision_sites.emplace(it->location.as_string(), j);
          snapshot(it, ops[j]);
        }
      }
    }

    // ---- MATERIALISE THE UNIT'S OWN RETURN VALUE ----
    //
    // MEASURED before this existed, on notes/coverage/poc/P19_ReturnShapes.sol
    // unit `tern_lit` (GATE cell, --verbosity coverage:9): bmc.cpp's harvest
    // classified 208 assignments and NOT ONE was the unit's return. The reason
    // is visible in the goto dump: the dispatcher calls a unit with NO lvalue
    // (`FUNCTION_CALL: tern_lit(&obj, NONDET, NONDET)`) and the RETURN carries
    // an EXPRESSION, never a write to a symbol. So the value does not exist as
    // an assignment anywhere -- the rival explanation, "it is written but after
    // the harvest's break", is refuted by the same dump.
    //
    // A SEPARATE PASS, not a branch inside the Phase-1 loop above. That loop's
    // insertion ORDER is what pairs each snapshot with the DFS fan-out, and
    // interleaving another insertion with it would be a change to the decision
    // accounting rather than to the payload.
    //
    // ORDERING IS THE WHOLE POINT AND IT IS NOT INCIDENTAL. This pass inserts
    // immediately before the RETURN; Phase 2 then inserts this path's asserts
    // immediately before the RETURN as well, so the final order is
    // ASSIGN, ASSERT(s), RETURN -- verified against the dump above, where the
    // Phase-1 tr/cnt updates likewise sit in front of the Phase-2 asserts. It
    // has to be this way round: bmc.cpp's harvest STOPS at this path's own
    // assert, so a write placed after it would never be seen. That is exactly
    // why binding the value at the call site instead would not have been
    // enough.
    //
    // THIS ADDS NO DECISION. The inserted ASSIGN carries property("skipped")
    // like every other ghost write here, the DFS does not fan out on it, and no
    // `enc` or depth changes -- so DECISION_SET_VERSION must NOT be bumped for
    // it (bumping would discard every existing covered set for nothing). The
    // check that this held is that the enumerated path count is unchanged.
    //
    // SCALARS ONLY, deliberately. A tuple / struct / dynamic return has no
    // single renderable value; the ghost is simply not created,
    // `return_value_known` stays false, and the report says the value is
    // UNKNOWN rather than claiming the unit returns nothing. TWO ghosts, not
    // one, and the second is not optional. MEASURED with only the value ghost:
    // `tern_lit`'s REVERT path (enc=2, which never reaches a RETURN) reported
    // `return_value_known=true, return_value="0"` -- the entry initialisation,
    // published as if the unit had returned it. "Was a value returned on this
    // execution" is a runtime fact and has to be recorded at runtime; it cannot
    // be recovered from the value, because 0 is also a perfectly good return
    // value.
    expr2tc ret_ghost, retset_ghost;
    irep_idt ret_ghost_id, retset_ghost_id;
    bool has_ret_ghost = false;
    // ONE retset for the whole unit, whatever shape its return has. Created
    // lazily by whichever pass finds a return first, so the scalar hook and the
    // tuple hook below cannot each make their own -- two "was a value
    // returned" flags is one fact in two ledgers, and the ladder's witness
    // would then be about whichever one it happened to read.
    auto ensure_retset = [&]() {
      if (!is_nil_expr(retset_ghost))
        return;
      symbolt ssym;
      ssym.set_type(bool_typet());
      ssym.name = "__ESBMC_path_retset$" + i2string(ghost_counter++);
      ssym.id = "path_cov::" + id2string(ssym.name);
      ssym.lvalue = true;
      ssym.static_lifetime = false;
      ssym.is_extern = false;
      symbolt *pssym;
      cov_context->move(ssym, pssym);
      retset_ghost = symbol2tc(migrate_type(pssym->get_type()), pssym->id);
      retset_ghost_id = pssym->id;
      if (protect_ce_symbols)
        config.no_slice_names.insert(retset_ghost_id.as_string());
    };
    {
      Forall_goto_program_instructions (rit, goto_program)
      {
        if (!rit->is_return() || !is_code_return2t(rit->code))
          continue;
        const expr2tc rv = to_code_return2t(rit->code).operand;
        if (is_nil_expr(rv))
          continue;
        const type2tc rt = rv->type;
        if (
          !is_unsignedbv_type(rt) && !is_signedbv_type(rt) && !is_bool_type(rt))
          continue;
        if (!has_ret_ghost)
        {
          symbolt rsym;
          rsym.set_type(migrate_type_back(rt));
          rsym.name = "__ESBMC_path_ret$" + i2string(ghost_counter++);
          rsym.id = "path_cov::" + id2string(rsym.name);
          rsym.lvalue = true;
          rsym.static_lifetime = false;
          rsym.is_extern = false;
          symbolt *prsym;
          cov_context->move(rsym, prsym);
          ret_ghost = symbol2tc(migrate_type(prsym->get_type()), prsym->id);
          ret_ghost_id = prsym->id;
          ensure_retset();

          // ---- EXEMPT BOTH FROM SLICING, OR THE WHOLE THING IS DEAD ----
          //
          // MEASURED, and it is why this is here rather than assumed
          // unnecessary: with the ghosts written and correctly ordered, the
          // report still came back `return_value_known=false` on every path,
          // while the SAME run under --no-slice reported 20 and 10. The symex
          // slicer works backwards from the claim, and a path claim's guard is
          // `tr != enc || cnt != depth` -- it mentions the accumulators and
          // nothing else. `tr`/`cnt` therefore survive for free and these two
          // do not: nothing downstream reads them, so they are dead by the
          // slicer's own (correct) reckoning.
          //
          // Registered only under `protect_ce_symbols`, the same condition the
          // block above uses for the contract object / stores / environment, so
          // a run that is not harvesting a payload keeps slicing exactly what
          // it sliced before.
          if (protect_ce_symbols)
            config.no_slice_names.insert(ret_ghost_id.as_string());
          has_ret_ghost = true;
        }
        // A unit has ONE return type, so a second RETURN of a different type
        // means the model disagrees with the source. Cast rather than assume:
        // an unexpected shape becomes a value to look at, not an abort.
        goto_programt::instructiont ra;
        ra.type = ASSIGN;
        ra.code = code_assign2tc(
          ret_ghost,
          ret_ghost->type == rt ? rv : typecast2tc(ret_ghost->type, rv));
        ra.location = rit->location;
        ra.location.property("skipped");
        ra.function = rit->location.get_function();
        goto_program.insert_swap(rit++, ra);
        --rit;
        // Immediately after the value and before the RETURN, so the two are set
        // by the same execution or by neither.
        goto_programt::instructiont rf;
        rf.type = ASSIGN;
        rf.code = code_assign2tc(retset_ghost, gen_true_expr());
        rf.location = rit->location;
        rf.location.property("skipped");
        rf.function = rit->location.get_function();
        goto_program.insert_swap(rit++, rf);
        --rit;
      }
    }

    // ---- TUPLE RETURNS: ONE GHOST PER MEMBER ----
    //
    // MEASURED on notes/coverage/poc/P27_TupleReturn.sol: a unit declared
    // `returns (uint256, uint256)` emits NO RETURN INSTRUCTION AT ALL. It
    // lowers to
    //     ASSIGN tuple_instance$42.mem0 = 11;
    //     ASSIGN tuple_instance$42.mem1 = 12;
    // and falls through to END_FUNCTION -- so the pass above, which hooks the
    // RETURN, sees nothing, and every tuple-returning unit had no return
    // candidate whatever. On the corpus that is not an edge case: aqua's two
    // value-returning units are exactly this shape.
    //
    // THE INSTANCE IS TIED TO THIS UNIT BY ITS OWN AST NODE ID.
    // `two_scalars#42` owns `tuple_instance$42`, which is the same key
    // bmc.cpp's counterexample harvest uses, so the two cannot drift and an
    // INLINED CALLEE's tuple -- which carries the callee's node id -- cannot be
    // mistaken for this unit's. Matching the name as a plain substring would do
    // exactly that, and would also let `tuple_instance$4` answer for
    // `tuple_instance$42`; hence the explicit end-of-id test below.
    //
    // The ghost is assigned AFTER the member write, not before it: it is the
    // written value that is the return value.
    std::map<unsigned long, expr2tc> ret_member_ghosts;
    std::map<unsigned long, irep_idt> ret_member_ids;
    std::map<unsigned long, std::string> ret_member_refused;
    if (!has_ret_ghost)
    {
      const std::string uid0 = f_it->first.as_string();
      const size_t hash0 = uid0.rfind('#');
      const std::string want = hash0 == std::string::npos
                                 ? std::string()
                                 : "sol:@C@" + contract_of(uid0) +
                                     "@tuple_instance$" +
                                     uid0.substr(hash0 + 1);
      if (!want.empty())
      {
        Forall_goto_program_instructions (tit, goto_program)
        {
          if (!tit->is_assign() || !is_code_assign2t(tit->code))
            continue;
          if (tit->location.property().as_string() == "skipped")
            continue;
          const expr2tc &lhs = to_code_assign2t(tit->code).target;
          if (!is_member2t(lhs) || !is_symbol2t(to_member2t(lhs).source_value))
            continue;
          const std::string bid =
            to_symbol2t(to_member2t(lhs).source_value).thename.as_string();
          if (bid.rfind(want, 0) != 0)
            continue;
          const std::string tail = bid.substr(want.size());
          if (!tail.empty() && tail[0] != '#')
            continue;
          const std::string mem = to_member2t(lhs).member.as_string();
          if (mem.rfind("mem", 0) != 0)
            continue;
          char *endp = nullptr;
          const unsigned long k = strtoul(mem.c_str() + 3, &endp, 10);
          if (endp == nullptr || *endp != '\0')
            continue;
          const type2tc mt = lhs->type;
          if (
            !is_unsignedbv_type(mt) && !is_signedbv_type(mt) &&
            !is_bool_type(mt))
          {
            // NAMED, not skipped. A member absent from the table would read as
            // "measured and unconstrained", the same misreading the whole-unit
            // refusal exists to prevent -- one member down.
            ret_member_refused[k] =
              "member " + std::to_string(k) +
              " is not a scalar the ghost can hold (aggregate / dynamic type), "
              "so no candidate is formed for it";
            continue;
          }
          if (ret_member_ghosts.count(k) == 0)
          {
            symbolt msym;
            msym.set_type(migrate_type_back(mt));
            msym.name = "__ESBMC_path_ret" + i2string((unsigned)k) + "$" +
                        i2string(ghost_counter++);
            msym.id = "path_cov::" + id2string(msym.name);
            msym.lvalue = true;
            msym.static_lifetime = false;
            msym.is_extern = false;
            symbolt *pm;
            cov_context->move(msym, pm);
            ret_member_ghosts[k] =
              symbol2tc(migrate_type(pm->get_type()), pm->id);
            ret_member_ids[k] = pm->id;
            // Same slicing exemption, and for the same measured reason: nothing
            // downstream reads these unless a rung does, and the report side
            // does not.
            if (protect_ce_symbols)
              config.no_slice_names.insert(pm->id.as_string());
            ensure_retset();
          }
          // Plain list insert AFTER the write. `tit` keeps naming the original
          // instruction (insert_swap would move its CONTENT and make the
          // iterator name the new one -- the self-loop the ABI gate produced
          // once already), and the loop's own increment then steps onto the
          // inserted ASSIGN, which is `skipped` and matches nothing here.
          goto_programt::instructiont ma;
          ma.type = ASSIGN;
          ma.code = code_assign2tc(
            ret_member_ghosts[k],
            ret_member_ghosts[k]->type == mt
              ? lhs
              : typecast2tc(ret_member_ghosts[k]->type, lhs));
          ma.location = tit->location;
          ma.location.property("skipped");
          ma.function = tit->location.get_function();
          goto_program.instructions.insert(std::next(tit), ma);
          goto_programt::instructiont mf;
          mf.type = ASSIGN;
          mf.code = code_assign2tc(retset_ghost, gen_true_expr());
          mf.location = tit->location;
          mf.location.property("skipped");
          mf.function = tit->location.get_function();
          goto_program.instructions.insert(std::next(tit), mf);
        }
      }
    }

    // DECL tr and initialise `tr = 1` at function entry (in that order),
    // both before the original first instruction.
    {
      auto entry = goto_program.instructions.begin();
      locationt eloc = entry->location;
      irep_idt efn = entry->location.get_function();
      goto_programt::instructiont dcl;
      dcl.type = DECL;
      dcl.code = code_decl2tc(utype, tr_id);
      dcl.location = eloc;
      dcl.location.property("skipped");
      dcl.function = efn;
      goto_program.insert_swap(entry++, dcl); // DECL before entry
      --entry;                                // entry back at original
      goto_programt::instructiont ini;
      ini.type = ASSIGN;
      ini.code = code_assign2tc(tr, constant_int2tc(utype, BigInt(1)));
      ini.location = eloc;
      ini.location.property("skipped");
      ini.function = efn;
      goto_program.insert_swap(entry++, ini); // ASSIGN after DECL, before orig
      --entry;
      goto_programt::instructiont cdcl;
      cdcl.type = DECL;
      cdcl.code = code_decl2tc(utype, cnt_id);
      cdcl.location = eloc;
      cdcl.location.property("skipped");
      cdcl.function = efn;
      goto_program.insert_swap(entry++, cdcl);
      --entry;
      goto_programt::instructiont cini;
      cini.type = ASSIGN;
      cini.code = code_assign2tc(cnt, constant_int2tc(utype, BigInt(0)));
      cini.location = eloc;
      cini.location.property("skipped");
      cini.function = efn;
      goto_program.insert_swap(entry++, cini);
      --entry;
      // The return-value ghost, declared and zeroed alongside tr/cnt. The zero
      // is NOT a value claim: a path that reverts before reaching any RETURN
      // leaves it at 0, and `return_value_known` in the payload is the only
      // thing that says whether a reported 0 means anything. Without the DECL
      // the assignment above would reference an undeclared symbol.
      if (has_ret_ghost)
      {
        goto_programt::instructiont rdcl;
        rdcl.type = DECL;
        rdcl.code = code_decl2tc(ret_ghost->type, ret_ghost_id);
        rdcl.location = eloc;
        rdcl.location.property("skipped");
        rdcl.function = efn;
        goto_program.insert_swap(entry++, rdcl);
        --entry;
        goto_programt::instructiont rini;
        rini.type = ASSIGN;
        rini.code = code_assign2tc(ret_ghost, gen_zero(ret_ghost->type));
        rini.location = eloc;
        rini.location.property("skipped");
        rini.function = efn;
        goto_program.insert_swap(entry++, rini);
        --entry;
      }
      // The same DECL + zero for every TUPLE MEMBER ghost. Zero is not a value
      // claim here either -- `retset` is the only thing that says whether any
      // of them means anything.
      for (const auto &[k, g] : ret_member_ghosts)
      {
        goto_programt::instructiont mdcl;
        mdcl.type = DECL;
        mdcl.code = code_decl2tc(g->type, ret_member_ids[k]);
        mdcl.location = eloc;
        mdcl.location.property("skipped");
        mdcl.function = efn;
        goto_program.insert_swap(entry++, mdcl);
        --entry;
        goto_programt::instructiont mini;
        mini.type = ASSIGN;
        mini.code = code_assign2tc(g, gen_zero(g->type));
        mini.location = eloc;
        mini.location.property("skipped");
        mini.function = efn;
        goto_program.insert_swap(entry++, mini);
        --entry;
      }
      // ONE retset, declared when EITHER shape produced a ghost.
      if (!is_nil_expr(retset_ghost))
      {
        goto_programt::instructiont sdcl;
        sdcl.type = DECL;
        sdcl.code = code_decl2tc(retset_ghost->type, retset_ghost_id);
        sdcl.location = eloc;
        sdcl.location.property("skipped");
        sdcl.function = efn;
        goto_program.insert_swap(entry++, sdcl);
        --entry;
        // FALSE at entry is the whole point: a path that reverts before any
        // RETURN keeps it false, and the harvest then publishes UNKNOWN instead
        // of the initialisation value. Without this the revert path of
        // P19_ReturnShapes.tern_lit reported a returned `0`.
        goto_programt::instructiont sini;
        sini.type = ASSIGN;
        sini.code = code_assign2tc(retset_ghost, gen_false_expr());
        sini.location = eloc;
        sini.location.property("skipped");
        sini.function = efn;
        goto_program.insert_swap(entry++, sini);
        --entry;
      }
    }

    goto_program.compute_target_numbers();

    // Phase 2: bounded DFS over complete entry->exit decision sequences.
    // Each path's number enc mirrors the runtime tr (start 1; at a decision
    // enc*2+1 for the guard-true/taken successor, enc*2+0 for
    // guard-false/fallthrough), so `assert(tr != enc)` at the exit is
    // falsified exactly on that path. Loops: a back-edge (goto whose target
    // is earlier) is followed at most path_cov_unwind times per path, so
    // paths are enumerated up to that many iterations — matching the symex
    // --unwind bound. State per path: (pc, enc, back-edge-follow count).
    // State per path: (pc, enc, per-loop back-edge counts, decision depth).
    // Each loop is keyed by its head (the back-edge's target target_number),
    // so nested loops get INDEPENDENT budgets (a single shared counter would
    // make outer+inner share path_cov_unwind and miss valid nested paths;
    // symex unwinds each loop independently). codex #3.
    // 5th field: has this partial path already walked over a rollback restore
    // (i.e. it is a require/revert("msg") reverting path)?
    // ---- WHERE TWO PATHS ACTUALLY DIVERGE ----
    //
    // The reach gate reports "the witness agrees with this path's
    // counterexample on every scalar in the payload", and the question that
    // leaves open is WHICH quantity separates them. It is not in the payload,
    // so no comparison of payloads can name it. The decision SITE can: the
    // guard at the site where two sibling paths diverge is where the quantity
    // is written in source.
    //
    // No new query and no change to the DFS state. `enc` already IS the prefix
    // identity -- it starts at 1 and each decision appends a bit -- so
    // recording "the enc value this decision produced -> the site that produced
    // it" is enough to walk any path's ordered sequence afterwards: its
    // decisions sit at enc>>(depth-1), enc>>(depth-2), ..., enc. Two paths
    // sharing a prefix share those keys by construction, so the map cannot be
    // ambiguous.
    //
    // Local to the unit, because enc values collide across units.
    //
    // Gated, and the gate is the reason this is affordable: only the unit an
    // outer-box batch is actually targeting records anything. A whole-contract
    // run would otherwise pay for units like `ship`, which enumerates 2733
    // paths and is never the one being asked about.
    // Interned: the DESCRIPTOR table is per distinct decision site (tens), and
    // only the prefix->index map is per path prefix. That is the difference
    // between this being affordable on a 120166-path unit and not: the
    // log-only recorder this replaces stored a fresh string per prefix, which
    // is why it had to be gated behind a single-unit spec.
    std::vector<path_decisiont> dec_table;
    std::map<std::string, uint32_t> dec_intern;
    std::map<uint64_t, uint32_t> dec_index;

    // ---- R0 EVENT RUNG: per-prefix emit sequence ----
    //
    // Same prefix-keyed scheme the decisions use, and for the same reason: a
    // path's sequence is recoverable afterwards from enc>>(depth-1) ... enc, so
    // nothing has to be carried in the DFS stack. Carrying it there was the
    // rejected alternative -- it would put a std::vector in a per-branch stack
    // entry, which is exactly the cost dec_table/dec_intern were introduced to
    // remove (the recorder they replaced stored a fresh string per prefix and
    // had to be gated behind a single-unit spec; the units quoted are 2733 and
    // 120166 paths).
    //
    // TWO PROPERTIES THE DECISION SCHEME DOES NOT NEED AND THIS ONE DOES:
    //  * IDEMPOTENCE. dec_index ASSIGNS, so re-walking a prefix is harmless.
    //    An event list would APPEND, and a prefix is re-walked once per branch
    //    explored beneath it -- a naive append multiplies every event by the
    //    number of paths under it. Keying the inner map by the instruction's
    //    program POSITION makes a re-walk overwrite the same slot with the same
    //    value instead of growing the list.
    //  * ORDER WITHIN A PREFIX. Several emits can sit between two decisions, so
    //    the inner container is ordered by that same position rather than being
    //    a set.
    // Names are interned: the table is per distinct event (tens), only the
    // position keys are per prefix.
    std::vector<std::string> ev_table;
    std::map<std::string, uint32_t> ev_intern;
    std::map<uint64_t, std::map<uint32_t, uint32_t>> ev_index;
    // Program position of each instruction. Pointer identity is stable for the
    // life of the program and iterators are not orderable, so one pass builds
    // the total order the inner map is keyed by.
    std::unordered_map<const goto_programt::instructiont *, uint32_t> ev_pos;
    {
      uint32_t n = 0;
      forall_goto_program_instructions (pi, goto_program)
        ev_pos[&*pi] = n++;
    }
    // Recover a complete path's emit sequence: every prefix it passed through,
    // in order, and within each prefix every emit in program position order.
    auto events_for = [&ev_index, &ev_table](uint64_t enc, uint64_t depth) {
      std::vector<std::string> out;
      for (uint64_t k = depth + 1; k-- > 0;)
      {
        auto it = ev_index.find(enc >> k);
        if (it == ev_index.end())
          continue;
        for (const auto &[pos, id] : it->second)
        {
          (void)pos;
          out.push_back(ev_table[id]);
        }
      }
      return out;
    };
    // The gate names ONE unit per mode, and the `outer_on` / `assert_on` guard
    // in front of each name is not decoration: with the mode off the spec name
    // is empty and `find("@F@" + "" + "#")` matches every unit in the program
    // -- precisely the whole-contract cost this gate exists to avoid.
    auto spec_names_this_unit = [&f_it](const std::string &spec) {
      return f_it->first.as_string() == spec ||
             f_it->first.as_string().find("@F@" + spec + "#") !=
               std::string::npos;
    };
    const bool trace_decisions =
      (outer_on && spec_names_this_unit(outer_unit)) ||
      (assert_on && spec_names_this_unit(assert_unit));
    // Recording for the REPORT is not gated on the outer-box spec: the
    // projection needs every unit's sequences, and the outer-box gate names
    // one.
    const bool record_decisions = trace_decisions || emit_decision_sites;
    // Register the decision that produced prefix value `key_enc`. `cond` is the
    // decision's own expression, so both branch-claim arm texts are built HERE,
    // with the same from_expr/gen_not_expr the branch metric uses — a consumer
    // reconstructing them would be reimplementing the expression printer.
    auto note_decision = [&](
                           uint64_t key_enc,
                           const locationt &l,
                           const expr2tc &cond,
                           unsigned sub) {
      if (!record_decisions)
        return;
      const std::string loc = l.as_string();
      const std::string ikey =
        loc + "\t" + std::to_string(sub) + "\t" + from_expr(ns, "", cond);
      auto ins = dec_intern.emplace(ikey, (uint32_t)dec_table.size());
      if (ins.second)
      {
        path_decisiont d;
        d.loc = loc;
        d.cond_arm_false = from_expr(ns, "", cond);
        d.cond_arm_true = from_expr(ns, "", gen_not_expr(cond));
        d.sub = sub;
        d.synthetic_abi_gate = l.get_bool("sol_abi_value_gate");
        dec_table.push_back(d);
      }
      dec_index[key_enc] = ins.first->second;
    };

    using becntt = std::map<unsigned, unsigned>;
    // Per-site occurrence counter for the content-addressed path id: how many
    // times this partial path has already passed through each decision SITE.
    // The count is part of the key, so a loop's 2nd traversal of a decision is
    // a different element of the sequence than its 1st.
    using occt = std::map<uint64_t, unsigned>;
    // 6th field: has this partial path walked through the function epilogue?
    // 7th/8th: running content-addressed id, and its occurrence counters.
    // 9th field: has this partial path walked over a source-level decision that
    // the frontend lowered away (see is_lost_decision below)?
    std::vector<std::tuple<
      goto_programt::targett,
      uint64_t,
      becntt,
      uint64_t,
      bool,
      bool,
      uint64_t,
      occt>>
      stack;
    // Seed the id with the unit signature, so the finished hash IS the path id.
    const uint64_t unit_seed = fnv1a("unit:" + id2string(f_it->first));
    stack.push_back(
      {goto_program.instructions.begin(),
       (uint64_t)1,
       becntt{},
       (uint64_t)0,
       false,
       false,
       unit_seed,
       occt{}});

    // DECISION-SET CENSUS (symmetric to the exit census below, and aimed at a
    // strictly worse failure).
    //
    // A source-level `require(c)` in an internal library or a free function is
    // lowered to a bare `assume(c)` with NO control flow: the `!c` execution
    // does not exist in the model at all, while on-chain it reverts. Measured:
    // a contract whose only guard lives in an internal library enumerates two
    // paths, neither of which is the revert.
    //
    // The consequence is not a wrong label, it is a wrong TEST. `!c` inputs
    // belong to no enumerated path, so the stage-3 subtraction never removes
    // them; the interval bound is a syntactic product of ranges and cannot
    // carry `c`; and the stage-3 assertion query runs under the same
    // `assume(c)`, so the verifier certifies a candidate over inputs it has
    // never seen. The emitted test then reverts on the unmodified contract
    // while carrying a certified label — the one outcome this pipeline must
    // never produce.
    //
    // So a path walking such a site is a NAMED OBSTACLE, not an inaccuracy.
    // An explicitly written `__ESBMC_assume` is indistinguishable from a
    // lowered `require` here and is marked too: over-marking costs a test,
    // under-marking ships a red one.
    auto is_lost_decision = [&](goto_programt::const_targett i) -> bool {
      if (!i->is_assume())
        return false;
      if (i->location.property().as_string() == "skipped")
        return false;
      return location_pool.count(
               get_filename_from_path(i->location.file().as_string())) != 0;
    };
    // CONTAINMENT IS PER UNIT, NOT PER PATH, and the difference is soundness
    // rather than caution. The missing revert is not a marked sibling — it is
    // NOT A SIBLING AT ALL, so nothing subtracts it from anything. A path that
    // never goes near the lowered site is still unsafe: its bound is a
    // syntactic over-approximation that can cover the missing path's inputs,
    // and the subtraction cannot remove what was never enumerated. Per-path
    // containment is safe only when the missing path happens to be separated
    // from the survivor on the very coordinate the split chose — luck that
    // cannot be established case by case. So any user-source ASSUME anywhere in
    // the unit disqualifies EVERY path of that unit.
    std::set<std::string> lost_decision_locs;
    forall_goto_program_instructions (li, goto_program)
      if (is_lost_decision(li))
        lost_decision_locs.insert(li->location.as_string());
    const bool unit_has_lost_decision = !lost_decision_locs.empty();
    if (unit_has_lost_decision)
      ++obstacle_units;

    // ---- R0 EVENT RUNG: the emits this unit carries, IN PROGRAM ORDER ----
    //
    // `expand_here` stamps `sol_emit_name` on the instruction that used to be
    // the emit call. That instruction is not deleted -- it becomes a LOCATION
    // and stays in place -- so walking the unit here recovers the sequence at
    // exactly the position the path walk will meet it.
    //
    // ⛔ DO NOT ADD an is_function_call() filter. By this point an emit is a
    // LOCATION, and three earlier versions of this census read zero for
    // precisely that reason while the stamp was working.
    //
    // This is still the OBSERVATION half. Nothing is written into a path's
    // identity yet, deliberately: an always-empty channel and a contract that
    // emits nothing render identically, so the sequence is proven visible here
    // before any consumer is built on it.
    {
      std::vector<std::string> ev;
      forall_goto_program_instructions (li, goto_program)
      {
        const irep_idt nm = li->location.get("sol_emit_name");
        if (!nm.empty())
          ev.push_back(id2string(nm));
      }
      if (!ev.empty())
      {
        std::string seq;
        for (const auto &e : ev)
          seq += (seq.empty() ? "" : " -> ") + e;
        log_status(
          "--solidity-path-coverage: unit '{}' emits {} event(s), in program "
          "order: {}",
          id2string(f_it->first),
          ev.size(),
          seq);
      }
    }

    // Fold one decision into the running id. `site` is the decision's SOURCE
    // location and `sub` its operand index within that site (several folded
    // short-circuit operands share one location), so the key is content
    // addressed: it does not move when other decisions are added or when the
    // enumeration order changes. Mutates `occ` — callers copy it first for the
    // branch they push.
    // Recording the DFS side HERE, rather than at the three fan-out sites, is
    // deliberate: this lambda is the single place a decision enters a path's
    // identity, so a future fan-out that forgets to register itself cannot
    // exist — it would also have to bypass the id, which would break the
    // cross-run key first and far more loudly.
    auto step_id = [&dfs_decision_sites](
                     uint64_t idh,
                     occt &occ,
                     const std::string &site,
                     unsigned sub,
                     bool polarity) {
      dfs_decision_sites.emplace(site, sub);
      const uint64_t sk = fnv1a(site + "#" + std::to_string(sub));
      const unsigned n = occ[sk]++;
      uint64_t h = fnv1a("|", idh);
      h = fnv1a(hex64(sk), h);
      h = fnv1a(polarity ? "T" : "F", h);
      h = fnv1a(std::to_string(n), h);
      return h;
    };

    // Deferred exit asserts (insert after the walk so we don't mutate the
    // program mid-DFS). Each entry: (insertion pc, tr!=enc||cnt!=depth guard,
    // claim comment, is_revert). An is_revert path exits through a custom-error
    // `#sol_error` revert; its assert is placed right BEFORE that call
    // (upstream of the callee's ASSUME(false), which would otherwise make an
    // END_FUNCTION-placed assert vacuous -> path never covered) and gets
    // stamped `sol_revert_edge` so the Foundry generator renders
    // vm.expectRevert() (R0). Normal paths exit at END_FUNCTION,
    // is_revert=false. 5th field: the content-addressed stable path id
    // (cross-run key).
    std::vector<
      std::
        tuple<goto_programt::targett, expr2tc, std::string, bool, std::string>>
      to_insert;
    bool capped = false;
    // Set when a back-edge budget refused a continuation, i.e. this unit has a
    // path that the loop bound cut short. The exit census below downgrades its
    // hard failure to a bound obstacle when this is set, because an exit that
    // is only reachable after more than `path_cov_unwind` iterations is
    // legitimately absent from the enumeration.
    bool loop_truncated = false;
    const size_t dropped_before_unit = dropped_paths;
    // Indices into `to_insert` whose path exits via a rollback revert; resolved
    // to claim keys after the walk (the key needs the insertion location).
    std::set<size_t> rollback_exits;
    // Indices whose exit shape is ambiguous between a bare require-revert and a
    // plain early return (see the END_FUNCTION arm below).
    std::set<size_t> undetermined_exits;
    // Hard cap on DFS work so a pathological CFG can never exhaust memory.
    size_t pushes = 0;
    const size_t push_cap = 50 * path_cov_max_goals + 100000;

    // Emit one deferred exit assert for a complete path reaching `loc` with
    // path number `penc` and decision depth `pdepth`. Returns false (and sets
    // capped) when the per-function goal cap is hit, so the caller stops.
    auto emit_exit = [&](
                       goto_programt::targett loc,
                       uint64_t penc,
                       uint64_t pdepth,
                       bool is_revert,
                       uint64_t pidh) -> bool {
      if (to_insert.size() >= path_cov_max_goals)
      {
        capped = true;
        ++dropped_paths;
        return false;
      }
      // assert(tr != enc || cnt != depth): falsified only on the exact path
      // (same decision sequence AND same length), so a wrapped tr from a longer
      // path cannot fire this shorter path's assert.
      expr2tc g = or2tc(
        notequal2tc(tr, constant_int2tc(utype, BigInt(penc))),
        notequal2tc(cnt, constant_int2tc(utype, BigInt(pdepth))));
      // The comment stays the readable run-local ordinal — it is what appears
      // in the solver log and in every test.desc. The cross-run identity is the
      // separate stable id below; the two are deliberately not the same string.
      std::string comment =
        id2string(f_it->first) + ":path:" + std::to_string(penc);
      // Mix the exit site in too. The decision sequence already determines the
      // exit, so this adds nothing today; it costs nothing and means a future
      // change that lets two decision sequences share an exit cannot collide.
      const std::string stable =
        hex64(fnv1a("exit:" + loc->location.as_string(), pidh));
      // Recorded here rather than derived later: this is the only place that
      // knows the path's decision depth, and the stage-2 queries need it to
      // identify the path at all.
      path_decision_depth[{comment, loc->location.as_string()}] = pdepth;
      to_insert.emplace_back(loc, g, comment, is_revert, stable);
      return true;
    };

    // True iff `i` is the lowered call of a custom-error `revert E()` — a
    // FUNCTION_CALL to a `#sol_error` function whose body is ASSUME(false).
    // Reaching such a call means the path reverts unconditionally, so its
    // identity assert must be placed right BEFORE the call (upstream of the
    // callee's ASSUME(false)); an END_FUNCTION-placed assert would be
    // downstream and vacuous. Checking the call instruction itself (rather than
    // the incoming edge) catches EVERY revert shape — guarded `if(c) revert
    // E()`, straight-line `revert E()` as the whole body, and reverts reached
    // via an intervening unconditional GOTO — because the DFS always walks onto
    // the call instruction. require()/revert("msg") lower to a state-restoring
    // rollback with NO #sol_error call, so they are excluded and correctly fall
    // through to END_FUNCTION (try/catch). (codex: an earlier per-edge check
    // missed unguarded / goto-reached reverts; this instruction check fixes
    // it.) True iff `i` is the state-restoring assignment of a rollback revert:
    // `require(cond)` / `require(cond,"msg")` / `revert("msg")` in a function
    // with an entry snapshot lower to `*this = _sol_save_this` followed by a
    // jump to END_FUNCTION. Keying on the frontend's canonical snapshot symbol
    // name is exact — no other assignment sources it. A path that walks over
    // this instruction reverts, even though it reaches END_FUNCTION like a
    // normal exit.
    auto is_rollback_restore = [&](goto_programt::const_targett i) -> bool {
      if (!i->is_assign() || !is_code_assign2t(i->code))
        return false;
      const expr2tc &src = to_code_assign2t(i->code).source;
      return is_symbol2t(src) && to_symbol2t(src).thename.as_string().find(
                                   "_sol_save_this") != std::string::npos;
    };

    // True iff `i` is the frontend's explicit revert marker
    // `_ESBMC_sol_mark_revert()`. Under the revert-observation gate (which
    // --solidity-path-coverage turns on) EVERY require/revert failure edge
    // carries this call, including the shapes that emit no state restore at
    // all. It is the ONLY positive evidence separating a reverting exit from a
    // plain early `return`: both otherwise lower to the identical
    // `IF <guard> THEN GOTO <END_FUNCTION>`.
    auto is_revert_mark = [&](goto_programt::const_targett i) -> bool {
      if (!i->is_function_call() || !is_code_function_call2t(i->code))
        return false;
      const expr2tc &fn = to_code_function_call2t(i->code).function;
      return is_symbol2t(fn) &&
             to_symbol2t(fn).thename.as_string().find(
               "_ESBMC_sol_mark_revert") != std::string::npos;
    };

    // True iff `i` is the function EPILOGUE's restore of the enclosing-contract
    // context (`_ESBMC_enclosing_contract_address = _saved_encl_addr`). Every
    // ordinary exit of a Solidity public function walks through it; a
    // `require`-failure edge that precedes any state write is compiled as a
    // BARE jump straight to END_FUNCTION and skips it. That makes "did this
    // path pass the epilogue?" the only positive evidence available to tell an
    // ordinary exit from such a bare revert edge.
    auto is_epilogue_restore = [&](goto_programt::const_targett i) -> bool {
      if (!i->is_assign() || !is_code_assign2t(i->code))
        return false;
      // An expanded callee brings its OWN epilogue along, and it matches the
      // same symbol name. Counting it would let a path that ran the callee's
      // epilogue and then took the caller's bare revert edge look like an
      // ordinary exit. Only THIS unit's epilogue is evidence about this unit's
      // exit, so instructions flagged by the expansion are not evidence.
      if (i->location.get_bool("sol_path_inlined"))
        return false;
      const expr2tc &src = to_code_assign2t(i->code).source;
      return is_symbol2t(src) && to_symbol2t(src).thename.as_string().find(
                                   "_saved_encl_addr") != std::string::npos;
    };

    // Does this function HAVE an epilogue at all? Without one the marker above
    // carries no information, so every path would look "bypassed" and be
    // reported undetermined. Only apply the test where it is meaningful.
    bool has_epilogue = false;
    forall_goto_program_instructions (eit, goto_program)
      if (is_epilogue_restore(eit))
      {
        has_epilogue = true;
        break;
      }

    auto is_error_call = [&](goto_programt::const_targett i) -> bool {
      if (!i->is_function_call() || !is_code_function_call2t(i->code))
        return false;
      const expr2tc &fn = to_code_function_call2t(i->code).function;
      if (!is_symbol2t(fn))
        return false;
      const symbolt *s = ns.lookup(to_symbol2t(fn).thename);
      return s && !s->get_type().get("#sol_error").as_string().empty();
    };

    // "A rolled-back execution never reaches a RETURN" — MEASURED AND REFUTED.
    //
    // Had it held, reaching a RETURN would have been POSITIVE evidence of a
    // normal exit, independent of the revert-observation gate and of which
    // scopes that gate covers. It does not hold, and the reason is structural
    // rather than a corner case: when the enclosing function returns a value,
    // the frontend lowers a failing `require` to
    //     { *this = _sol_save_this; return [nondet]; }
    // so the reverting execution ends at a RETURN of the frontend's own making.
    // Refuted on the simplest shape tried (compute a value, mutate state, then
    // fail a require) and again on a modifier's require — the latter had been
    // predicted to be a positive case, and is one only when the function
    // returns nothing.
    //
    // Consequence, recorded so it is not quietly re-derived: the inference
    // "ends at a RETURN, therefore normal" is NOT available. The counter below
    // stays as an internal consistency check — such a path must always be
    // classified as a rollback revert, never as normal — and is expected to be
    // nonzero on ordinary contracts, so it logs at debug level rather than
    // warning.
    size_t rolled_back_return_exits = 0;
    std::vector<std::string> rolled_back_return_locs;

    // ---- WHY an exit came out `undetermined` ----
    //
    // `undetermined` means "no positive evidence of a normal exit was found".
    // Since the frontend contract landed, every regression reports zero of
    // them, so a nonzero count is now a hard signal rather than background
    // noise — and a bare count says only THAT evidence is missing, never WHICH
    // KIND. There are exactly three ways to get here, each missing a different
    // witness, and they call for different fixes:
    //
    //   (1) the unit has NO epilogue at all, so the epilogue marker carries no
    //       information for any of its paths (library / free-function scope);
    //   (2) the unit HAS an epilogue and this path reached END_FUNCTION without
    //       walking it — the bare-jump shape a pre-state-write `require`
    //       failure and a plain early `return` share;
    //   (3) the path ends at a RETURN carrying no `sol_source_return` marker
    //   and
    //       walked no epilogue — either a return shape the frontend does not
    //       stamp, or a RETURN the frontend synthesised for something else.
    //
    // Grouped per unit and by cause rather than listed per path: the lesson
    // from chasing these one at a time is that two different defects can sit on
    // the same function, and only a grouping shows which witness the whole
    // group is missing.
    size_t und_no_epilogue = 0;
    size_t und_epilogue_skipped = 0;
    size_t und_return_unmarked = 0;
    std::set<std::string> und_locs_no_epilogue;
    std::set<std::string> und_locs_epilogue_skipped;
    std::set<std::string> und_locs_return_unmarked;

    // Classify the exit just appended to `to_insert`. Shared by the
    // END_FUNCTION arm and the RETURN arm so the two cannot drift apart.
    //
    // `src_return` is the frontend's POSITIVE marker (`sol_source_return`),
    // stamped on RETURNs lowered from a source-level `return` and NOT on the
    // one the frontend synthesises for a failing `require`
    // (`{ *this = _sol_save_this; return [nondet]; }`). It is the second piece
    // of positive evidence, alongside the epilogue: the epilogue cannot testify
    // for a returning path because it is emitted after the RETURN, which is why
    // every value-returning unit's normal exit used to be `undetermined`.
    //
    // The rollback test still runs FIRST and still wins. A source `return` can
    // sit on a path that already walked a revert marker or a state restore, and
    // on such a path the transaction reverts whatever the return statement
    // says. Only a path with no rollback evidence AND an affirmative
    // source-return marker is called normal — that is still a positive
    // inference, not "nothing said revert, so it must be fine".
    // (`se` can only be true when the function HAS an epilogue, since it is set
    // by walking that very instruction — so the previous `!has_epilogue || !se`
    // is exactly `!se`, and dropping the redundant term changes nothing.)
    auto classify_exit = [&](
                           size_t idx,
                           bool rb,
                           bool se,
                           bool src_return,
                           const std::string &site) {
      if (rb)
        rollback_exits.insert(idx);
      else if (!se && !src_return)
      {
        undetermined_exits.insert(idx);
        ++und_return_unmarked;
        und_locs_return_unmarked.insert(site);
      }
    };

    while (!stack.empty())
    {
      auto [pc, enc, becnt, depth, rolled_back, saw_epilogue, idh, occ] =
        stack.back();
      stack.pop_back();

      while (true)
      {
        if (pc == goto_program.instructions.end() || pc->is_end_function())
        {
          if (pc != goto_program.instructions.end())
          {
            if (!emit_exit(pc, enc, depth, false, idh))
              break;
            const size_t idx = to_insert.size() - 1;
            // R0 event rung: this COMPLETE path's emit sequence, recovered
            // from the prefixes it walked. ALSO published, as the report's
            // per-claim `events` array (bmc.cpp), which is what a generator
            // reads; this line stays because it is the only view of the
            // sequence for a run that dies before the report is written, and
            // because a producer-side print that disagrees with the report
            // would expose a publication bug that no consumer could see.
            //
            // Printed only when non-empty, unlike the report field: stdout is
            // per path and a line saying "this path emitted nothing" on every
            // path of every unit is noise. The distinction the empty array
            // carries — recorded-and-empty vs never-recorded — lives in the
            // report, where a consumer can act on it.
            {
              const auto evs = events_for(enc, depth);
              if (!evs.empty())
              {
                std::string seq;
                for (const auto &e : evs)
                  seq += (seq.empty() ? "" : " -> ") + e;
                log_status(
                  "--solidity-path-coverage: path enc={} depth={} emits {} "
                  "event(s): {}",
                  enc,
                  depth,
                  evs.size(),
                  seq);
              }
            }
            // An END_FUNCTION exit is never itself a source `return`: a bare
            // `return;` lowers to a JUMP to END_FUNCTION, so the marker sits on
            // the jump, not here. Such a path in a function WITH an epilogue
            // already walks it and is normal on that evidence.
            if (rolled_back)
              // Positive evidence of a rollback revert.
              rollback_exits.insert(idx);
            else if (!has_epilogue || !saw_epilogue)
            // No positive evidence of a normal exit. Either the path reached
            // END_FUNCTION while SKIPPING the epilogue, or the function has
            // no epilogue at all (library / free function — exactly the
            // scopes the revert-observation gate does NOT mark, so a revert
            // there carries no marker either). Both a `require` failing
            // before any state write and a plain early `return` compile to
            // this same shape, with nothing on the edge to separate them.
            // Report undetermined rather than guess: calling it "normal"
            // would claim a reverted transaction succeeded — measured on
            // a library whose function reverts, that is exactly what the
            // previous "no epilogue => normal" default did.
            //
            // The two sub-cases are counted apart (see the declaration): "the
            // unit has no epilogue at all" is a scope problem affecting every
            // path of the unit, while "the epilogue exists and this path
            // skipped it" is a per-path shape. Fixing one does nothing for the
            // other.
            {
              undetermined_exits.insert(idx);
              if (has_epilogue)
              {
                ++und_epilogue_skipped;
                und_locs_epilogue_skipped.insert(pc->location.as_string());
              }
              else
              {
                ++und_no_epilogue;
                und_locs_no_epilogue.insert(pc->location.as_string());
              }
            }
          }
          break;
        }
        if (is_rollback_restore(pc) || is_revert_mark(pc))
          rolled_back = true;
        if (is_epilogue_restore(pc))
          saw_epilogue = true;
        // R0 event rung: record this emit against the prefix in effect, keyed
        // by program position so a re-walk overwrites rather than appends.
        {
          const irep_idt ev_nm = pc->location.get("sol_emit_name");
          if (!ev_nm.empty())
          {
            auto ins =
              ev_intern.emplace(id2string(ev_nm), (uint32_t)ev_table.size());
            if (ins.second)
              ev_table.push_back(id2string(ev_nm));
            ev_index[enc][ev_pos[&*pc]] = ins.first->second;
          }
        }
        // Custom-error revert exit: the DFS reached the `#sol_error` call
        // (guarded, straight-line, or via an unconditional GOTO). Emit the
        // identity assert HERE (upstream of the callee's ASSUME(false), so it
        // is reachable) and stop; flag it for vm.expectRevert() (R0).
        if (is_error_call(pc))
        {
          if (!emit_exit(pc, enc, depth, true, idh))
            break;
          break;
        }
        if (pc->is_goto())
        {
          const bool back = pc->is_backwards_goto();
          if (is_true(pc->guard))
          {
            // Unconditional goto. A backward one is one iteration of the loop
            // whose head is the goto's target; bound that loop independently.
            if (back)
            {
              const unsigned key = pc->get_target()->target_number;
              if (becnt[key] >= path_cov_unwind)
              {
                loop_truncated = true;
                break; // this loop's bound reached: path truncated
              }
              ++becnt[key];
            }
            pc = pc->get_target();
            continue;
          }
          // Conditional. Keep enc within 64 bits (leading sentinel + one bit
          // per decision on the path); drop over-long paths rather than alias.
          if (enc >= (uint64_t(1) << 62))
          {
            ++dropped_paths;
            break;
          }
          // guard-true/taken successor -> target. If that edge is a loop
          // back-edge, its own loop is bounded by path_cov_unwind.
          bool take = true;
          becntt becnt_taken = becnt;
          if (back)
          {
            const unsigned key = pc->get_target()->target_number;
            if (becnt_taken[key] >= path_cov_unwind)
            {
              take = false;
              loop_truncated = true;
            }
            else
              ++becnt_taken[key];
          }
          // Push the guard-true/taken successor; the guard-false/fall-through
          // continues in-place. A reverting successor (custom-error revert) is
          // detected at the top of the loop when the DFS reaches the
          // `#sol_error` call instruction, so no per-edge revert test is
          // needed.
          const std::string dsite = pc->location.as_string();
          if (take)
          {
            if (++pushes > push_cap)
            {
              capped = true;
              ++dropped_paths;
              break;
            }
            // Both arms consume the SAME occurrence of this site, so each
            // starts from its own copy of the counters.
            occt occ_taken = occ;
            const uint64_t idh_taken =
              step_id(idh, occ_taken, dsite, 0, /*polarity=*/true);
            note_decision(enc * 2 + 1, pc->location, pc->guard, 0);
            stack.push_back(
              {pc->get_target(),
               enc * 2 + 1,
               becnt_taken,
               depth + 1,
               rolled_back,
               saw_epilogue,
               idh_taken,
               occ_taken});
          }
          // guard-false/fallthrough successor -> next (never a back-edge).
          idh = step_id(idh, occ, dsite, 0, /*polarity=*/false);
          note_decision(enc * 2 + 0, pc->location, pc->guard, 0);
          enc = enc * 2 + 0;
          ++depth;
          pc = std::next(pc);
          continue;
        }
        // A RETURN is a path EXIT, not a straight-line instruction.
        //
        // symex terminates the frame at RETURN — it does not fall through to
        // END_FUNCTION — so an identity assert placed at END_FUNCTION sits
        // downstream of the frame exit and can never execute. Measured on four
        // variants that isolate the cause: a unit that writes state and returns
        // nothing covers 3/3, while the SAME body with a return value covers
        // 1/3, and the one path that stays coverable is exactly the ABI
        // value-reject path — the only one that reaches END_FUNCTION by a plain
        // GOTO instead of through the RETURN. Purity, and whether an explicit
        // `return` statement is written at all (a named return variable behaves
        // identically), make no difference.
        //
        // Effect before this: EVERY unit with a return value had all of its
        // body paths reported U — a systematic, silent deflation of coverage on
        // any real contract, since getters and view functions all return
        // values.
        //
        // Emitting here places the assert immediately BEFORE the RETURN:
        // downstream of this path's tr/cnt updates (Phase 1 inserted those
        // before the RETURN earlier, so they end up further from it) and
        // upstream of the frame exit.
        if (pc->is_return() && is_code_return2t(pc->code))
        {
          const expr2tc &rsrc = to_code_return2t(pc->code).operand;
          // Collected, not merely counted: the operand EXPRESSION is what the
          // branch metric keys its claim on, so recording the decision needs
          // the operand itself. Collect order is the bit order (see the DFS
          // below).
          std::vector<expr2tc> rops;
          if (pc->location.property().as_string() != "skipped")
            collect_short_circuit_decisions(
              rsrc, [&](const expr2tc &e) { rops.push_back(e); });
          const size_t RK = rops.size();
          const std::string rsite = pc->location.as_string();
          // The frontend's positive normal-exit marker (see classify_exit).
          // Read from the RETURN instruction itself, so a synthesised
          // rollback RETURN — which carries no marker — cannot borrow it.
          const bool src_return = pc->location.get_bool("sol_source_return");
          if (RK > 0 && RK <= SC_DECISION_MAX)
          {
            // Folded short-circuit operands in the returned expression are
            // decisions like any other; each combination is its own exit.
            for (uint64_t mask = 0; mask < (uint64_t(1) << RK); ++mask)
            {
              uint64_t e = enc, d = depth, h = idh;
              occt o = occ;
              bool overflowed = false;
              for (size_t j = 0; j < RK; ++j)
              {
                if (e >= (uint64_t(1) << 62))
                {
                  overflowed = true;
                  break;
                }
                const bool bit = ((mask >> j) & 1) != 0;
                e = e * 2 + (bit ? 1 : 0);
                h = step_id(h, o, rsite, (unsigned)j, bit);
                note_decision(e, pc->location, rops[j], (unsigned)j);
                ++d;
              }
              if (overflowed)
              {
                ++dropped_paths;
                break;
              }
              if (!emit_exit(pc, e, d, false, h))
                break;
              classify_exit(
                to_insert.size() - 1,
                rolled_back,
                saw_epilogue,
                src_return,
                rsite);
              if (rolled_back)
              {
                ++rolled_back_return_exits;
                rolled_back_return_locs.push_back(pc->location.as_string());
              }
            }
          }
          else if (emit_exit(pc, enc, depth, false, idh))
          {
            classify_exit(
              to_insert.size() - 1,
              rolled_back,
              saw_epilogue,
              src_return,
              rsite);
            if (rolled_back)
            {
              ++rolled_back_return_exits;
              rolled_back_return_locs.push_back(pc->location.as_string());
            }
          }
          break;
        }
        // Folded short-circuit/ternary operands (no control-flow branch):
        // each was snapshotted into tr in Phase 1. Fan the DFS out over the
        // 2^K operand-value combinations, appending K bits to enc/depth (in
        // collect order, matching tr) so each combination is a distinct path.
        if (pc->is_assign() && pc->location.property().as_string() != "skipped")
        {
          const expr2tc &src = to_code_assign2t(pc->code).source;
          // Collected rather than counted, for the same reason as the RETURN
          // arm above: the recorder needs the operand expression.
          std::vector<expr2tc> aops;
          collect_short_circuit_decisions(
            src, [&](const expr2tc &e) { aops.push_back(e); });
          const size_t K = aops.size();
          // Cap MUST match Phase 1's (see SC_DECISION_MAX): a site Phase 1
          // skipped contributes nothing to tr/cnt, so the DFS must not add bits
          // for it either — and vice versa.
          if (K > 0 && K <= SC_DECISION_MAX)
          {
            bool overflowed = false;
            const std::string asite = pc->location.as_string();
            for (uint64_t mask = 0; mask < (uint64_t(1) << K); ++mask)
            {
              uint64_t e = enc, d = depth, h = idh;
              occt o = occ;
              for (size_t j = 0; j < K; ++j)
              {
                if (e >= (uint64_t(1) << 62))
                {
                  overflowed = true;
                  break;
                }
                const bool bit = ((mask >> j) & 1) != 0;
                e = e * 2 + (bit ? 1 : 0);
                h = step_id(h, o, asite, (unsigned)j, bit);
                note_decision(e, pc->location, aops[j], (unsigned)j);
                ++d;
              }
              if (overflowed)
              {
                ++dropped_paths;
                break;
              }
              if (++pushes > push_cap)
              {
                capped = true;
                ++dropped_paths;
                break;
              }
              stack.push_back(
                {std::next(pc), e, becnt, d, rolled_back, saw_epilogue, h, o});
            }
            break; // this path forked into the 2^K continuations
          }
        }
        pc = std::next(pc); // straight-line
      }
      if (capped)
        break;
    }

    // Did this unit lose paths to the per-unit goal cap / length cap?
    //
    // Truncation WEAKENS the assertions this unit can support. It does NOT make
    // them wrong, and the distinction is mechanical rather than a judgement
    // call:
    //
    //   the certification query is `assume(L <= x <= U); assert(tr == pi)`.
    //   The goal cap limits how many EXIT ASSERTS are emitted; it does not
    //   touch the Phase-1 accounting, which still updates `tr`/`cnt` at every
    //   decision of every path. So an input that walks a DROPPED path pi''
    //   still carries pi'''s number in `tr` at the exit, the query `tr == pi`
    //   fails on it, and the candidate interval is rejected and shrunk.
    //   Certification therefore never needs the dropped path to have been
    //   enumerated — it only needs the accounting, which is intact.
    //
    // That is what separates this from a `require` lowered to a
    // control-flow-free assume, which stays a named obstacle: there the
    // reverting execution does not exist in the model AT ALL, so no query can
    // see it and no interval can be shrunk away from it.
    // Existing-but-unenumerated and non-existent are not the same failure, and
    // only the second can ship a test that is red on the unmodified contract.
    //
    // For completeness, the other two ways paths go missing:
    //   * a call left unexpanded (depth bound, or a withdrawal by degradation)
    //     COARSENS the recorded decision sequence. The classes still partition
    //     the input space, so this is sound and weaker.
    //   * a loop cut at the unwind bound is cut IN BOTH PLACES — symex assumes
    //     away the same over-bound iterations. Inside the declared bound the
    //     domain is complete; outside it, the model and the real chain diverge
    //     and the certification query cannot see that either. That is a
    //     declared bound of the method rather than a defect, so it belongs in
    //     the proposition's wording, not here.
    //
    // So this is reported as an absolute count and a strength annotation, and
    // degradation exists precisely so it should not be reached at all.
    const bool unit_truncated = capped || dropped_paths > dropped_before_unit;

    // ---- Path-count distribution measurement ----
    {
      const std::string uname = f_it->first.as_string();
      const size_t after = to_insert.size();
      if (after > max_unit_paths)
      {
        max_unit_paths = after;
        max_unit_name = uname;
      }
      if (unit_truncated)
      {
        ++units_at_cap;
        // ORDERING INVARIANT, reported rather than assumed: degradation runs
        // first and is supposed to bring every unit inside the budget, so in
        // the intended steady state the goal cap never fires. That it fired is
        // therefore a result in its own right, and the two ways it can happen
        // say different things — hence they are distinguished here instead of
        // being lumped into one "capped" count.
        auto bs = budget_state.find(uname);
        const budget_statet st =
          bs == budget_state.end() ? budget_statet::fits : bs->second;
        switch (st)
        {
        case budget_statet::degraded_over:
          log_warning(
            "--solidity-path-coverage: unit '{}' hit the goal cap ({}) after "
            "degradation withdrew every one of its {} withdrawable call "
            "point(s) and it was STILL over budget. Expected, not a policy "
            "failure: the unit's own source decisions exceed the budget, so "
            "there was nothing left to give up",
            uname,
            path_cov_max_goals,
            degraded_call_sites[uname].size());
          break;
        case budget_statet::no_candidates:
          log_warning(
            "--solidity-path-coverage: unit '{}' hit the goal cap ({}) with no "
            "withdrawable call point to degrade (warned above). Expected, not "
            "a policy failure",
            uname,
            path_cov_max_goals);
          break;
        case budget_statet::degraded_fits:
          // NOT a defect, and deliberately not worded as one. Degradation
          // stopped because the flat counter said the unit fits; the cap then
          // fired on a count produced by the enumerating DFS. Two different
          // computations of "how many paths" — that is the point of keeping
          // them separate (a DFS bug cannot hide in both) and the price is that
          // their COUNTING UNITS can differ. Both numbers are printed so the
          // question "is this the same quantity?" can be answered by looking,
          // rather than by opening an investigation into a bug that may not
          // exist.
          log_warning(
            "--solidity-path-coverage: unit '{}' hit the goal cap ({}) after "
            "degradation withdrew {} call point(s) and stopped, because the "
            "pre-enumeration count then read {} (within budget). The "
            "enumeration produced {}. These are two different computations, so "
            "a gap here means their counting units disagree — reconcile the "
            "two "
            "definitions; it is not by itself a defect. If they DO agree, the "
            "degradation policy simply stopped one withdrawal too early",
            uname,
            path_cov_max_goals,
            degraded_call_sites[uname].size(),
            estimated_paths[uname],
            after);
          break;
        case budget_statet::fits:
          log_warning(
            "--solidity-path-coverage: unit '{}' hit the goal cap ({}) but its "
            "pre-enumeration count was {}, inside the budget, so degradation "
            "was never offered the unit — while the enumeration produced {}. "
            "This is the one case that IS a defect: the number the budget "
            "decision was taken on is not the number the enumeration produces, "
            "and nothing in the pipeline noticed",
            uname,
            path_cov_max_goals,
            estimated_paths[uname],
            after);
          break;
        }
      }

      size_t before = 0;
      auto snap = pre_inline_body.find(uname);
      if (snap != pre_inline_body.end())
      {
        bool snap_capped = false;
        before = count_paths_no_instrument(
          snap->second, ns, path_cov_unwind, path_cov_max_goals, snap_capped);
        pre_expansion_total += before;

        // The measurement is itself measured. When nothing was expanded into
        // this unit, the counter and the real enumeration are looking at the
        // same program and MUST agree — after discounting the ABI gate, which
        // is inserted after the snapshot and contributes exactly one path. A
        // mismatch means the two traversals have drifted, which would silently
        // corrupt every ratio computed from them.
        const size_t after_no_gate =
          after - (gate_inserted && after > 0 ? 1 : 0);
        if (
          expanded_into_unit[uname] == 0 && !capped && !snap_capped &&
          !loop_truncated && before != after_no_gate)
        {
          log_error(
            "--solidity-path-coverage: INTERNAL DEFECT: the path counter and "
            "the path enumeration disagree on unit '{}' ({} vs {}) even though "
            "nothing was expanded into it and no bound was hit. The two "
            "traversals have drifted, so every expansion ratio derived from "
            "them would be wrong.",
            uname,
            before,
            after_no_gate);
          abort();
        }
      }
      log_debug(
        "coverage",
        "unit '{}': {} path(s) after expansion, {} before, {} call(s) expanded",
        uname,
        after,
        before,
        expanded_into_unit[uname]);
    }

    // ---- Report WHY this unit has undetermined exits (grouped by cause) ----
    if (!undetermined_exits.empty())
    {
      // The three causes must account for every undetermined exit. This is a
      // "two counts agree" property, independent of how any exit is classified,
      // so it stays true whatever the frontend does next — unlike an assertion
      // that forbids a particular outcome, which pins the symptom of whatever
      // was broken when it was written.
      const size_t summed =
        und_no_epilogue + und_epilogue_skipped + und_return_unmarked;
      if (summed != undetermined_exits.size())
      {
        log_error(
          "--solidity-path-coverage: INTERNAL DEFECT in unit '{}': {} exit(s) "
          "were classified undetermined but only {} of them were attributed to "
          "a cause. An undetermined exit with no recorded cause means a fourth "
          "route into the class exists and the breakdown below is silently "
          "incomplete.",
          id2string(f_it->first),
          undetermined_exits.size(),
          summed);
        abort();
      }
      // An END_FUNCTION instruction carries no source location, so its set is
      // legitimately empty. Say that, rather than printing `[]` — an empty
      // bracket reads like "the locations were lost", which would send someone
      // looking for a bug in the reporting instead of at the named unit.
      auto join = [](const std::set<std::string> &s) {
        std::string out;
        for (const auto &e : s)
          out += (out.empty() ? "" : "; ") + e;
        return out.empty() ? std::string(
                               "no source location on the exit "
                               "instruction; see the unit named above")
                           : out;
      };
      log_warning(
        "--solidity-path-coverage: unit '{}' has {} exit(s) with NO positive "
        "evidence of a normal exit, by cause: "
        "(1) {} at END_FUNCTION in a unit with NO epilogue at all [{}]; "
        "(2) {} at END_FUNCTION having SKIPPED this unit's epilogue [{}]; "
        "(3) {} at a RETURN carrying no `sol_source_return` marker [{}]. "
        "Each cause is missing a DIFFERENT witness and needs a different fix; "
        "an undetermined exit cannot become an oracle, so these are the paths "
        "R0 cannot serve",
        id2string(f_it->first),
        undetermined_exits.size(),
        und_no_epilogue,
        join(und_locs_no_epilogue),
        und_epilogue_skipped,
        join(und_locs_epilogue_skipped),
        und_return_unmarked,
        join(und_locs_return_unmarked));
    }

    if (rolled_back_return_exits > 0)
    {
      std::string where;
      for (const auto &l : rolled_back_return_locs)
        where += (where.empty() ? "" : "; ") + l;
      // Every such path must have been classified as a rollback revert. If one
      // ever escapes that classification it would be reported as an ordinary
      // exit of a transaction that in fact reverted — the single wrong answer
      // this classifier must never give.
      //
      // Stated as a COUNT, not as "no RETURN exit may be ordinary". The older
      // form asserted the latter, which was equivalent only while a RETURN
      // could never be classified normal at all: the epilogue that would have
      // proved it is emitted after the RETURN. Now that the frontend marks its
      // source-level returns, a RETURN exit CAN legitimately be normal, and the
      // old form fired on the first correct classification — it was pinning an
      // artefact of the missing evidence, not the property it names.
      size_t rb_return_exits_classified = 0;
      for (size_t i = 0; i < to_insert.size(); ++i)
        if (std::get<0>(to_insert[i])->is_return() && rollback_exits.count(i))
          ++rb_return_exits_classified;
      if (rb_return_exits_classified != rolled_back_return_exits)
      {
        log_error(
          "--solidity-path-coverage: INTERNAL DEFECT in unit '{}': {} path(s) "
          "walked a rollback/revert marker and ended at a RETURN ({}), but "
          "only {} of them were classified as reverting exits. The rest would "
          "be reported as ordinary exits of transactions that in fact "
          "reverted.",
          id2string(f_it->first),
          rolled_back_return_exits,
          where,
          rb_return_exits_classified);
        abort();
      }
      log_debug(
        "coverage",
        "unit '{}': {} path(s) end at a RETURN after a rollback/revert marker "
        "({}) — expected: the frontend lowers a failing require in a "
        "value-returning function to a restore-and-return block",
        id2string(f_it->first),
        rolled_back_return_exits,
        where);
    }

    // ---- `tr` completeness check (see the declaration above) ----
    //
    // Runs BEFORE the exit census so that, if both would fire, the reader sees
    // the accumulator problem first: an unaccounted decision makes every path
    // through it uncoverable, which then also shows up as a missing exit. Order
    // the diagnosis from cause to symptom.
    {
      std::vector<std::string> unaccounted;
      for (const auto &[site, sub] : dfs_decision_sites)
        if (phase1_decision_sites.count({site, sub}) == 0)
          unaccounted.push_back(
            site + " (operand " + std::to_string(sub) + ")");
      if (!unaccounted.empty())
      {
        std::string where;
        for (const auto &u : unaccounted)
          where += (where.empty() ? "" : "; ") + u;
        log_error(
          "--solidity-path-coverage: INTERNAL DEFECT in unit '{}'. {} decision "
          "site(s) are branched on by the path enumeration but are NOT "
          "accumulated into `tr` at run time: {}. Every real execution then "
          "carries fewer decisions than the emitted path expects, so "
          "`cnt != depth` holds always and those paths are permanently "
          "uncoverable — while being reported as PASSED, i.e. a false proof of "
          "unreachability. This also breaks the stage-3 certification query "
          "`assume(interval); assert(tr == pi)`, which is sound only because "
          "`tr` records the complete decision sequence of whatever actually "
          "executes.",
          id2string(f_it->first),
          unaccounted.size(),
          where);
        abort();
      }
      // The reverse direction is dead code, not a defect: a decision that was
      // snapshotted but that no path ever reaches. Reported at debug level so
      // it is available when a count looks odd, without adding noise.
      size_t never_traversed = 0;
      for (const auto &s : phase1_decision_sites)
        if (dfs_decision_sites.count(s) == 0)
          ++never_traversed;
      log_debug(
        "coverage",
        "unit '{}': {} decision site(s) accumulated into tr, {} of them never "
        "traversed by any enumerated path (unreachable code)",
        id2string(f_it->first),
        phase1_decision_sites.size(),
        never_traversed);
    }

    // ---- Exit census: does the enumeration account for every exit? ----
    //
    // Motivation: the failure mode this guards against is a whole CLASS of exit
    // being silently swallowed. That already happened once — asserts were
    // placed at END_FUNCTION while a RETURN terminates the frame, so every path
    // of every value-returning unit became uncoverable and was reported U.
    // Nothing crashed and nothing warned; the coverage number simply read low,
    // and "U" is indistinguishable from an honest solver timeout. A number that
    // can absorb an implementation defect is worse than a number that is
    // missing.
    //
    // The census is deliberately built from a DIFFERENT computation than the
    // enumeration: a flat forward reachability scan keyed on INSTRUCTION KIND,
    // sharing none of the DFS's path/enc/depth bookkeeping. A bug in the DFS
    // therefore cannot hide in both.
    //
    // Known limitation, stated rather than papered over: this is a goto-level
    // census, not an AST-level one. It cannot see a source-level exit that the
    // frontend dropped before goto conversion — for that the census would have
    // to count `return` / `revert` / `require` sites in the Solidity AST and be
    // plumbed through from the frontend. This catches everything the goto
    // program still contains, which is where every defect so far has lived.
    {
      // Terminators, from measurement: a RETURN ends the frame (symex does not
      // fall through to END_FUNCTION), and a `#sol_error` callee is nothing but
      // ASSUME(false).
      auto is_exit_kind = [&](goto_programt::const_targett i) {
        return i->is_return() || i->is_end_function() || is_error_call(i);
      };

      std::set<const goto_programt::instructiont *> reachable_exits;
      std::set<const goto_programt::instructiont *> seen;
      std::vector<goto_programt::targett> work;
      if (!goto_program.instructions.empty())
        work.push_back(goto_program.instructions.begin());
      while (!work.empty())
      {
        auto i = work.back();
        work.pop_back();
        if (i == goto_program.instructions.end())
          continue;
        if (!seen.insert(&*i).second)
          continue;
        if (is_exit_kind(i))
        {
          reachable_exits.insert(&*i);
          continue; // terminator: no successors
        }
        if (i->is_goto())
        {
          work.push_back(i->get_target());
          if (!is_true(i->guard))
            work.push_back(std::next(i));
          continue;
        }
        work.push_back(std::next(i));
      }

      std::set<const goto_programt::instructiont *> enumerated_exits;
      for (const auto &e : to_insert)
        enumerated_exits.insert(&*std::get<0>(e));

      // AST-level half of the census. The goto scan above can only see exits
      // the goto program still contains; a source-level exit dropped before
      // goto conversion leaves no trace there. The frontend records the number
      // of `return` statements it actually saw in the source
      // (`#sol_ast_return_sites`), which is the only independent witness that
      // such an exit existed. If the source has returns and the enumeration
      // ends no path at a RETURN, a whole class of exit has gone missing
      // between the AST and here.
      {
        const symbolt *usym = ns.lookup(f_it->first);
        const std::string ast_rets =
          usym ? usym->get_type().get("#sol_ast_return_sites").as_string()
               : std::string();
        if (!ast_rets.empty() && std::stoul(ast_rets) > 0)
        {
          bool any_return_exit = false;
          for (const auto &e : to_insert)
            if (std::get<0>(e)->is_return())
            {
              any_return_exit = true;
              break;
            }
          if (!any_return_exit && !loop_truncated && !capped)
          {
            log_error(
              "--solidity-path-coverage: INTERNAL DEFECT in unit '{}'. The "
              "source contains {} value-returning `return` statement(s), but "
              "no enumerated path exits at a RETURN and no bound was hit. A "
              "class of exit "
              "has been lost between the AST and the enumeration, so coverage "
              "and every U verdict for this unit would be wrong.",
              id2string(f_it->first),
              ast_rets);
            abort();
          }
        }

        // Named obstacle: `this.f(...)`. On a real EVM this re-enters through
        // the ABI, so `msg.sender` inside the callee becomes the contract's own
        // address; the frontend lowers it to a plain direct call, which keeps
        // the original caller's msg.sender. The model can therefore admit a
        // path that reverts on-chain (`require(msg.sender == owner)` being the
        // canonical case). Declaring it here means a downstream emitter can
        // refuse to emit a test for this unit and count the refusal, instead of
        // emitting one that is labelled certified and fails when run.
        const std::string tc =
          usym ? usym->get_type().get("#sol_this_call_count").as_string()
               : std::string();
        if (!tc.empty())
          log_warning(
            "--solidity-path-coverage: unit '{}' contains {} `this.<f>(...)` "
            "call site(s) [{}]. On-chain that is an EXTERNAL call and "
            "msg.sender "
            "inside the callee is this contract's own address; the model "
            "lowers "
            "it to a direct call and keeps the caller's msg.sender, so a path "
            "through it may not exist on-chain. NAMED OBSTACLE: paths through "
            "these sites must not be turned into tests",
            id2string(f_it->first),
            tc,
            usym->get_type().get("#sol_this_call_names").as_string());
      }

      std::vector<std::string> unaccounted;
      for (const auto *ex : reachable_exits)
        if (enumerated_exits.count(ex) == 0)
          unaccounted.push_back(ex->location.as_string());

      if (!unaccounted.empty())
      {
        std::string where;
        for (const auto &u : unaccounted)
          where += (where.empty() ? "" : "; ") + u;
        const bool bounded_out =
          loop_truncated || capped || dropped_paths > dropped_before_unit;
        if (bounded_out)
          // A legitimate budget effect, not a defect: an exit that only becomes
          // reachable past the loop bound or past a goal/length cap. Named, so
          // it can never be mistaken for full accounting.
          log_warning(
            "--solidity-path-coverage: unit '{}' has {} reachable exit(s) that "
            "no enumerated path ends at, because this unit hit a bound (loop "
            "bound {} / goal cap {}). Reported as a bound obstacle, not as "
            "coverage. Exits: {}",
            id2string(f_it->first),
            unaccounted.size(),
            path_cov_unwind,
            path_cov_max_goals,
            where);
        else
        {
          // No bound was hit, so every reachable exit MUST be the exit of some
          // enumerated path. It is not: the enumeration is dropping a class of
          // exit on the floor. Aborting is the point — the alternative is a
          // coverage percentage that silently omits it.
          log_error(
            "--solidity-path-coverage: INTERNAL DEFECT in unit '{}'. {} "
            "exit(s) "
            "are reachable in the goto program but no enumerated path ends at "
            "them, and no bound was hit to explain it. The enumeration is "
            "swallowing a class of exit, so the coverage denominator and every "
            "U verdict derived from it would be wrong. Exits: {}",
            id2string(f_it->first),
            unaccounted.size(),
            where);
          abort();
        }
      }
    }

    // ---- PUBLISH THE DECISION SEQUENCES ----
    //
    // Placed HERE, before the outer-box and certify branches, because both of
    // those `continue` out of the loop body — publishing after either one would
    // leave the tables empty in exactly the modes that ask for them, with no
    // symptom other than a report field quietly missing.
    //
    // Keyed by unit id because `enc` values COLLIDE ACROSS UNITS (enc is a
    // per-unit accumulator seeded at 1). A single flat map would silently serve
    // one unit's decision for another unit's path — a wrong site, not a missing
    // one, and therefore invisible.
    if (record_decisions)
    {
      const std::string uid_pub = f_it->first.as_string();
      path_decision_table[uid_pub] = std::move(dec_table);
      path_decision_index[uid_pub] = std::move(dec_index);
      // Published UNCONDITIONALLY, including when this unit emitted nothing.
      // An absent key and an empty table are the same thing to a consumer that
      // only ever asks "is there an entry", but they are NOT the same fact: the
      // first says the recorder never ran on this unit, the second says it ran
      // and saw no emit. The report distinguishes them by always writing the
      // array once the unit is here, so "no field" can only mean "recording was
      // off", never "no events".
      path_event_table[uid_pub] = std::move(ev_table);
      path_event_index[uid_pub] = std::move(ev_index);
    }

    // ---- STAGE 2, step 1: the outer-box batch ----
    //
    // Runs in the same place, and for the same reason, as the certification
    // query below: expansion, the ABI gate, Phase-1 accounting and all three
    // censuses have already run, and the `tr`/`cnt` read here are the same ones
    // the enumeration writes.
    if (outer_on)
    {
      const std::string uid = f_it->first.as_string();
      if (
        uid != outer_unit &&
        uid.find("@F@" + outer_unit + "#") == std::string::npos)
        continue;

      // ---- S2: record, do NOT refuse ----
      //
      // The asymmetry with certify is deliberate and is the same one the
      // refused-coordinate handling already draws. An outer box is a
      // CONTAINMENT statement per coordinate, and it stays true on an
      // obstructed unit; certification is an assertion ABOUT the box and can
      // come back SUCCESSFUL over executions the chain does not have. So this
      // side measures and labels, and the certify side refuses.
      //
      // Read from the per-unit LOCALS. `named_obstacle_paths` is filled by the
      // insertion loop this branch `continue`s past, so in this mode it is
      // empty and a reader of it would print no caveat while looking like a
      // reader that had checked.
      if (unit_has_lost_decision || unit_calls_gated_unit)
        path_cov_outer_box_obstacle =
          std::string(
            unit_has_lost_decision
              ? "the unit contains a source decision the frontend lowered to a "
                "control-flow-free assume, so the reverting execution does not "
                "exist in the model at all"
              : "") +
          (unit_has_lost_decision && unit_calls_gated_unit ? "; " : "") +
          (unit_calls_gated_unit
             ? "the unit still calls another UNIT's own body unexpanded (" +
                 residual_unit_names +
                 "), routing an INTERNAL call through the EXTERNAL-entry body "
                 "and its ABI value gate"
             : "");

      // Snapshot each coordinate at ENTRY. The assertion sits at the exit, and
      // a parameter may have been reassigned in between; asserting on the live
      // symbol would then bound the wrong value — and bound it in the safe-
      // looking direction, since a reassigned parameter usually has a smaller
      // range than the argument did.
      const symbolt *fsym = ns.lookup(f_it->first);
      std::map<std::string, expr2tc> snap;
      // Snapshot the union of measured and PINNED coordinates. A pin on a state
      // variable has to mean its value AT ENTRY: read live at the exit it would
      // name the post-state, so a path that writes the variable would be pinned
      // to a value it only reaches on the way out. That is not a slice of the
      // input space at all.
      std::vector<outer_coordt> snap_targets = outer_coords;
      for (const auto &[pname, pval] : outer_pins)
      {
        bool present = false;
        for (const auto &c : outer_coords)
          if (c.name == pname)
            present = true;
        if (!present)
          snap_targets.push_back({pname, "0", "0", {}});
      }
      for (const auto &c : snap_targets)
      {
        expr2tc cexpr;
        std::string why;
        if (!resolve_coord(fsym, c.name, cexpr))
          why =
            "the name does not resolve to an input of this unit. Name a "
            "parameter, an environment value (`msg.value` ...), or a state "
            "variable at entry (`state.<field>`); note that `state.<field>` "
            "reaches the contract object's own components only, so a WHOLE "
            "mapping or dynamic array does not resolve -- name ONE SLOT of it "
            "as `state.<name>[<key>]` instead, where <key> is a decimal, an "
            "0x-prefixed hex literal, a parameter, or `msg.sender`";
        else
          coord_expressible(cexpr->type, why);
        if (!why.empty())
        {
          // REFUSE THE COORDINATE, KEEP THE ROUND. This used to abort, which
          // cost the whole batch — measured on three separate projects, where
          // one unusable state coordinate meant nothing at all was measured for
          // any unit. The other coordinates are independent containment
          // statements and are still worth having.
          //
          // What must NOT happen is treating it as a measured `[0, TYPE_MAX]`:
          // that widens THIS path's own region, and "only ever narrower" is the
          // invariant the whole subtraction rests on. Emitting no probe leaves
          // it out of `bounds` entirely, so no bound is ever attributed to it —
          // and it is recorded by name so the omission is visible instead of
          // reading as "measured, and it came out as the whole type".
          path_cov_refused_coords[c.name] = why;
          log_warning(
            "--path-cov-outer-box: unit '{}' — REFUSING coordinate '{}': {}. "
            "No "
            "probe is emitted for it and no bound is attributed to it; the "
            "remaining coordinates are measured as usual. This is a refusal, "
            "NOT a measurement of the full type range: those two are the same "
            "constraint to the solver and opposite claims to a reader",
            uid,
            c.name,
            why);
          continue;
        }
        const type2tc ct = cexpr->type;
        symbolt ssym;
        ssym.set_type(migrate_type_back(ct));
        ssym.name = "__ESBMC_outer$" + i2string(ghost_counter++);
        ssym.id = "path_cov::" + id2string(ssym.name);
        ssym.lvalue = true;
        ssym.static_lifetime = false;
        ssym.is_extern = false;
        symbolt *psn;
        cov_context->move(ssym, psn);
        expr2tc sn = symbol2tc(migrate_type(psn->get_type()), psn->id);
        auto entry = goto_program.instructions.begin();
        goto_programt::instructiont dcl;
        dcl.type = DECL;
        dcl.code = code_decl2tc(ct, psn->id);
        dcl.location = entry->location;
        dcl.location.property("skipped");
        dcl.function = entry->location.get_function();
        goto_program.instructions.insert(entry, dcl);
        goto_programt::instructiont asg;
        asg.type = ASSIGN;
        asg.code = code_assign2tc(sn, cexpr);
        asg.location = entry->location;
        asg.location.property("skipped");
        asg.function = entry->location.get_function();
        goto_program.instructions.insert(entry, asg);
        snap[c.name] = sn;
        // The type's own range, taken without asking the solver anything.
        if (is_unsignedbv_type(ct))
        {
          BigInt hi = 1;
          for (unsigned b = 0; b < ct->get_width(); ++b)
            hi *= 2;
          path_cov_outer_box_type_range[c.name] = {"0", integer2string(hi - 1)};
          // PUBLISHED, not just used internally. The driver chooses the ladder
          // and cannot choose it correctly without knowing how wide the
          // coordinate is -- measured: the geometric ladder lays probes up to
          // 2^255 whatever the type is, so on a 160-bit `address` the
          // out-of-type values WRAP, the bracket comes back claiming a holding
          // lower bound of 2^255 on a type that stops at 2^160-1, and the
          // driver's next span is inverted. The tool had this number the whole
          // time and kept it to itself.
          log_status(
            "--path-cov-outer-box: coordinate '{}' has TYPE RANGE [0, {}] "
            "({}-bit unsigned). Probe values outside it are meaningless: a "
            "bound is built as a constant of this type, so an out-of-range "
            "value wraps and the probe asks about a different number",
            c.name,
            integer2string(hi - 1),
            ct->get_width());
        }
        else if (is_bool_type(ct))
        {
          // ---- S5: PUBLISH IT, or the subtraction is skipped entirely ----
          //
          // The publication above is unsigned-only, and this map is what seeds
          // `bounds` with the free bound in report_outer_boxes. A coordinate
          // absent from it has `have_l`/`have_u` false unless some probe HOLDS,
          // so with no seed a bool coordinate can land in "(unbounded within
          // the probed span)" and be dropped from `box`, and the sibling
          // subtraction never runs on it.
          //
          // The range is [0, 1] and NOT [0, 2^get_width()-1]: get_width() is 8
          // for a bool (the memory-model byte), which would publish [0, 255] to
          // the driver and invite a ladder of 255 meaningless probes.
          path_cov_outer_box_type_range[c.name] = {"0", "1"};
          log_status(
            "--path-cov-outer-box: coordinate '{}' has TYPE RANGE [0, 1] "
            "(BOOLEAN). It is an EQUALITY coordinate, not an interval one: a "
            "probe is emitted as `c == false` / `c == true` rather than as an "
            "ordering comparison, and only the values 0 and 1 are meaningful. "
            "Note its bit width reads as 8 (the memory-model byte) — that is "
            "NOT its value range",
            c.name);
        }
      }

      // A pin whose coordinate was refused cannot be established, so it must
      // stop being CLAIMED. The pins are what every measured and every
      // subtracted region below is labelled with ("measured under bal == 0"),
      // and an unapplied pin left in that label describes a slice nothing was
      // restricted to. Dropping it from the antecedent instead makes the run
      // measure a LARGER slice — less informative, and honestly labelled.
      std::vector<std::pair<std::string, std::string>> pins_applied;
      for (const auto &p : outer_pins)
        if (snap.count(p.first) != 0)
          pins_applied.push_back(p);
      if (pins_applied.size() != outer_pins.size())
        log_warning(
          "--path-cov-outer-box: unit '{}' — {} of {} requested pin(s) name a "
          "REFUSED coordinate and were NOT applied. Every region below is "
          "therefore a statement about a WIDER slice than the one requested, "
          "and is labelled with the pins that were actually applied — never "
          "with the ones that were asked for",
          uid,
          outer_pins.size() - pins_applied.size(),
          outer_pins.size());
      path_cov_outer_box_pins = pins_applied;

      // Each path's ladder goes at THAT path's own exit. Putting every probe at
      // every exit would multiply the claim count by the number of exits for no
      // information: at another exit the antecedent `tr == enc` is false and
      // the implication is vacuous.
      size_t emitted = 0;
      for (const auto &[penc, pdepth] : path_cov_outer_box_paths)
      {
        goto_programt::targett exit_pc;
        bool found = false;
        for (auto &e : to_insert)
        {
          const std::string &cm = std::get<2>(e);
          const size_t q = cm.rfind(":path:");
          if (q == std::string::npos)
            continue;
          if (strtoull(cm.substr(q + 6).c_str(), nullptr, 10) == penc)
          {
            exit_pc = std::get<0>(e);
            found = true;
            break;
          }
        }
        if (!found)
        {
          log_warning(
            "--path-cov-outer-box: path enc={} is not among this unit's "
            "enumerated paths; no probe emitted for it, and it therefore also "
            "contributes NOTHING to the sibling subtraction — which makes "
            "every "
            "other path's certified region a claim about a partition that is "
            "missing a part",
            penc);
          continue;
        }
        // The ordered decision sequence, walked out of `enc`. Printed per path
        // so that two paths the payload cannot separate can be compared HERE:
        // their sequences agree up to the decision that splits them, and the
        // guard at that source line is where the separating quantity is
        // written. That is the one place it can be named when it is absent from
        // the counterexample — which, measured, is every reach-gate path so
        // far.
        if (trace_decisions && pdepth > 0)
        {
          // Reads the PUBLISHED tables, not a second private copy: the log line
          // and the report field are then the same data, so a defect in the
          // recording cannot show up in one and not the other.
          const auto &tbl = path_decision_table[uid];
          const auto &idx = path_decision_index[uid];
          std::string seq;
          for (uint64_t k = 0; k < pdepth; ++k)
          {
            const uint64_t key = penc >> (pdepth - 1 - k);
            auto dit = idx.find(key);
            seq += "\n    #" + std::to_string(k + 1) + " ";
            // A missing key is reported, not skipped. Silence here would read
            // as "this path has fewer decisions", which is a claim about the
            // path rather than about the recording.
            if (dit == idx.end() || dit->second >= tbl.size())
              seq += "<not recorded — enc key " + std::to_string(key) + ">";
            else
            {
              const auto &d = tbl[dit->second];
              seq +=
                d.loc + " (operand " + std::to_string(d.sub) + ") [guard " +
                ((key & 1) ? "TRUE" : "FALSE") + "]" +
                (d.synthetic_abi_gate ? " [synthesised ABI value gate]" : "");
            }
          }
          log_status(
            "--path-cov-outer-box: path enc={} depth={} DECISION SEQUENCE (in "
            "order; two paths diverge at the first index where these differ, "
            "and the guard on that line names the quantity that separates "
            "them):{}",
            penc,
            pdepth,
            seq);
        }

        // `tr != enc || cnt != depth || !pins || <bound>` — the implication,
        // written out. The pins join the ANTECEDENT: the bound is then asserted
        // only about the slice through them, which is exactly what makes a
        // diagonal domain measurable at all.
        expr2tc not_this_path = or2tc(
          notequal2tc(tr, constant_int2tc(utype, BigInt(penc))),
          notequal2tc(cnt, constant_int2tc(utype, BigInt(pdepth))));
        for (const auto &[pname, pval] : pins_applied)
        {
          const expr2tc &pexpr = snap.at(pname);
          if (is_bool_type(pexpr->type))
          {
            // ---- S5: no constant_int2tc on a bool, and no silent coercion
            // ----
            //
            // `!= 0 means true` would accept a pin value of 7 and label every
            // region below "measured under flag == 7" while having pinned it to
            // true. The pin is part of the ANSWER (it is printed with every box
            // and every certified region), so a value the coordinate cannot
            // hold is refused rather than rounded.
            const BigInt pv = string2integer(pval);
            if (pv < 0 || pv > 1)
            {
              log_error(
                "--path-cov-outer-box: REFUSING the pin '{} == {}': the "
                "coordinate is a BOOLEAN and its value domain is {{0, 1}}. "
                "Coercing this to true/false would label every region below "
                "with a pin that was never applied, and the pin is part of the "
                "answer, not a detail of how it was obtained",
                pname,
                pval);
              exit(1);
            }
            not_this_path = or2tc(
              not_this_path,
              notequal2tc(pexpr, pv == 0 ? gen_false_expr() : gen_true_expr()));
            continue;
          }
          not_this_path = or2tc(
            not_this_path,
            notequal2tc(
              pexpr, constant_int2tc(pexpr->type, string2integer(pval))));
        }
        for (const auto &c : outer_coords)
        {
          // Refused above: no snapshot exists, so no probe can be built. Note
          // `snap[c.name]` would CREATE a null entry here rather than tell us
          // that — the lookup has to come first.
          if (snap.count(c.name) == 0)
            continue;
          const type2tc ct = snap[c.name]->type;
          // Either the driver's explicit values, or a uniform subdivision of
          // [lo, hi]. Both end up as the same kind of probe; only the choice of
          // where to put them differs, and that choice is policy.
          std::vector<BigInt> probe_vals;
          for (const auto &v : c.values)
            probe_vals.push_back(string2integer(v));
          if (c.subdivide)
          {
            const BigInt lo = string2integer(c.lo), hi = string2integer(c.hi);
            if (hi < lo)
            {
              // exit, NOT abort. This is a malformed SPEC -- the caller handed
              // in an inverted span -- and it is reachable from an ordinary
              // driver bug: a bracket computed across a coordinate's wrapped
              // probes produces exactly this. A SIGABRT turns a recordable
              // refusal into a core dump, which unattended is the difference
              // between a datum and a lost run; measured, that is how it
              // surfaced.
              log_error(
                "--path-cov-outer-box: REFUSING coordinate '{}': hi < lo "
                "(lo={}, hi={}), so the span contains no probe value. This is "
                "a "
                "malformed span in the SPEC, not a property of the path",
                c.name,
                c.lo,
                c.hi);
              exit(1);
            }
            const BigInt span = hi - lo;
            for (size_t k = 0; k <= outer_probes + 1; ++k)
              probe_vals.push_back(
                lo + (span * BigInt((int64_t)k)) /
                       BigInt((int64_t)(outer_probes + 1)));
          }
          // ---- DROP probe values the coordinate's type cannot hold ----
          //
          // The bound is `constant_int2tc(ct, v)`, so a value above the type's
          // maximum WRAPS and the probe asks about a different number than the
          // one the driver wrote down. That is not a weaker measurement, it is
          // a wrong one: measured on a 160-bit `address`, the geometric
          // ladder's 2^255 probe came back HOLDING as a lower bound -- on a
          // type whose largest value is 2^160-1 -- and the driver's next span
          // was inverted, which killed the whole loop.
          //
          // Dropped rather than clamped. Clamping invents a probe nobody asked
          // for; dropping removes one that could not mean anything, and the
          // type maximum is already seeded as the free bound, so nothing is
          // lost. Counted and reported, because a silently shorter ladder is a
          // silently coarser measurement.
          //
          // ---- S5: THE DROP WAS UNSIGNED-ONLY, SO A BOOL BYPASSED IT ----
          //
          // With bool accepted by coord_expressible, a driver value of `7` on a
          // bool coordinate reached the probe builder untouched. Its type
          // maximum is 1, not 2^get_width()-1 = 255: get_width() answers a
          // memory-model question, not a value-domain one.
          size_t out_of_type = 0;
          if (is_unsignedbv_type(ct) || is_bool_type(ct))
          {
            BigInt tmax = 1;
            if (is_bool_type(ct))
              tmax = 1;
            else
            {
              for (unsigned b = 0; b < ct->get_width(); ++b)
                tmax *= 2;
              tmax -= 1;
            }
            std::vector<BigInt> kept;
            kept.reserve(probe_vals.size());
            for (const BigInt &v : probe_vals)
            {
              if (v >= 0 && v <= tmax)
                kept.push_back(v);
              else
                ++out_of_type;
            }
            probe_vals.swap(kept);
          }
          // The two sources can overlap; a duplicated probe is a duplicated
          // claim name, which collides in `all_claims` and silently drops one.
          std::sort(probe_vals.begin(), probe_vals.end());
          probe_vals.erase(
            std::unique(probe_vals.begin(), probe_vals.end()),
            probe_vals.end());
          if (out_of_type > 0)
            log_warning(
              "--path-cov-outer-box: coordinate '{}' — DROPPED {} probe "
              "value(s) that do not fit its type; {} probe value(s) remain. "
              "Those values would have been built as constants of this type "
              "and "
              "WRAPPED, so each would have measured a different number than "
              "the "
              "one requested. The ladder is correspondingly coarser here, "
              "which "
              "is a resolution loss, not a wrong bound",
              c.name,
              out_of_type,
              probe_vals.size());
          if (probe_vals.empty())
          {
            log_warning(
              "--path-cov-outer-box: coordinate '{}' — NO probe value survives "
              "the type check, so NOTHING is measured on it and it appears in "
              "no box below. Its absence is a refusal, not a measurement of "
              "the "
              "full type range",
              c.name);
            path_cov_refused_coords[c.name] =
              "every requested probe value lies outside the coordinate's own "
              "type range, so every probe would have wrapped and measured a "
              "different number";
            continue;
          }
          for (const BigInt &v : probe_vals)
          {
            for (int dir = 0; dir < 2; ++dir)
            {
              const bool upper = dir == 0;
              expr2tc cmp;
              if (is_bool_type(ct))
              {
                // ---- S5: the SAME probe, written without an ordering op ----
                //
                // On {0,1} the two ladder predicates are exactly expressible as
                // equalities, so no new probe KIND is needed and
                // report_outer_boxes' tightening and type-range seed are
                // untouched:
                //
                //     c <= 0  ==  c == false        c <= 1  ==  true
                //     c >= 1  ==  c == true         c >= 0  ==  true
                //
                // The trivially-true arms are emitted as `true` rather than
                // skipped: the probe then HOLDS, which is the correct reading —
                // 1 IS an upper bound for a bool, and 0 IS a lower bound. A
                // skipped probe would instead go undecided and read as "the
                // solver was never asked".
                //
                // Values outside {0,1} cannot arrive here: the type drop above
                // removes them and the coordinate is refused when none survive.
                const expr2tc &cv = snap[c.name];
                if (upper)
                  cmp = (v <= 0) ? equality2tc(cv, gen_false_expr())
                                 : gen_true_expr();
                else
                  cmp = (v >= 1) ? equality2tc(cv, gen_true_expr())
                                 : gen_true_expr();
              }
              else
                cmp =
                  upper
                    ? lessthanequal2tc(snap[c.name], constant_int2tc(ct, v))
                    : greaterthanequal2tc(snap[c.name], constant_int2tc(ct, v));
              const std::string comment =
                id2string(f_it->first) + ":path:" + std::to_string(penc) + "#" +
                (upper ? "ub" : "lb") + "_" + c.name + "_" + integer2string(v);
              const std::string loc = exit_pc->location.as_string();
              all_claims.insert({comment, loc});
              path_cov_outer_box_probes.push_back(
                {penc, c.name, upper, integer2string(v), {comment, loc}});
              insert_assert(
                goto_program, exit_pc, or2tc(not_this_path, cmp), comment);
              ++emitted;
            }
          }
        }
      }
      log_status(
        "--path-cov-outer-box: unit '{}' — emitted {} ladder probe(s) as ONE "
        "batch. The assumption is fixed per path and only the assertions vary, "
        "which is exactly what lets a whole ladder be judged in a single run "
        "instead of one query per widening step",
        uid,
        emitted);
      total_paths += emitted;
      continue;
    }

    // ---- STAGE 2: the certification query ----
    //
    // Reached only with --path-cov-certify. Everything above still ran —
    // expansion, the ABI gate, Phase-1 `tr`/`cnt` accounting, the
    // `tr`-completeness invariant, the exit census, the decision-set census —
    // because they are the defences, and certifying against accounting that
    // nothing checked would be certifying nothing. It also matters that the
    // query below reads the SAME `tr` the enumeration writes: the argument that
    // certification is immune to goal-cap truncation is exactly "the cap drops
    // exit asserts, never the accounting", and two separate accumulators would
    // break it.
    if (certify_on)
    {
      const std::string uid = f_it->first.as_string();
      const bool is_target =
        uid == certify_unit ||
        uid.find("@F@" + certify_unit + "#") != std::string::npos;
      if (!is_target)
        continue; // other units contribute nothing to this query

      // ---- S2: A NAMED OBSTACLE UNIT MAY NOT BE CERTIFIED ----
      //
      // The stage-3 ladder refuses one (N4) and certification did not, although
      // certification is the OLDER gate and the one every region in the report
      // is stamped with. On an obstructed unit the model admits an execution
      // the chain does not have, so `assume(box); assert(tr == pi)` can answer
      // SUCCESSFUL for a box whose inputs include executions that cannot happen
      // -- and the region is then handed to an emitter as certified.
      //
      // THE TRAP, and it is the reason this is read from the LOCALS: the public
      // `named_obstacle_paths` map is filled by the insertion loop BELOW, which
      // this branch `continue`s past. In certify mode that map is EMPTY, so a
      // gate reading it would never fire while looking exactly like a gate. The
      // same trap is documented on the stage-3 side and was written down before
      // either was implemented; both flags here are per-unit locals computed
      // above, independent of that map.
      //
      // Two routes, counted apart because they are different defects needing
      // the same containment: a source decision lowered to a control-flow-free
      // assume (the reverting execution does not exist in the model AT ALL, so
      // no query can see it), and an unexpanded call to another UNIT (which
      // routes an internal call through the external-entry body and its ABI
      // value gate).
      if (unit_has_lost_decision || unit_calls_gated_unit)
      {
        log_error(
          "--path-cov-certify: unit '{}' — REFUSING THE QUERY: this unit is a "
          "NAMED OBSTACLE, so the model admits an execution the chain does not "
          "have. A box certified here can contain inputs whose modelled "
          "execution cannot happen on chain, and the region would be reported "
          "as certified all the same — which is a false certificate, not a "
          "weak "
          "one (lost decision: {}; calls a gated unit: {}). Certification is "
          "not attempted",
          uid,
          unit_has_lost_decision ? "yes" : "no",
          unit_calls_gated_unit ? "yes" : "no");
        exit(1);
      }

      // 1. Assume the box at unit entry. Each bound names either a call
      //    argument (resolved against this unit's own parameter list) or an EVM
      //    environment value (`msg.value` etc.). An unresolvable name is FATAL:
      //    silently dropping a bound would widen the box being certified beyond
      //    what was asked for, and the run would still say SUCCESSFUL.
      const symbolt *fsym = ns.lookup(f_it->first);
      size_t bounds_emitted = 0;
      // Counted at the EMISSION, not at the parse. A hole that was read out of
      // the spec and then never reached the assumption would leave the query
      // certifying a WIDER region than the one reported — and the parse-time
      // line would still say the holes were there.
      size_t holes_emitted = 0;

      // ---- AN EMPTY BOX IS NOT A CERTIFICATE ----
      //
      // `lo > hi` makes the entry assumption `lo <= c <= hi` UNSATISFIABLE.
      // Then nothing executes, every exit assert holds because it is never
      // reached, and the query answers VERIFICATION SUCCESSFUL. Measured on
      // this very fixture before this check existed: `a in [100, 11]` certified
      // cleanly, exit 0, with the box printed beside it.
      //
      // That is the SECOND route to a false certificate found in one night —
      // the first was a driver reading the verdict as a substring of an ESBMC
      // warning — and it is the worse-looking one, because the output is not
      // merely green: it is a certificate naming a region that contains no
      // inputs at all. The driver already refuses these, but the only
      // reliability gate this method has should not depend on its caller
      // guarding it, and that caller's own gate has already failed once.
      //
      // The refusal belongs BEFORE the query is emitted, not where its answer
      // is read. An unsatisfiable assumption does not make the question hard,
      // it makes the question meaningless; there is nothing to interpret
      // afterwards.
      {
        std::set<std::string> box_names;
        for (const auto &b : certify_box)
        {
          std::string bad;
          if (string2integer(b.hi) < string2integer(b.lo))
            bad = "the box is EMPTY on this coordinate (lo=" + b.lo +
                  " > hi=" + b.hi +
                  "), so the entry assumption is unsatisfiable and every exit "
                  "assert would hold for want of an execution";
          // Closes the obvious hole in the check above: bounding one name twice
          // can intersect to nothing while each bound is individually fine, and
          // a per-bound test would wave both through. Duplicates carry no
          // meaning in this spec anyway, so refusing them costs nothing and
          // leaves no case where "not empty by this test" and "not empty" come
          // apart.
          else if (!box_names.insert(b.name).second)
            bad =
              "the coordinate is bounded TWICE in this spec; two bounds on "
              "one name can intersect to an empty box while each is "
              "individually well-formed, which the emptiness test above "
              "would not see";
          else if (!b.holes.empty())
          {
            // A PUNCHED interval has a SECOND way of being empty, and it is the
            // one a `lo <= hi` test cannot see: `a in [5,5] \ {5}` passes that
            // test and still admits no input at all. Same consequence as the
            // inverted interval — an unsatisfiable assumption certifies
            // VACUOUSLY — so it gets the same refusal rather than a warning.
            //
            // Counting distinct holes INSIDE [lo, hi] is what makes this exact:
            // a hole outside the interval removes nothing, and counting it
            // would refuse a perfectly good box. The count is bounded by the
            // spec's own size, so comparing it against a 256-bit span is safe.
            const BigInt lo = string2integer(b.lo), hi = string2integer(b.hi);
            std::set<std::string> inside;
            for (const auto &h : b.holes)
            {
              const BigInt hv = string2integer(h);
              if (hv >= lo && hv <= hi)
                inside.insert(integer2string(hv));
            }
            if (BigInt((int64_t)inside.size()) >= (hi - lo + 1))
              bad = "the PUNCHED box is EMPTY on this coordinate: [" + b.lo +
                    ", " + b.hi + "] holds " + integer2string(hi - lo + 1) +
                    " value(s) and the holes remove all of them, so the entry "
                    "assumption is unsatisfiable. `lo <= hi` does NOT catch "
                    "this — the interval is well-formed and the punching is "
                    "what empties it";
          }
          if (!bad.empty())
          {
            log_error(
              "--path-cov-certify: unit '{}' — REFUSING THE QUERY on "
              "coordinate "
              "'{}': {}. An unsatisfiable assumption certifies VACUOUSLY: the "
              "run would print VERIFICATION SUCCESSFUL next to a box holding "
              "no "
              "inputs, which is a false certificate rather than a weak one. "
              "Certification is not attempted",
              uid,
              b.name,
              bad);
            exit(1);
          }
        }
      }

      for (const auto &b : certify_box)
      {
        expr2tc bs;
        std::string why;
        if (!resolve_coord(fsym, b.name, bs))
          why =
            "the name does not resolve to an input of this unit. Name a "
            "parameter of this unit, an environment value as `msg.value` / "
            "`tx.origin` / `block.timestamp`, a state variable at entry as "
            "`state.<field>` (which reaches the contract object's own "
            "components only — a WHOLE mapping or dynamic array does not "
            "resolve), or ONE SLOT of such a store as `state.<name>[<key>]` "
            "with <key> a decimal, an 0x-prefixed hex literal, a parameter of "
            "this unit, or `msg.sender`";
        else
          coord_expressible(bs->type, why);
        if (!why.empty())
        {
          // REFUSE THE QUERY, not just the coordinate — the opposite of the
          // outer box, and deliberately so. There, one missing coordinate costs
          // information; here it would change the ANSWER: certification asks
          // "does every input in THIS box walk this path", and a box missing
          // one of its requested bounds is a strictly WIDER box. The run could
          // then answer SUCCESSFUL about a region nobody asked about, which is
          // the single outcome this query exists to prevent.
          //
          // Exiting rather than aborting is the actual fix. The verdict is the
          // same refusal it always was; what changes is that it is now a clean
          // non-zero exit with a readable reason instead of SIGABRT, so an
          // unattended driver records a named failure instead of a core dump.
          // Note that no verdict line is printed either, so a caller reading
          // SUCCESSFUL/FAILED as whole lines sees its explicit third state.
          // Deliberately NOT recorded in path_cov_refused_coords. The only
          // reader of that map is report_outer_boxes, which is unreachable from
          // here -- this branch exits the process. A write nothing can observe
          // is not defence in depth, it is a line that reads like one, and the
          // next person to add a consumer would have to discover for themselves
          // that this entry never arrives.
          log_error(
            "--path-cov-certify: unit '{}' — REFUSING THE QUERY because "
            "coordinate '{}' cannot be expressed: {}. Certification is not "
            "attempted: dropping the bound would certify a WIDER box than the "
            "one asked for, and answering about a different box is worse than "
            "not answering. Re-run with a box this stage can express, or read "
            "this as the path being out of reach of the current coordinate set",
            uid,
            b.name,
            why);
          exit(1);
        }
        const type2tc bt = bs->type;

        // ---- The spec's decimals must FIT the coordinate's own type ----
        //
        // Every bound is built with constant_int2tc ON THE COORDINATE'S TYPE,
        // so a decimal above the type's maximum WRAPS silently and the query is
        // emitted about a different value than the one written down. The
        // verdict then describes a box nobody asked for, and if it comes back
        // SUCCESSFUL it is a false certificate — the same shape as the signed
        // hole documented in coord_expressible, arrived at through the value
        // rather than through the type.
        //
        // Checked here rather than in the structural block above because this
        // is the first point where the coordinate's TYPE is known; the block
        // above can only compare decimals with each other. coord_expressible
        // has already restricted `bt` to an unsigned bit-vector or a bool.
        //
        // The range now comes from `path_cov_fits_type` rather than from a
        // second inline `2^width - 1`. That is what the duplication note at the
        // top of this file asks for, and S5 is precisely the change that would
        // otherwise have gone into one copy and not the other: a bool's
        // `get_width()` is 8, so the inline arithmetic admitted [0, 255] and
        // `state.flag in [0, 200]` would have been certified as written.
        {
          std::vector<std::pair<std::string, std::string>> vals = {
            {"lo", b.lo}, {"hi", b.hi}};
          for (const auto &h : b.holes)
            vals.push_back({"hole", h});
          for (const auto &[what, txt] : vals)
          {
            std::string tmax;
            if (path_cov_fits_type(bt, txt, tmax))
              continue;
            log_error(
              "--path-cov-certify: unit '{}' — REFUSING THE QUERY on "
              "coordinate "
              "'{}': the {} value {} does not fit the coordinate's own type "
              "(admissible range [0, {}]). The bound is built as a constant of "
              "that type, so an out-of-range decimal WRAPS and the query would "
              "be emitted about a different value than the one written here — "
              "answering SUCCESSFUL about a box nobody asked for. "
              "Certification "
              "is not attempted",
              uid,
              b.name,
              what,
              txt,
              tmax);
            exit(1);
          }
        }

        // `lo <= c <= hi`, then one `c != h` per hole (Definition 5). With no
        // holes this is byte for byte the assumption emitted before punched
        // intervals existed, so every existing spec is unaffected.
        expr2tc bguard;
        if (is_bool_type(bt))
        {
          // ---- S5: a bool is an EQUALITY coordinate ----
          //
          // No ordering operator is built on it at all (that is a SIGABRT, see
          // path_cov_bool_domain_guard) and no integer constant of its type is
          // built either. lo/hi/holes collapse to the allowed set and the whole
          // bound — interval AND holes — is one disjunction of equalities.
          size_t applied = 0;
          bguard = path_cov_bool_domain_guard(bs, b.lo, b.hi, b.holes, applied);
          if (is_nil_expr(bguard))
          {
            // Unreachable through the structural gates above (`lo > hi` and
            // punched-empty are both refused there, and on {0,1} those two
            // exhaust the ways S can be empty). Kept anyway, and kept as a
            // REFUSAL rather than an assert: if a fourth route ever opens, the
            // failure it produces is a VERIFICATION SUCCESSFUL next to a box
            // holding no inputs, which is the one outcome this query exists to
            // prevent.
            log_error(
              "--path-cov-certify: unit '{}' — REFUSING THE QUERY on "
              "coordinate "
              "'{}': it is a BOOLEAN and the box admits NEITHER value "
              "(lo={}, hi={}, {} hole(s)). The entry assumption is then "
              "unsatisfiable and every exit assert would hold for want of an "
              "execution — a false certificate, not a weak one",
              uid,
              b.name,
              b.lo,
              b.hi,
              b.holes.size());
            exit(1);
          }
          holes_emitted += applied;
        }
        else
        {
          bguard = and2tc(
            greaterthanequal2tc(bs, constant_int2tc(bt, string2integer(b.lo))),
            lessthanequal2tc(bs, constant_int2tc(bt, string2integer(b.hi))));
          for (const auto &h : b.holes)
          {
            bguard = and2tc(
              bguard, notequal2tc(bs, constant_int2tc(bt, string2integer(h))));
            // Incremented HERE, inside the conjunction, and not from
            // `b.holes.size()` next to the insert. MEASURED on the fault
            // injection for this very change: with the conjunction disabled the
            // query correctly flipped to FAILED while the line above still
            // reported "1 hole(s) punched" — a counter that reads the SPEC
            // cannot witness whether the spec reached the formula, which is the
            // only thing it was added to witness.
            ++holes_emitted;
          }
        }

        goto_programt::instructiont asm_i;
        asm_i.type = ASSUME;
        asm_i.guard = bguard;
        asm_i.location = goto_program.instructions.begin()->location;
        // "skipped" keeps this out of the decision-set census, which flags a
        // user-source ASSUME as a lowered-away branch. This one is ours.
        asm_i.location.property("skipped");
        asm_i.function =
          goto_program.instructions.begin()->location.get_function();
        goto_program.instructions.insert(
          goto_program.instructions.begin(), asm_i);
        ++bounds_emitted;
      }

      // 2. Assert the path identity at EVERY exit — see the header comment.
      //    Collected first, then inserted: inserting while walking the same
      //    list is how the ABI gate acquired a self-loop once already.
      std::vector<goto_programt::targett> exits;
      Forall_goto_program_instructions (xit, goto_program)
        if (xit->is_return() || xit->is_end_function() || is_error_call(xit))
          exits.push_back(xit);

      const expr2tc cert_guard = and2tc(
        equality2tc(tr, constant_int2tc(utype, BigInt(certify_enc))),
        equality2tc(cnt, constant_int2tc(utype, BigInt(certify_depth))));
      size_t exit_idx = 0;
      for (auto xpc : exits)
      {
        // Comment shape MUST stay `<unit-id>:path:<enc>` — the unit id first,
        // with nothing in front of it. MEASURED: a leading "certify:" made the
        // report's `path_function` read `certify:sol:@C@Box@F@f#18`, the
        // counterexample harvest builds the expected argument scope from that
        // string, every nondet then failed the scope test and was filed as
        // harness-internal (dropped 19 -> 25, `inputs` empty). The refutation
        // still printed a verdict, so the loss was silent — and the witness
        // VALUE is the entire point of a refutation: without it there is
        // nothing to shrink the box with. The mode is announced by the banner
        // above, not by decorating a key another component parses.
        const std::string comment = id2string(f_it->first) +
                                    ":path:" + std::to_string(certify_enc) +
                                    "#exit" + std::to_string(exit_idx++);
        const std::string xloc = xpc->location.as_string();
        all_claims.insert({comment, xloc});
        // Recorded so the RESULT line can tell REFUTED from VACUOUS. Reading
        // "some claim in all_claims was refuted" instead would call every
        // successful certification a refutation, because the non-vacuity
        // witness emitted below is refuted on exactly those runs.
        path_cov_certify_exit_keys.push_back({comment, xloc});
        insert_assert(goto_program, xpc, cert_guard, comment);
      }

      // ---- THE NON-VACUITY WITNESS (see the header) ----
      //
      // At pi's OWN exit, carrying only `tr != enc || cnt != depth`. REFUTED
      // means some execution the box admits walks THIS path; anything else
      // means the box is semantically empty and every `#exitN` assert above
      // holds for want of an execution.
      //
      // pi's own exit comes from `to_insert`, the enumeration's own record --
      // not from `exits`, which is a flat scan of exit-kind instructions and
      // does not know which one belongs to enc.
      {
        goto_programt::targett nv_pc;
        bool nv_found = false;
        for (const auto &e : to_insert)
        {
          const std::string &cm = std::get<2>(e);
          const size_t q = cm.rfind(":path:");
          if (q == std::string::npos)
            continue;
          if (strtoull(cm.substr(q + 6).c_str(), nullptr, 10) == certify_enc)
          {
            nv_pc = std::get<0>(e);
            nv_found = true;
            break;
          }
        }
        // ---- enc is not a path of this unit ----
        //
        // Today this is not silently green -- `tr == enc` fails on every
        // execution, so the run answers FAILED -- but it answers the WRONG
        // QUESTION, and the reason it gives ("an input inside the box leaves
        // the path") is about a path that does not exist. The driver then
        // shrinks a box against a refutation it can never satisfy. Refused for
        // the same reason the stage-3 ladder refuses it: a spec naming a
        // non-existent path is a caller defect, and answering it at all
        // launders that defect into a measurement.
        if (!nv_found)
        {
          log_error(
            "--path-cov-certify: unit '{}' — REFUSING THE QUERY: path enc={} "
            "is not among this unit's {} enumerated path(s). `tr == {}` is "
            "then "
            "false on every execution, so the run would answer FAILED and name "
            "an escaping input for a path that does not exist — a refutation "
            "the driver can never satisfy by shrinking. Check enc against the "
            "enumeration run's report",
            uid,
            certify_enc,
            to_insert.size(),
            certify_enc);
          exit(1);
        }
        const std::string nv_comment = id2string(f_it->first) +
                                       ":path:" + std::to_string(certify_enc) +
                                       "#nonvacuous";
        const std::string nv_loc = nv_pc->location.as_string();
        all_claims.insert({nv_comment, nv_loc});
        path_cov_certify_nonvacuous_key = {nv_comment, nv_loc};
        insert_assert(
          goto_program,
          nv_pc,
          or2tc(
            notequal2tc(tr, constant_int2tc(utype, BigInt(certify_enc))),
            notequal2tc(cnt, constant_int2tc(utype, BigInt(certify_depth)))),
          nv_comment);
      }

      ++certify_units_matched;
      log_status(
        "--path-cov-certify: unit '{}' — assumed {} input bound(s) ({} hole(s) "
        "punched) at entry "
        "and asserted `tr == {} && cnt == {}` at ALL {} exit(s) of the unit. "
        "Asserting at every exit is what makes the query non-vacuous: an input "
        "inside the box that walks a DIFFERENT path leaves through a different "
        "exit, and would never be checked if the assert sat only on this "
        "path's own exit",
        uid,
        bounds_emitted,
        holes_emitted,
        certify_enc,
        certify_depth,
        exits.size());
      total_paths += exits.size();
      continue;
    }

    // ---- STAGE 3: POST-STATE ASSERTION SYNTHESIS (--path-cov-assert) ----
    //
    // Third branch in the same place and for the same reason as the two above:
    // expansion, the ABI gate, Phase-1 `tr`/`cnt` accounting, the
    // `tr`-completeness invariant, the exit census and the decision-set census
    // have all already run, and the antecedent asserted here -- `tr != enc ||
    // cnt != depth` -- IS that accounting.
    if (assert_on)
    {
      const std::string uid = f_it->first.as_string();
      if (
        uid != assert_unit &&
        uid.find("@F@" + assert_unit + "#") == std::string::npos)
        continue; // other units contribute nothing to this ladder
      ++assert_units_matched;

      auto entry = goto_program.instructions.begin();

      // ---- N4: A NAMED OBSTACLE UNIT MAY NOT BE GIVEN AN ORACLE ----
      //
      // On an obstructed unit the model admits executions the chain does not
      // have, so a HOLDS verdict authorises an `assertEq` that can be RED on
      // the unmodified contract -- the single outcome this pipeline must never
      // produce.
      if (unit_has_lost_decision || unit_calls_gated_unit)
      {
        log_error(
          "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: this unit is a "
          "NAMED OBSTACLE. A post-state assertion is what a generated test "
          "turns into an assertEq, and on an obstructed unit the model admits "
          "an execution the chain does not have -- so a candidate that HOLDS "
          "here can still be RED on the unmodified contract. No candidate is "
          "emitted: an assertion that cannot be trusted is worse than none "
          "(lost decision: {}; calls a gated unit: {})",
          uid,
          unit_has_lost_decision ? "yes" : "no",
          unit_calls_gated_unit ? "yes" : "no");
        exit(1);
      }

      // ---- Routes 1-3: the region's structure, before any type is known ----
      {
        std::set<std::string> region_names;
        for (const auto &b : assert_region)
        {
          const std::string bad = path_cov_structural_refusal(b, region_names);
          if (!bad.empty())
          {
            log_error(
              "--path-cov-assert: unit '{}' -- REFUSING THE LADDER on region "
              "coordinate '{}': {}. An unsatisfiable entry assumption is worse "
              "here than in certification: nothing executes, so EVERY "
              "candidate "
              "on the ladder holds for want of an execution and the report "
              "reads as a whole set of certified post-state assertions",
              uid,
              b.name,
              bad);
            exit(1);
          }
        }
      }

      // ---- Resolve, type-check and ASSUME the region at unit entry ----
      const symbolt *fsym = ns.lookup(f_it->first);
      size_t bounds_emitted = 0;
      // Counted at the EMISSION, inside the conjunction, never from
      // `b.holes.size()`: a counter that reads the SPEC cannot witness whether
      // the spec reached the formula.
      size_t holes_emitted = 0;
      for (const auto &b : assert_region)
      {
        expr2tc bs;
        std::string why;
        if (!resolve_coord(fsym, b.name, bs))
          why =
            "the name does not resolve to an input of this unit. Name a "
            "parameter, an environment value as `msg.value` / `tx.origin` / "
            "`block.timestamp`, a state variable at entry as `state.<field>` "
            "(which reaches the contract object's own components only -- a "
            "WHOLE mapping or dynamic array does not resolve), or ONE SLOT of "
            "such a store as `state.<name>[<key>]` with <key> a decimal, an "
            "0x-prefixed hex literal, a parameter of this unit, or "
            "`msg.sender`";
        else
          coord_expressible(bs->type, why);
        if (!why.empty())
        {
          // REFUSE THE LADDER, not just the coordinate -- the certify
          // disposition. Dropping a requested bound would assume a WIDER region
          // and certify every candidate over inputs nobody asked about.
          log_error(
            "--path-cov-assert: unit '{}' -- REFUSING THE LADDER because "
            "region "
            "coordinate '{}' cannot be expressed: {}",
            uid,
            b.name,
            why);
          exit(1);
        }
        const type2tc bt = bs->type;
        {
          const std::string bad = path_cov_out_of_type_refusal(b, bt);
          if (!bad.empty())
          {
            log_error(
              "--path-cov-assert: unit '{}' -- REFUSING THE LADDER on region "
              "coordinate '{}': {}. The bound is built as a constant of that "
              "type, so an out-of-range decimal WRAPS and the region assumed "
              "would not be the region written here",
              uid,
              b.name,
              bad);
            exit(1);
          }
        }
        expr2tc bguard;
        if (is_bool_type(bt))
        {
          // ---- S5: same collapse as the certify side, same helper ----
          //
          // Deliberately the SAME function rather than a second copy: the file
          // already carries one duplicated bound-parse/gate pair and says at
          // the top what that costs ("a fix to one copy does not reach the
          // other").
          size_t applied = 0;
          bguard = path_cov_bool_domain_guard(bs, b.lo, b.hi, b.holes, applied);
          if (is_nil_expr(bguard))
          {
            log_error(
              "--path-cov-assert: unit '{}' -- REFUSING THE LADDER on region "
              "coordinate '{}': it is a BOOLEAN and the region admits NEITHER "
              "value (lo={}, hi={}, {} hole(s)). Nothing executes, so EVERY "
              "candidate on the ladder would hold for want of an execution and "
              "the table would read as a whole set of certified post-state "
              "assertions",
              uid,
              b.name,
              b.lo,
              b.hi,
              b.holes.size());
            exit(1);
          }
          holes_emitted += applied;
        }
        else
        {
          bguard = and2tc(
            greaterthanequal2tc(bs, constant_int2tc(bt, string2integer(b.lo))),
            lessthanequal2tc(bs, constant_int2tc(bt, string2integer(b.hi))));
          for (const auto &h : b.holes)
          {
            bguard = and2tc(
              bguard, notequal2tc(bs, constant_int2tc(bt, string2integer(h))));
            ++holes_emitted;
          }
        }
        goto_programt::instructiont asm_i;
        asm_i.type = ASSUME;
        asm_i.guard = bguard;
        asm_i.location = entry->location;
        // "skipped" keeps this out of the decision-set census, which flags a
        // user-source ASSUME as a lowered-away branch. This one is ours.
        asm_i.location.property("skipped");
        asm_i.function = entry->location.get_function();
        goto_program.instructions.insert(entry, asm_i);
        ++bounds_emitted;
      }

      // ---- Find pi's OWN exit, and refuse three ways it can be wrong ----
      goto_programt::targett exit_pc;
      size_t exit_idx = 0;
      bool found = false;
      for (size_t i = 0; i < to_insert.size(); ++i)
      {
        const std::string &cm = std::get<2>(to_insert[i]);
        const size_t q = cm.rfind(":path:");
        if (q == std::string::npos)
          continue;
        if (strtoull(cm.substr(q + 6).c_str(), nullptr, 10) == assert_enc)
        {
          exit_pc = std::get<0>(to_insert[i]);
          exit_idx = i;
          found = true;
          break;
        }
      }
      // ---- N2: the path's `enc` does not exist for this unit ----
      //
      // The outer-box branch only WARNS here, correctly: there a missing path
      // costs one measurement. Here it means the ladder was emitted nowhere and
      // the run prints VERIFICATION SUCCESSFUL with exit 0.
      if (!found)
      {
        log_error(
          "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: path enc={} is "
          "not among this unit's {} enumerated path(s), so not one assertion "
          "would be emitted and the run would print VERIFICATION SUCCESSFUL "
          "for a ladder it never built",
          uid,
          assert_enc,
          to_insert.size());
        exit(1);
      }
      // ---- N3: `depth` disagrees with the enumerated depth ----
      //
      // The most dangerous of the new routes because today it produces NO
      // diagnostic at all: a wrong `depth` makes `tr != enc || cnt != depth`
      // TRUE on every execution, so every candidate holds VACUOUSLY and the
      // report is indistinguishable from a completely successful certification.
      {
        auto dit = path_decision_depth.find(
          {std::get<2>(to_insert[exit_idx]), exit_pc->location.as_string()});
        if (dit == path_decision_depth.end() || dit->second != assert_depth)
        {
          log_error(
            "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: the spec "
            "says "
            "path enc={} has depth={}, the enumeration says {}. The antecedent "
            "is `tr != enc || cnt != depth`, so a wrong depth is TRUE on every "
            "execution: every candidate would hold vacuously and the report "
            "would be indistinguishable from a completely successful "
            "certification. Refused rather than warned about precisely because "
            "the wrong answer looks exactly like the right one",
            uid,
            assert_enc,
            assert_depth,
            dit == path_decision_depth.end()
              ? std::string("<no depth recorded for this exit>")
              : std::to_string(dit->second));
          exit(1);
        }
      }
      // ---- N5: what KIND of exit is it? ----
      //
      // THE TRAP: revert_paths / rollback_revert_paths /
      // undetermined_exit_paths are filled by the insertion loop BELOW, which
      // this branch `continue`s past -- so in this mode they are EMPTY and
      // reading them would classify every exit as normal. The classification
      // comes from the locals.
      {
        const bool is_error_revert = std::get<3>(to_insert[exit_idx]);
        const bool is_rollback = rollback_exits.count(exit_idx) != 0;
        const bool is_undetermined = undetermined_exits.count(exit_idx) != 0;
        if (is_error_revert)
        {
          log_error(
            "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: path enc={} "
            "exits through a CUSTOM-ERROR revert, which the frontend lowers "
            "with no state rollback. The state readable there is the state at "
            "the revert point, not the EVM post-state -- on chain every write "
            "of that transaction is undone. A post-state assertion there "
            "describes a state that does not exist",
            uid,
            assert_enc);
          exit(1);
        }
        if (is_undetermined)
        {
          log_error(
            "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: path enc={} "
            "has an UNDETERMINED exit -- no positive evidence separates a "
            "reverting execution from a normal one there, and an undetermined "
            "exit cannot become an oracle",
            uid,
            assert_enc);
          exit(1);
        }
        if (is_rollback)
          // ALLOWED and LABELLED. Here the rollback IS modelled, so the exit
          // read is the correctly restored state -- but `post == pre` holding
          // means the rollback worked, NOT that the function leaves state
          // alone. Same numbers, opposite claims.
          log_warning(
            "--path-cov-assert: unit '{}' path enc={} exits through a ROLLBACK "
            "revert. The rollback IS modelled, so the values below are the "
            "correctly RESTORED state and the ladder is emitted -- but read "
            "every verdict as a statement about a REVERTING transaction. In "
            "particular `post == pre` holding here means the rollback worked, "
            "NOT that the function leaves state alone",
            uid,
            assert_enc);
      }

      // ---- Enumerate the candidate state variables ----
      //
      // (A): by EXACT contract match, never by the substring test resolve_coord
      // uses. See path_cov_contract_object for what reading the wrong object
      // does to this mode specifically.
      const std::string own_contract = contract_of(uid);
      const symbolt *obj = path_cov_contract_object(*cov_context, own_contract);
      if (obj == nullptr)
      {
        log_error(
          "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: no contract "
          "instance object 'sol:@_ESBMC_Object_{}#' exists for this unit's own "
          "contract. The object is resolved by EXACT name on purpose: a "
          "substring match would happily pick a DIFFERENT contract's object, "
          "and the exit read would then observe an object nothing wrote -- "
          "`post == pre` would hold vacuously and the whole ladder would come "
          "back green for a contract that was never measured",
          uid,
          own_contract);
        exit(1);
      }
      const typet ostruct = ns.follow(obj->get_type());
      if (ostruct.id() != "struct")
      {
        log_error(
          "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: the contract "
          "instance object does not follow to a struct",
          uid);
        exit(1);
      }

      std::vector<std::string> comp_names;
      for (const auto &comp : to_struct_type(ostruct).components())
      {
        std::string vn = comp.get("#base_name").as_string();
        if (vn.empty())
          vn = comp.get_name().as_string();
        if (path_cov_user_state_name(vn))
          comp_names.push_back(vn);
      }

      // ---- SECOND SCAN: the state variables that are NOT components ----
      //
      // A mapping or dynamic array is not a field of the contract object -- the
      // frontend lowers those to contract-scope globals -- so iterating
      // components alone would let them VANISH from the report. In this mode an
      // absent variable reads as "no assertion was needed", i.e. as
      // "unchanged".
      //
      // The symbols are also KEPT, not merely named: the whole mapping has no
      // scalar post-state, but ONE SLOT of it does, and the slot ladder below
      // needs exactly this lookup. Recording them in one place is what stops
      // that ladder from growing a second, differently-filtered scan.
      std::map<std::string, const symbolt *> store_syms;
      {
        const std::string cpfx = "sol:@C@" + own_contract + "@";
        cov_context->foreach_operand([&](const symbolt &s) {
          const std::string id = s.id.as_string();
          if (id.rfind(cpfx, 0) != 0 || id.find("@F@") != std::string::npos)
            return;
          std::string nm = id.substr(cpfx.size());
          const size_t hash = nm.find('#');
          if (hash != std::string::npos)
            nm = nm.substr(0, hash);
          if (!path_cov_user_state_name(nm))
            return;
          if (
            std::find(comp_names.begin(), comp_names.end(), nm) !=
            comp_names.end())
            return;
          store_syms.emplace(nm, &s);
          path_cov_refused_coords[nm] =
            "a mapping or dynamic array: the frontend lowers it to a "
            "contract-scope global, not a component of the contract object, so "
            "no scalar post-state candidate can be formed for it. Its absence "
            "from the table is a REFUSAL, not a measurement. A SINGLE SLOT of "
            "it is a scalar and can be asserted about: name it in \"vars\" as "
            "`" +
            nm +
            "[<key>]`, where <key> is a parameter of this unit, an environment "
            "value or `state.<field>`";
        });
      }

      std::set<std::string> named_wanted, named_seen;
      for (const auto &v : assert_vars)
        named_wanted.insert(v.name);

      // ---- A SLOT ENTRY MUST NOT TURN THE COMPONENT LOOP INTO A WHITELIST --
      //
      // `vars` is a whitelist over the contract object's COMPONENTS. A slot
      // (`m[k]`) names something that loop can never reach, so a spec that
      // asks only for slots would otherwise silence every scalar rung and the
      // return rungs as well -- and the caller would have to list every
      // component by name to get them back. It cannot: the component set is
      // the goto model's, and the driver reads solc's storage layout, which is
      // a DIFFERENT set (a constant/immutable is a component with no slot; a
      // mapping is a slot with no component). Making the driver reconcile two
      // sets that disagree by construction is how a name ends up listed that
      // the ladder then refuses by name.
      //
      // So the whitelist is over the NON-SLOT entries only. With no slot named
      // this is bit-identical to what it was.
      bool comp_vars_present = false;
      for (const auto &v : assert_vars)
        if (v.name.find('[') == std::string::npos)
          comp_vars_present = true;

      // ---- The antecedent, and the ladder ----
      const expr2tc not_this_path = or2tc(
        notequal2tc(tr, constant_int2tc(utype, BigInt(assert_enc))),
        notequal2tc(cnt, constant_int2tc(utype, BigInt(assert_depth))));

      // ---- THE NON-VACUITY WITNESS, emitted BEFORE any candidate ----
      //
      // Only the antecedent, at pi's own exit. REFUTED means some execution
      // admitted by the region walks THIS path, which is the property every
      // candidate below is conditioned on. Anything else means the region is
      // semantically empty and the whole ladder holds for want of an execution.
      // See the header for why the syntactic gates cannot see this.
      {
        const std::string nv_comment = id2string(f_it->first) +
                                       ":path:" + std::to_string(assert_enc) +
                                       "#nonvacuous";
        const std::string nv_loc = exit_pc->location.as_string();
        all_claims.insert({nv_comment, nv_loc});
        path_cov_assert_nonvacuous_key = {nv_comment, nv_loc};
        insert_assert(goto_program, exit_pc, not_this_path, nv_comment);
      }

      size_t emitted = 0, vars_emitted = 0;
      auto emit_rung = [&](
                         const std::string &var,
                         const std::string &rung,
                         const std::string &text,
                         const expr2tc &cand) {
        // THE COMMENT SHAPE IS A HARD CONSTRAINT: `<unit-id>:path:<enc>` with
        // the unit id FIRST and nothing in front of it, the candidate id a
        // SUFFIX. MEASURED on the certify side: a leading prefix made the
        // report's `path_function` unparseable, the counterexample harvest then
        // filed every nondet as harness-internal, `inputs` came back empty, and
        // the verdict still printed correctly -- an entirely silent loss.
        const std::string comment = id2string(f_it->first) +
                                    ":path:" + std::to_string(assert_enc) +
                                    "#" + rung + "_" + var;
        const std::string loc = exit_pc->location.as_string();
        // all_claims FIRST, before the insert -- the ordering every other
        // branch uses, and what makes the claim visible to
        // audit_entry_liveness.
        if (!all_claims.insert({comment, loc}).second)
        {
          log_error(
            "--path-cov-assert: INTERNAL DEFECT -- duplicate claim key '{}' at "
            "{}. Claim keys are a set, so one of the two candidates would be "
            "silently dropped and would read in the table as a candidate "
            "nobody asked for",
            comment,
            loc);
          abort();
        }
        path_cov_assert_candidates.push_back(
          {assert_enc, var, rung, text, {comment, loc}});
        insert_assert(
          goto_program, exit_pc, or2tc(not_this_path, cand), comment);
        ++emitted;
      };

      for (const auto &comp : to_struct_type(ostruct).components())
      {
        std::string vname = comp.get("#base_name").as_string();
        if (vname.empty())
          vname = comp.get_name().as_string();
        if (!path_cov_user_state_name(vname))
          continue;
        const assert_vart *spec = nullptr;
        for (const auto &v : assert_vars)
          if (v.name == vname)
            spec = &v;
        if (comp_vars_present && spec == nullptr)
          continue; // an explicit `vars` list is a whitelist
        if (spec != nullptr)
          named_seen.insert(vname);

        expr2tc live = symbol2tc(migrate_type(ostruct), obj->id);
        if (!walk_fields(ns, live, vname))
        {
          path_cov_refused_coords[vname] =
            "the component does not resolve through the contract object's "
            "field walk, so no post-state expression can be built for it";
          continue;
        }
        const type2tc vt = live->type;

        // ---- (F): coord_expressible is the EQUALITY gate, NOT the interval
        // one
        //
        // The two must be computed in THIS order and never collapsed. Before
        // S5, `coord_expressible` refused bool and this read
        // `interval_ok = coord_expressible(...)`, `equality_ok = interval_ok ||
        // is_bool_type(vt)` -- which was correct only for as long as the
        // whitelist kept bool out. The moment S5 widened the whitelist,
        // `interval_ok` became TRUE for a bool, the `if (!interval_ok)
        // continue` below stopped firing, and the four ordering rungs were
        // built as
        // `>=` / `<=` / `>` / `<` over a bool -- which lands in the
        // `assert(is_signedbv_type(...))` arms of smt_conv (2494 / 2525 / 2556
        // / 2587) and SIGABRTs. Widening the whitelist WITHOUT this split turns
        // a deliberately correct path into the exact crash the whitelist exists
        // to prevent, so the two edits are one edit.
        //
        // `post == pre` / `post != pre` remain perfectly expressible on a bool,
        // and they are the class a flag-setting function is entirely about, so
        // the equality rungs are still emitted.
        std::string why;
        const bool equality_ok = coord_expressible(vt, why);
        const bool interval_ok = equality_ok && !is_bool_type(vt);
        if (!equality_ok)
        {
          path_cov_refused_coords[vname] = why;
          continue;
        }
        if (!interval_ok)
          path_cov_refused_coords[vname + " [ordering/interval rungs]"] =
            // `why` is EMPTY when coord_expressible accepted the type, which is
            // now the bool case -- the only way to reach here with equality_ok.
            // Printing an empty reason would read as "refused, cause unknown".
            (why.empty()
               ? std::string(
                   "it resolves to a BOOLEAN -- a two-point domain has no "
                   "ordering to measure, and `post >= pre` built on a bool "
                   "operand reaches the signedbv-asserting arm of the SMT "
                   "conversion rather than a comparison")
               : why) +
            ". The equality rungs (post == pre / post != pre) ARE emitted for "
            "it -- only the ordering, interval and delta rungs are not";

        // ---- pre_v: the entry snapshot ----
        //
        // From the SAME member expression the exit read uses. Without it the
        // assertion at the exit would compare the post-state with itself.
        // `.location.property("skipped")` is load bearing. Plain list insert,
        // never insert_swap: insert_swap moves the instruction's CONTENT, so
        // the iterator naming the original first instruction ends up naming the
        // new one and the function acquires a self-loop (measured, ABI gate).
        symbolt ssym;
        ssym.set_type(migrate_type_back(vt));
        ssym.name = "__ESBMC_pre$" + i2string(ghost_counter++);
        ssym.id = "path_cov::" + id2string(ssym.name);
        ssym.lvalue = true;
        ssym.static_lifetime = false;
        ssym.is_extern = false;
        symbolt *psn;
        cov_context->move(ssym, psn);
        expr2tc pre_v = symbol2tc(migrate_type(psn->get_type()), psn->id);
        goto_programt::instructiont dcl;
        dcl.type = DECL;
        dcl.code = code_decl2tc(vt, psn->id);
        dcl.location = entry->location;
        dcl.location.property("skipped");
        dcl.function = entry->location.get_function();
        goto_program.instructions.insert(entry, dcl);
        goto_programt::instructiont asg;
        asg.type = ASSIGN;
        asg.code = code_assign2tc(pre_v, live);
        asg.location = entry->location;
        asg.location.property("skipped");
        asg.function = entry->location.get_function();
        goto_program.instructions.insert(entry, asg);
        ++vars_emitted;

        // R1 -- the equality rungs. Emitted as a PAIR, always, and that is the
        // one thing this mode can testify to on its own: the two are
        // necessarily opposite, so a run in which both HOLD is a run in which
        // the exit read is not observing the unit's writes, and a run in which
        // both are REFUTED is one in which the antecedent never matched.
        emit_rung(vname, "eq", "post == pre", equality2tc(live, pre_v));
        emit_rung(vname, "ne", "post != pre", notequal2tc(live, pre_v));

        if (!interval_ok)
          continue;

        emit_rung(vname, "ge", "post >= pre", greaterthanequal2tc(live, pre_v));
        emit_rung(vname, "le", "post <= pre", lessthanequal2tc(live, pre_v));
        emit_rung(vname, "gt", "post > pre", greaterthan2tc(live, pre_v));
        emit_rung(vname, "lt", "post < pre", lessthan2tc(live, pre_v));

        auto require_fits = [&](const char *what, const std::string &dec) {
          std::string tmax;
          if (path_cov_fits_type(vt, dec, tmax))
            return;
          log_error(
            "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: variable "
            "'{}' {} value {} does not fit its own type (admissible range [0, "
            "{}]). The bound is built as a constant of that type, so an "
            "out-of-range decimal WRAPS and the candidate asserted would not "
            "be "
            "the candidate written here",
            uid,
            vname,
            what,
            dec,
            tmax);
          exit(1);
        };

        if (spec != nullptr && spec->has_abs)
        {
          require_fits("abs_lo", spec->abs_lo);
          require_fits("abs_hi", spec->abs_hi);
          emit_rung(
            vname,
            "abs",
            "post in [" + spec->abs_lo + ", " + spec->abs_hi + "]",
            and2tc(
              greaterthanequal2tc(
                live, constant_int2tc(vt, string2integer(spec->abs_lo))),
              lessthanequal2tc(
                live, constant_int2tc(vt, string2integer(spec->abs_hi)))));
        }
        if (spec != nullptr && spec->has_delta)
        {
          require_fits("delta_lo", spec->delta_lo);
          require_fits("delta_hi", spec->delta_hi);
          // ---- THE DIRECTION CONJUNCT IS NOT DECORATION ----
          //
          // Candidate variables are unsigned, so `post - pre` WRAPS when the
          // value decreased: a decrease of d shows up as 2^w - d. A naive
          // `lo <= post - pre <= hi` therefore HOLDS on a decreasing path
          // whenever the wrapped difference lands in the window -- and for the
          // wide window a driver writes first, on EVERY decreasing path.
          const expr2tc d = spec->delta_dir == "inc" ? sub2tc(vt, live, pre_v)
                                                     : sub2tc(vt, pre_v, live);
          const expr2tc dir = spec->delta_dir == "inc"
                                ? greaterthanequal2tc(live, pre_v)
                                : greaterthanequal2tc(pre_v, live);
          emit_rung(
            vname,
            "delta",
            (spec->delta_dir == "inc" ? std::string("post - pre in [")
                                      : std::string("pre - post in [")) +
              spec->delta_lo + ", " + spec->delta_hi + "] with " +
              (spec->delta_dir == "inc" ? "post >= pre" : "pre >= post"),
            and2tc(
              dir,
              and2tc(
                greaterthanequal2tc(
                  d, constant_int2tc(vt, string2integer(spec->delta_lo))),
                lessthanequal2tc(
                  d, constant_int2tc(vt, string2integer(spec->delta_hi))))));
        }
      }

      // ---- MAPPING SLOTS AS OBSERVABLES ----
      //
      // The loop above can only ever reach a COMPONENT of the contract object,
      // and a mapping is not one. That is where every mapping-valued contract
      // stopped, by name: `NOT ONE candidate assertion could be formed`,
      // because the only state those contracts have is a mapping.
      //
      // What is inexpressible is the WHOLE mapping. ONE SLOT of it is an
      // ordinary scalar, and at THIS layer it is literally an array index --
      // MEASURED on notes/coverage/poc/P28_MapMin.sol with
      // --goto-functions-only, where `bal[k] = v` lowers to
      //
      //     ASSIGN bal[k]=v;
      //     ASSIGN bal[k]=bal[k] - v;
      //
      // with `bal` the contract-scope global recorded in `store_syms` above and
      // `k` the unit's own parameter. No hashing, no slot arithmetic, nothing
      // to reimplement: the rungs, the pre-snapshot and the antecedent are byte
      // for byte the ones the component loop emits.
      //
      // ⛔ ONLY A NAMED SLOT, never a default sweep. Which key a unit's oracle
      // is about is a fact about the unit; walking every mapping with every
      // parameter as a key would emit rungs about slots the unit never touches.
      // Those HOLD -- correctly, and vacuously as far as the unit is concerned
      // -- and would be rendered downstream as an `assertEq(post, pre)` on an
      // unrelated entry, which is an oracle that can never fail.
      //
      // ---- THE KEY IS SNAPSHOTTED AT ENTRY, AND THAT IS NOT OPTIONAL ----
      //
      // The pre-read happens at entry and the post-read at the exit. A
      // parameter may be reassigned in between, so reading the key LIVE at the
      // exit would compare slot `bal[k_entry]` with slot `bal[k_exit]` -- two
      // different slots, reported as one variable's before and after. Both
      // reads therefore index with the same entry ghost. This is the same
      // reason the outer-box batch snapshots its coordinates.
      for (const auto &v : assert_vars)
      {
        const size_t ob = v.name.find('[');
        if (ob == std::string::npos || v.name.empty() || v.name.back() != ']')
          continue; // not a slot spelling; the component loop owns this name
        const std::string mname = v.name.substr(0, ob);
        const std::string kname = v.name.substr(ob + 1, v.name.size() - ob - 2);
        if (mname.empty() || kname.empty())
        {
          log_error(
            "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: \"vars\" "
            "names '{}', which is not a well-formed slot. The shape is "
            "`<mapping>[<key>]` with both parts non-empty",
            uid,
            v.name);
          exit(1);
        }
        named_seen.insert(v.name);

        auto sit = store_syms.find(mname);
        if (sit == store_syms.end())
        {
          // NAMED, and the available stores are listed. "It does not exist"
          // and "it is a component, so drop the brackets" need different
          // fixes, and a bare refusal sends the reader to the spelling in
          // both cases.
          std::string avail;
          for (const auto &[n, s] : store_syms)
            avail += (avail.empty() ? "" : ", ") + n;
          log_error(
            "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: \"vars\" "
            "names the slot '{}', but '{}' is not a contract-scope store of "
            "contract '{}'. The stores available are: {}. A SCALAR state "
            "variable is a component of the contract object and is named "
            "WITHOUT brackets",
            uid,
            v.name,
            mname,
            own_contract,
            avail.empty() ? "<none>" : avail);
          exit(1);
        }

        const type2tc mt = migrate_type(sit->second->get_type());
        if (!is_array_type(mt))
        {
          log_error(
            "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: '{}' is a "
            "contract-scope store but does not lower to an ARRAY, so `{}` "
            "denotes no slot. Refused rather than skipped: a dropped candidate "
            "reads in the table as a variable nothing needed to be asserted "
            "about",
            uid,
            mname,
            v.name);
          exit(1);
        }
        const type2tc et = to_array_type(mt).subtype;

        // Same two-gate split as the component loop at (F): the whitelist is
        // the EQUALITY gate, and the ordering rungs additionally require a
        // non-bool. Collapsing them builds `>=` on a bool operand, which is a
        // SIGABRT in smt_conv rather than a comparison.
        std::string ewhy;
        const bool eq_ok = coord_expressible(et, ewhy);
        const bool iv_ok = eq_ok && !is_bool_type(et);
        if (!eq_ok)
        {
          path_cov_refused_coords[v.name] =
            "the mapping's VALUE type cannot carry a candidate: " + ewhy;
          continue;
        }

        expr2tc kexpr;
        std::string kwhy;
        if (!resolve_coord(fsym, kname, kexpr))
          kwhy =
            "the key does not resolve to an input of this unit. Name a "
            "parameter, an environment value as `msg.sender` / `msg.value`, or "
            "a state variable at entry as `state.<field>`";
        else
          coord_expressible(kexpr->type, kwhy);
        if (!kwhy.empty())
        {
          // REFUSE THE LADDER, the certify disposition. A slot whose key could
          // not be built is not a weaker observable, it is a different one:
          // there is no expression to read, so continuing would silently drop
          // the candidate the spec asked for.
          log_error(
            "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: the slot "
            "'{}' names the key '{}', which cannot be expressed: {}",
            uid,
            v.name,
            kname,
            kwhy);
          exit(1);
        }

        // The key ghost, then the slot's entry snapshot. Both plain list
        // inserts before `entry`, never insert_swap -- insert_swap moves the
        // instruction's CONTENT and the iterator would end up naming the new
        // instruction, which is how the ABI gate acquired a self-loop once.
        symbolt ksym;
        ksym.set_type(migrate_type_back(kexpr->type));
        ksym.name = "__ESBMC_key$" + i2string(ghost_counter++);
        ksym.id = "path_cov::" + id2string(ksym.name);
        ksym.lvalue = true;
        ksym.static_lifetime = false;
        ksym.is_extern = false;
        symbolt *pk;
        cov_context->move(ksym, pk);
        expr2tc key_v = symbol2tc(migrate_type(pk->get_type()), pk->id);
        {
          goto_programt::instructiont kd;
          kd.type = DECL;
          kd.code = code_decl2tc(kexpr->type, pk->id);
          kd.location = entry->location;
          kd.location.property("skipped");
          kd.function = entry->location.get_function();
          goto_program.instructions.insert(entry, kd);
          goto_programt::instructiont ka;
          ka.type = ASSIGN;
          ka.code = code_assign2tc(key_v, kexpr);
          ka.location = entry->location;
          ka.location.property("skipped");
          ka.function = entry->location.get_function();
          goto_program.instructions.insert(entry, ka);
        }

        // The slot itself, read through the ENTRY key on both sides.
        expr2tc marr = symbol2tc(mt, sit->second->id);
        expr2tc live = index2tc(et, marr, key_v);

        symbolt ssym;
        ssym.set_type(migrate_type_back(et));
        ssym.name = "__ESBMC_pre$" + i2string(ghost_counter++);
        ssym.id = "path_cov::" + id2string(ssym.name);
        ssym.lvalue = true;
        ssym.static_lifetime = false;
        ssym.is_extern = false;
        symbolt *ps;
        cov_context->move(ssym, ps);
        expr2tc pre_v = symbol2tc(migrate_type(ps->get_type()), ps->id);
        {
          goto_programt::instructiont pd;
          pd.type = DECL;
          pd.code = code_decl2tc(et, ps->id);
          pd.location = entry->location;
          pd.location.property("skipped");
          pd.function = entry->location.get_function();
          goto_program.instructions.insert(entry, pd);
          goto_programt::instructiont pa;
          pa.type = ASSIGN;
          pa.code = code_assign2tc(pre_v, live);
          pa.location = entry->location;
          pa.location.property("skipped");
          pa.function = entry->location.get_function();
          goto_program.instructions.insert(entry, pa);
        }
        ++vars_emitted;

        // The equality PAIR, always, for the reason the component loop states:
        // the two are necessarily opposite, so a run in which BOTH hold is a
        // run in which the exit read is not observing this unit's writes --
        // which on a slot is the failure to watch for, since an entry ghost
        // holding the wrong key would produce exactly that.
        emit_rung(v.name, "eq", "post == pre", equality2tc(live, pre_v));
        emit_rung(v.name, "ne", "post != pre", notequal2tc(live, pre_v));
        if (!iv_ok)
        {
          path_cov_refused_coords[v.name + " [ordering/interval rungs]"] =
            "the mapping's value type is a BOOLEAN -- a two-point domain has "
            "no ordering to measure. The equality rungs ARE emitted for it";
          continue;
        }
        emit_rung(
          v.name, "ge", "post >= pre", greaterthanequal2tc(live, pre_v));
        emit_rung(v.name, "le", "post <= pre", lessthanequal2tc(live, pre_v));
        emit_rung(v.name, "gt", "post > pre", greaterthan2tc(live, pre_v));
        emit_rung(v.name, "lt", "post < pre", lessthan2tc(live, pre_v));

        auto slot_fits = [&](const char *what, const std::string &dec) {
          std::string tmax;
          if (path_cov_fits_type(et, dec, tmax))
            return;
          log_error(
            "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: slot '{}' "
            "{} value {} does not fit the mapping's value type (admissible "
            "range [0, {}]). The bound is built as a constant of that type, so "
            "an out-of-range decimal WRAPS and the candidate asserted would "
            "not "
            "be the candidate written here",
            uid,
            v.name,
            what,
            dec,
            tmax);
          exit(1);
        };
        if (v.has_abs)
        {
          slot_fits("abs_lo", v.abs_lo);
          slot_fits("abs_hi", v.abs_hi);
          emit_rung(
            v.name,
            "abs",
            "post in [" + v.abs_lo + ", " + v.abs_hi + "]",
            and2tc(
              greaterthanequal2tc(
                live, constant_int2tc(et, string2integer(v.abs_lo))),
              lessthanequal2tc(
                live, constant_int2tc(et, string2integer(v.abs_hi)))));
        }
        if (v.has_delta)
        {
          slot_fits("delta_lo", v.delta_lo);
          slot_fits("delta_hi", v.delta_hi);
          // The direction conjunct is not decoration, for the reason the
          // component loop records: the value type is unsigned, so `post - pre`
          // WRAPS on a decrease and a bare window HOLDS on every decreasing
          // path.
          const expr2tc d = v.delta_dir == "inc" ? sub2tc(et, live, pre_v)
                                                 : sub2tc(et, pre_v, live);
          const expr2tc dir = v.delta_dir == "inc"
                                ? greaterthanequal2tc(live, pre_v)
                                : greaterthanequal2tc(pre_v, live);
          emit_rung(
            v.name,
            "delta",
            (v.delta_dir == "inc" ? std::string("post - pre in [")
                                  : std::string("pre - post in [")) +
              v.delta_lo + ", " + v.delta_hi + "] with " +
              (v.delta_dir == "inc" ? "post >= pre" : "pre >= post"),
            and2tc(
              dir,
              and2tc(
                greaterthanequal2tc(
                  d, constant_int2tc(et, string2integer(v.delta_lo))),
                lessthanequal2tc(
                  d, constant_int2tc(et, string2integer(v.delta_hi))))));
        }
      }

      // ---- THE UNIT'S OWN RETURN VALUE IS A CANDIDATE TOO ----
      //
      // WHY IT HAS TO GO THROUGH THE LADDER, rather than straight from the
      // report into an assertEq. The report's `return_value` is the value on
      // ONE counterexample point. The test the emitter writes fuzzes the whole
      // region, so a point value asserted directly is RED across the fuzz runs
      // -- measured on aqua, whose PUT fuzzes `maker`/`app`/`token` over the
      // entire address space while the payload names one triple. Only a rung
      // the verifier judged HOLDS over the ASSUMED region may become an
      // assertion, which is exactly the contract the state rungs above already
      // satisfy.
      //
      // RESERVED NAME. The candidate is called `return`, which is a Solidity
      // KEYWORD and therefore cannot be the name of a state variable -- so the
      // whitelist below and `path_cov_refused_coords` can hold it beside real
      // variables with no possibility of collision, and the rung keys
      // (`reteq0_return` ...) cannot collide with a state rung (`eq_<var>`).
      //
      // SCALARS ONLY, and the gap is NAMED rather than silent: a tuple return
      // emits no RETURN instruction at all (measured on
      // notes/coverage/poc/P27_TupleReturn.sol), so no ghost exists for it. A
      // non-void unit with no ghost is recorded as a REFUSAL, because an absent
      // row in this table otherwise reads as "no assertion was needed".
      const size_t emitted_before_return = emitted;
      {
        const bool ret_wanted =
          !comp_vars_present ||
          std::any_of(
            assert_vars.begin(), assert_vars.end(), [](const assert_vart &v) {
              return v.name == "return" || v.name.rfind("return.", 0) == 0;
            });
        const assert_vart *rspec = nullptr;
        for (const auto &v : assert_vars)
          if (v.name == "return")
            rspec = &v;
        if (rspec != nullptr)
          named_seen.insert("return");

        // ---- "DOES THIS UNIT RETURN SOMETHING?" IS NOT ANSWERED BY ITS TYPE
        //
        // MEASURED on P27_TupleReturn.two_scalars (`returns (uint256,
        // uint256)`): `to_code_type(fsym->type).return_type().id()` reads
        // `empty` -- byte for byte what a void unit reads -- and
        // `#sol_ast_return_sites` is 0 as well, which is why the AST-half
        // census above does not fire on it either. Both of the obvious tests
        // would therefore call a tuple-returning unit VOID and record no
        // refusal at all.
        //
        // The positive evidence is the frontend's own lowering: a tuple return
        // writes into a contract-scope `tuple_instance$<node-id>` object keyed
        // by THIS unit's AST node id (`two_scalars#42` owns
        // `tuple_instance$42`). That is the same key bmc.cpp's harvest uses to
        // tie a tuple to its unit, so the two cannot drift; and being keyed on
        // the node id, an inlined callee's tuple cannot be mistaken for this
        // unit's.
        bool has_tuple_instance = false;
        {
          const size_t hash = uid.rfind('#');
          if (hash != std::string::npos)
          {
            const std::string want = "sol:@C@" + own_contract +
                                     "@tuple_instance$" + uid.substr(hash + 1);
            cov_context->foreach_operand([&](const symbolt &s) {
              const std::string id = s.id.as_string();
              if (id.rfind(want, 0) != 0)
                return;
              // `tuple_instance$4` must not answer for `tuple_instance$42`.
              const std::string rest = id.substr(want.size());
              if (rest.empty() || rest[0] == '#')
                has_tuple_instance = true;
            });
          }
        }

        // Only when NEITHER shape produced a ghost. A tuple whose members ARE
        // scalars now has one ghost per member and is not refused; a tuple of
        // aggregates still is, and so is any other shape the instrumenter
        // cannot materialise.
        if (!has_ret_ghost && ret_member_ghosts.empty() && has_tuple_instance)
          path_cov_refused_coords["return"] =
            "the unit returns a value, but no return ghost could be built for "
            "it. A tuple return emits no RETURN instruction carrying a single "
            "renderable operand, and not one of its members is a scalar the "
            "per-member ghosts can hold either, so there is nothing at the "
            "exit to assert about. Its absence from this table "
            "is a REFUSAL, not a measurement that the value is unconstrained";
        // A tuple SOME of whose members are unusable: the usable ones get
        // rungs, and the rest are named one by one. A partially-covered tuple
        // reported as if it were fully covered is the same misreading as a
        // wholly absent row, one member down.
        for (const auto &[k, why] : ret_member_refused)
          path_cov_refused_coords["return." + std::to_string(k)] = why;

        const bool any_ret = has_ret_ghost || !ret_member_ghosts.empty();
        if (any_ret && ret_wanted)
        {
          // ---- THE RETURN-VALUE NON-VACUITY WITNESS ----
          //
          // Every rung below carries `|| !retset` so that it says "IF a value
          // was returned on this execution, THEN ...". Without that, a path
          // reaching this exit WITHOUT executing a RETURN reads the entry
          // initialisation 0 and `return == 0` HOLDS -- an assertion certified
          // about a value the execution never produced. (This is the same
          // failure the retset ghost was added for on the report side, arriving
          // through the ladder instead.)
          //
          // The price of that guard is that the whole family can hold
          // VACUOUSLY, so the guard needs its own witness: `retlive` asserts
          // `!retset` and is REFUTED exactly when some execution of this path
          // does return a value. A driver that sees `retlive` anything other
          // than REFUTED must discard every other `ret*` rung of this run --
          // they hold for want of a returned value, not because the value is
          // constrained.
          const expr2tc no_ret = gen_not_expr(retset_ghost);
          emit_rung(
            "return",
            "retlive",
            "a value IS returned on this path (REFUTED == yes)",
            no_ret);

          // ONE emitter for both shapes. A scalar return and a tuple MEMBER
          // differ only in which ghost they read and what the candidate is
          // called; writing the rung set twice is how the two would come to
          // disagree about which rungs exist -- and a member silently missing
          // one rung reads, downstream, exactly like a member on which that
          // rung was REFUTED.
          auto emit_value_rungs = [&](
                                    const std::string &vname,
                                    const expr2tc &g,
                                    const assert_vart *sp) {
            if (is_bool_type(g->type))
            {
              // No constant_int2tc on a bool, and no ordering: the same rule
              // the state side documents at (F). The two-point domain makes
              // the equality pair exhaustive, which is all a bool needs.
              emit_rung(
                vname,
                "reteq0",
                "return == false",
                or2tc(no_ret, equality2tc(g, gen_false_expr())));
              emit_rung(
                vname,
                "retne0",
                "return == true",
                or2tc(no_ret, equality2tc(g, gen_true_expr())));
              return;
            }
            // The zero pair is emitted with NO spec and is the rung that
            // actually pays: a view function read on a freshly deployed
            // contract returns 0 over the whole region, and `assertEq(v, 0)` is
            // a real post-condition rather than a restatement of the input.
            // Necessarily opposite whenever a value was returned, so a run in
            // which BOTH hold is a run in which none of them was reached --
            // which `retlive` reports directly.
            const type2tc rt = g->type;
            emit_rung(
              vname,
              "reteq0",
              "return == 0",
              or2tc(no_ret, equality2tc(g, gen_zero(rt))));
            emit_rung(
              vname,
              "retne0",
              "return != 0",
              or2tc(no_ret, notequal2tc(g, gen_zero(rt))));

            if (sp != nullptr && sp->has_abs)
            {
              for (const auto &[what, dec] :
                   std::vector<std::pair<std::string, std::string>>{
                     {"abs_lo", sp->abs_lo}, {"abs_hi", sp->abs_hi}})
              {
                std::string tmax;
                if (path_cov_fits_type(rt, dec, tmax))
                  continue;
                log_error(
                  "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: the "
                  "\"{}\" {} value {} does not fit the unit's own return "
                  "type (admissible range [0, {}]). The bound is built as a "
                  "constant of that type, so an out-of-range decimal WRAPS and "
                  "the candidate asserted would not be the candidate written "
                  "here",
                  uid,
                  vname,
                  what,
                  dec,
                  tmax);
                exit(1);
              }
              emit_rung(
                vname,
                "retabs",
                "return in [" + sp->abs_lo + ", " + sp->abs_hi + "]",
                or2tc(
                  no_ret,
                  and2tc(
                    greaterthanequal2tc(
                      g, constant_int2tc(rt, string2integer(sp->abs_lo))),
                    lessthanequal2tc(
                      g, constant_int2tc(rt, string2integer(sp->abs_hi))))));
            }
          };

          if (has_ret_ghost)
            emit_value_rungs("return", ret_ghost, rspec);
          // Members are named `return.<k>` in DECLARATION order, which is the
          // order solc gives them and therefore the order a destructuring
          // `(a, b) = f(...)` binds them in. The index is part of the name and
          // not of the rung TEXT, so the emitter's text parser is the same one
          // for both shapes and the member identity lives in exactly one place.
          for (const auto &[k, g] : ret_member_ghosts)
          {
            const std::string mv = "return." + std::to_string(k);
            const assert_vart *msp = nullptr;
            for (const auto &v : assert_vars)
              if (v.name == mv)
                msp = &v;
            if (msp != nullptr)
              named_seen.insert(mv);
            emit_value_rungs(mv, g, msp);
          }
        }
      }

      // A named variable that matched nothing: the ladder is short and nothing
      // would say so.
      for (const auto &w : named_wanted)
        if (named_seen.count(w) == 0)
        {
          log_error(
            "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: \"vars\" "
            "names '{}', which is not a scalar component of this contract's "
            "instance object. Either it does not exist, or it is a mapping / "
            "dynamic array. Emitting a SHORTER ladder would answer a different "
            "question than the one the spec asked",
            uid,
            w);
          exit(1);
        }

      log_status(
        "--path-cov-assert: unit '{}' -- assumed {} region bound(s) ({} "
        "hole(s) "
        "punched) at entry and emitted {} candidate assertion(s), {} of them "
        "over the unit's own RETURN VALUE and the rest over {} state "
        "variable(s), at path enc={} depth={}'s OWN exit. Every candidate "
        "carries the antecedent `tr != {} || cnt != {}`, so at any other exit "
        "and on any other execution it is vacuous -- which is what lets the "
        "whole ladder be judged in ONE run instead of one query per candidate",
        uid,
        bounds_emitted,
        holes_emitted,
        emitted,
        emitted - emitted_before_return,
        vars_emitted,
        assert_enc,
        assert_depth,
        assert_enc,
        assert_depth);
      total_paths += emitted;
      continue;
    }

    size_t ins_idx = 0;
    for (auto &[pc, g, comment, is_revert, stable_id] : to_insert)
    {
      const size_t this_idx = ins_idx++;
      // Claim key == the (comment, location) pair get_total_cond_assert() and
      // bmc.cpp's claim_sig use, so universe / covered-set / numerator stay
      // key-aligned. insert_assert copies pc->location onto the new assert,
      // so reading it here (pre-insert) gives the same string.
      const std::string loc = pc->location.as_string();
      const std::pair<std::string, std::string> key(comment, loc);
      // Static universe FIRST: every enumerated path counts in the
      // denominator whether or not it is instrumented this round.
      all_claims.insert(key);
      // exit_kind for the report: this path leaves via a detected
      // custom-error revert rather than the normal END_FUNCTION exit.
      if (is_revert)
        revert_paths.insert(key);
      // ...or via a require/revert("msg") rollback, which reaches END_FUNCTION
      // but still reverts the transaction.
      if (rollback_exits.count(this_idx))
        rollback_revert_paths.insert(key);
      if (undetermined_exits.count(this_idx))
        undetermined_exit_paths.insert(key);
      // POSITIVE record of a normal exit. The other three sets already name
      // what went wrong; without this one, "normal" could only be inferred from
      // absence — and a consumer that infers it would also call every claim
      // from a DIFFERENT coverage mode normal, since those appear in no set at
      // all. Measured, not hypothetical: reading normality as absence broke
      // three branch-coverage regressions on the first run.
      //
      // Recorded here rather than derived, for the same reason the census
      // demands positive evidence in the first place: this is the one judgement
      // that authorises a generated test to assert something.
      //
      // ...AND the unit is not disqualified. An obstacle means the model admits
      // an execution the chain does not have, so this path's counterexample can
      // describe something that cannot happen -- precisely the case in which a
      // test must NOT be authorised to assert. The two flags are per-unit and
      // already computed above, so the guard is free.
      //
      // WITHOUT IT the two records contradicted each other: the same key went
      // into `named_obstacle_paths` ("must not be turned into a test", per the
      // header) three lines below, and into this set ("the one judgement that
      // authorises a generated test to assert something") right here. The
      // emitter reads this set and never read that map, so the authorisation
      // won -- an obstructed path was emitted BARE, with a comment saying the
      // path exits normally and a revert would fail the test.
      //
      // The emitter now refuses obstructed cases outright (foundry.cpp), which
      // is the primary fix; this is the same rule stated where the
      // authorisation is granted, so a future consumer of this set inherits it
      // without having to know about the other one.
      if (
        !is_revert && !rollback_exits.count(this_idx) &&
        !undetermined_exits.count(this_idx) && !unit_has_lost_decision &&
        !unit_calls_gated_unit)
        normal_exit_paths.insert(key);
      // Named obstacle, applied to EVERY path of the unit (see the census
      // comment above for why per-path containment is unsound here). The paths
      // stay in the denominator — they are real — but none of them may become a
      // test or join the sibling set for the stage-3 subtraction.
      if (unit_has_lost_decision)
      {
        ++obstacle_paths_assume;
        named_obstacle_paths[key] =
          "unit contains a source-level decision the frontend lowered to a "
          "control-flow-free assume (internal library / free-function "
          "require); the reverting execution does not exist in the model, so "
          "it is absent from the sibling set of EVERY path of this unit";
      }
      if (unit_calls_gated_unit)
      {
        ++obstacle_paths_residual;
        named_obstacle_paths[key] =
          "unit still calls another UNIT's own body unexpanded (" +
          residual_unit_names +
          "); that body carries the ABI value gate, which models an EXTERNAL "
          "entry, so the model admits the callee reverting for carrying value "
          "inside an INTERNAL call that on-chain proceeds — an execution that "
          "does not exist on chain";
      }
      if (unit_truncated)
      {
        // NOT an obstacle: a strength annotation. The dropped paths exist in
        // the model and their Phase-1 accounting is still instrumented, so the
        // certification query `assume(interval); assert(tr == pi)` still
        // rejects any candidate interval that reaches them — the interval
        // shrinks rather than being wrong. Recorded per path so a downstream
        // consumer can say "this one's certified region is narrower than it
        // would otherwise be", and deliberately kept OUT of
        // named_obstacle_paths so it cannot be confused with the model/reality
        // divergences that must not ship a test.
        ++truncation_weakened_paths;
        truncation_weakened[key] =
          "unit lost paths to the per-unit goal/length cap; the dropped paths "
          "exist in the model and are still accounted for in `tr`, so the "
          "certification query still excludes their inputs — this narrows the "
          "certified region rather than invalidating it";
      }
      // Remember this run's claim key -> stable id so the run-end write-back
      // can persist exactly the paths that were witnessed.
      path_stable_id[key] = stable_id;
      if (path_covered_ids.count(stable_id))
      {
        ++skipped_paths; // already witnessed in an earlier round
        continue;
      }
      insert_assert(goto_program, pc, g, comment);
      // Stamp the just-inserted assert (now at std::prev(pc)) so the Foundry
      // generator emits vm.expectRevert() for this detected revert path (R0).
      //
      // ONLY for `is_revert`, whose assert sits at the `#sol_error` call — an
      // instruction reachable on that path ALONE. A rollback revert's assert
      // sits at the shared END_FUNCTION, where every path's assert is stacked:
      // the generator marks a transaction as reverting when ANY reached assert
      // step carries the flag, so stamping there makes a NON-reverting path's
      // counterexample pick the flag up and emit `vm.expectRevert()` before a
      // call that does not revert — a test that fails when run. Measured: all 3
      // of D's tests (two of them normal paths) got the wrapper.
      // The JSON already carries `exit_kind: "revert"` for these, so a
      // generator can emit the oracle from there without this bleed.
      if (is_revert)
        std::prev(pc)->location.set("sol_revert_edge", true);
      ++total_paths;
    }
  }

  // ---- A --focus-function THAT MATCHED NO UNIT IS NOT A MEASUREMENT ----
  //
  // Same shape as the certify/assert route-5 gates below. Narrowing
  // instrumentation is what makes the symptom possible at all: before it, a
  // focus naming nothing still instrumented the whole contract and printed a
  // full report; now it would instrument nothing, and
  //
  //     Complete Paths : 0 / No complete path enumerated
  //
  // is byte-compatible with a contract that genuinely has no unit, so a
  // per-method sweep would record a clean zero for a name it got wrong.
  //
  // ---- THIS IS A SECOND LINE, AND IT HAS NO REPRODUCER TODAY ----
  //
  // Stated plainly rather than implied, because a guard described as if it
  // fires is one nobody re-checks. Both routes I could construct are closed
  // EARLIER, and both were measured rather than reasoned about:
  //
  //  * a misspelled name never reaches here. The frontend validator
  //    (solidity_convert.cpp, run at the top of convert()) requires the name to
  //    be a public/external, non-constructor, non-receive/fallback method of
  //    the target contract, and otherwise fails the conversion:
  //        ERROR: --focus-function 'nosuchfn' is not a public/external function
  //               of contract 'C'
  //        ERROR: CONVERSION ERROR            (exit 6)
  //    That is the layer `solidity_path_cov_focus_function_no_match_fails`
  //    pins, because it is the layer that actually enforces the property.
  //
  //  * "the name is right but --contract scoped the unit out" does not occur
  //    either, because Solidity inheritance is merge-BY-COPY here: measured
  //    with `contract D is B` and `--contract D --focus-function basefn`, the
  //    unit is `sol:@C@D@F@basefn#23` -- attributed to D, in scope, enumerated
  //    normally.
  //
  // So this gate exists for a route that does not exist yet: a focus name that
  // passes the frontend validator while its goto function is not an
  // `is_external_entry` unit, or any future caller that sets focus_function
  // without going through that validator. It costs one comparison. The
  // candidate list is printed with it because the two causes above need
  // different fixes and a bare "matched nothing" sends the reader to the
  // spelling in both cases.
  if (!focus_function.empty() && units_enumerated == 0)
  {
    std::string cands;
    for (const auto &c : focus_candidates)
      cands += (cands.empty() ? "" : "; ") + c;
    log_error(
      "--solidity-path-coverage: --focus-function '{}' matched NONE of the {} "
      "unit(s) in scope, so NOT ONE path was enumerated and this run measures "
      "nothing. It would otherwise print an empty coverage report that is "
      "indistinguishable from a contract with no externally-callable function. "
      "Note the frontend already checks the name against --contract '{}' and "
      "fails the conversion when it is not one of its public/external methods, "
      "so reaching HERE means the name passed that check and still selected no "
      "enumerated UNIT — compare it against the list rather than against the "
      "source. The unit(s) that were available: {}",
      focus_function,
      focus_candidates.size(),
      scope_contract.empty() ? "<unset>" : scope_contract,
      cands.empty() ? "<none: no unit reached the focus test at all>" : cands);
    exit(1);
  }
  if (units_skipped_by_focus > 0)
    log_status(
      "--solidity-path-coverage: --focus-function '{}' narrowed "
      "INSTRUMENTATION "
      "to {} unit(s); {} other in-scope unit(s) were not enumerated at all. "
      "Their paths are absent from the denominator ON PURPOSE: the dispatcher "
      "cannot enter them in this run, so no exploration could ever witness "
      "them "
      "and counting them made the reported coverage a contract-level number "
      "wearing a unit-level label. Internal-call EXPANSION still ran for every "
      "unit, so this unit's path identity is unchanged -- a callee's decisions "
      "are still part of it",
      focus_function,
      units_enumerated,
      units_skipped_by_focus);

  // ---- A CERTIFICATION QUERY THAT MATCHED NO UNIT IS NOT A CERTIFICATE ----
  //
  // The FIFTH false-certification route, found by enumerating the assumption
  // side of the gate rather than by colliding with it. `--path-cov-certify`
  // with a unit name nothing matches -- a typo, a renamed function, a driver
  // reading the wrong field -- skips every unit, so NO assume and NO assert are
  // ever emitted. The run then has nothing to check, everything holds for want
  // of an obligation, and it prints VERIFICATION SUCCESSFUL with exit 0.
  //
  // Measured: {"unit": "nosuchfn", ...} on a contract with one unit returned
  // SUCCESSFUL, exit 0, indistinguishable from a real certificate.
  //
  // Same shape as the other four (an inverted interval, a signed coordinate, a
  // punched-empty box, a one-value ladder): the query answers SUCCESSFUL for a
  // question it never asked. Exiting non-zero with a named reason puts it in
  // the third state, where a caller reading whole verdict lines can see it.
  if (certify_on && certify_units_matched == 0)
  {
    log_error(
      "--path-cov-certify: unit '{}' matched NO enumerated unit, so not one "
      "assume and not one assert was emitted -- the run would otherwise print "
      "VERIFICATION SUCCESSFUL for a query it never asked. Check the name "
      "against the report's `path_function` (mangled form "
      "sol:@C@<contract>@F@<name>#<id>). This is a false certificate, not a "
      "weak result",
      certify_unit);
    exit(1);
  }

  // ---- ROUTE 5 FOR STAGE 3: a ladder that matched no unit ----
  //
  // Tested BEFORE the empty-ladder gate below: "nothing matched" and
  // "everything was refused" both produce zero candidates, and answering the
  // first with the second's message sends the reader to look at the contract's
  // state variables when the real problem is a typo in the unit name.
  if (assert_on && assert_units_matched == 0)
  {
    log_error(
      "--path-cov-assert: unit '{}' matched NO enumerated unit, so not one "
      "assumption and not one candidate assertion was emitted -- the run would "
      "otherwise print VERIFICATION SUCCESSFUL for a ladder it never built. "
      "Check the name against the enumeration run's `path_function` (mangled "
      "form sol:@C@<contract>@F@<name>#<id>)",
      assert_unit);
    exit(1);
  }
  // A spec identifying ONE path may not be answered by two units: the two have
  // different bodies, so the two ladders are different claims printed under one
  // heading.
  if (assert_on && assert_units_matched > 1)
  {
    log_error(
      "--path-cov-assert: unit '{}' matched {} enumerated units. A post-state "
      "ladder identifies ONE path of ONE unit; two matches would print two "
      "different sets of claims under one heading. Name the unit in its full "
      "mangled form (sol:@C@<contract>@F@<name>#<id>)",
      assert_unit,
      assert_units_matched);
    exit(1);
  }
  // ---- N1, entry condition (b): every eligible variable was REFUSED ----
  if (assert_on && path_cov_assert_candidates.empty())
  {
    // NAME AND REASON, not just the name. This gate is the LAST thing printed
    // on such a run -- it exits before solving, so the per-candidate table and
    // the "carry NO candidate" warning that normally carry the reasons are
    // never reached. A bare list of names then leaves the reader to guess which
    // of several different refusals applied to which name, and the refusals are
    // not interchangeable: "a mapping lowered to a contract-scope global" and
    // "the unit returns a tuple, which materialises no single value" need
    // different fixes.
    std::string refused;
    for (const auto &rc : path_cov_refused_coords)
      refused +=
        (refused.empty() ? "" : "; ") + rc.first + " (" + rc.second + ")";
    log_error(
      "--path-cov-assert: unit '{}' -- REFUSING THE LADDER: NOT ONE candidate "
      "assertion could be formed. Every candidate this ladder could have "
      "formed "
      "was refused{}{}. Zero assertions means nothing is checked, and the run "
      "would "
      "print VERIFICATION SUCCESSFUL with exit 0 -- the same output a fully "
      "successful ladder produces. A contract whose state is entirely mappings "
      "or dynamic arrays is the common case: those are lowered to "
      "contract-scope globals, not components of the contract object",
      assert_unit,
      refused.empty() ? "" : ": ",
      refused);
    exit(1);
  }

  // all_claims is the no-skip static universe built in the loop above (one
  // entry per enumerated complete path), NOT get_total_cond_assert() — the
  // latter counts instrumented asserts only, so a covered-set skip would
  // shrink the denominator and spuriously inflate coverage.
  if (dropped_paths > 0)
    log_warning(
      "--solidity-path-coverage: per-function path/length cap ({}) hit; {} "
      "path(s) dropped (coverage is complete only up to the cap for those "
      "functions)",
      path_cov_max_goals,
      dropped_paths);
  log_status(
    "--solidity-path-coverage: instrumented {} complete path(s) across {} "
    "unit(s) (loop bound = {} iterations)",
    total_paths,
    units_enumerated,
    path_cov_unwind);
  if (!named_obstacle_paths.empty())
    log_warning(
      "--solidity-path-coverage: NAMED OBSTACLE — {} path(s) excluded, being "
      "ALL paths of every affected unit. Both causes are the SAME failure — "
      "the "
      "model and the chain disagree, so a counterexample can describe an "
      "execution that does not exist and the test built from it is RED on the "
      "UNMODIFIED contract — reached by two different routes, so they are "
      "counted apart:\n"
      "  (a) {} path(s) across {} unit(s): the unit contains a construct that "
      "removes executions WITHOUT a branch — an explicitly written "
      "`__ESBMC_assume`, or a `require` still lowered to a control-flow-free "
      "assume (the library / free-function case; seeing that again means the "
      "revert-observation widening regressed). Those executions do not exist "
      "in "
      "the model at all, so no certification query can even see them.\n"
      "  (b) {} path(s) across {} unit(s): the unit still calls another UNIT's "
      "own body unexpanded (depth bound), routing an INTERNAL call through the "
      "EXTERNAL-entry body and its ABI value gate. Raise --unwind to expand "
      "them; an expanded copy is gate-free.\n"
      "Exclusion is per UNIT, not per path. Reported as an absolute count and "
      "NOT folded into the coverage percentage — an obstacle is not partial "
      "credit",
      named_obstacle_paths.size(),
      obstacle_paths_assume,
      obstacle_units,
      obstacle_paths_residual,
      obstacle_units_residual);
  if (truncation_weakened_paths > 0)
    log_warning(
      "--solidity-path-coverage: ASSERTION STRENGTH — {} path(s) across {} "
      "unit(s) have a NARROWER certified region than they otherwise would, "
      "because their unit lost paths to the goal cap ({}). This is a strength "
      "annotation, NOT an obstacle: the dropped paths exist in the model and "
      "their decision accounting is still instrumented, so an input reaching "
      "one still carries its path number in `tr` and the certification query "
      "`assume(interval); assert(tr == pi)` rejects it — the interval shrinks "
      "rather than being wrong. Degradation runs first precisely so this "
      "should "
      "not be reached; each unit above says why it was",
      truncation_weakened_paths,
      units_at_cap,
      path_cov_max_goals);
  if (units_enumerated > 0)
  {
    // One line carrying everything the distribution question needs, so it never
    // has to be re-run to answer a follow-up.
    //
    // The total here is the ENUMERATED count (all_claims), NOT the count
    // instrumented this round. The distribution is a structural property of the
    // contract, so mixing in a per-round number would make it change between
    // rounds of the same escalation: measured on a resumed run, 7 of 8 paths
    // were carried over from the covered set and this line read "1 path(s)
    // total ... 0.33x" for a contract whose real figures are 8 and 2.67x. `max`
    // and the pre-expansion total were already structural, so the mix was
    // silent — the kind of number that gets quoted.
    const size_t enumerated_total = all_claims.size();
    const double ratio = pre_expansion_total > 0 ? (double)enumerated_total /
                                                     (double)pre_expansion_total
                                                 : 0.0;
    log_status(
      "--solidity-path-coverage: path distribution — {} unit(s), {} path(s) "
      "total, max {} in '{}', mean {:.1f}; before internal-call expansion the "
      "same units had {} path(s), so expansion multiplied them by {:.2f}x",
      units_enumerated,
      enumerated_total,
      max_unit_paths,
      max_unit_name,
      (double)enumerated_total / (double)units_enumerated,
      pre_expansion_total,
      ratio);
    if (units_at_cap > 0)
      log_warning(
        "--solidity-path-coverage: {} of {} unit(s) hit the per-unit goal cap "
        "({}), so the totals reported above are LOWER BOUNDS for those units "
        "and the tail of this distribution must not be presented as complete. "
        "Their paths keep a valid — but narrower — certified region (see the "
        "assertion-strength report); what is lost is resolution, not "
        "correctness",
        units_at_cap,
        units_enumerated,
        path_cov_max_goals);
  }
  if (non_unit_functions > 0)
    log_status(
      "--solidity-path-coverage: {} in-scope function(s) are internal/private "
      "and are therefore not units; they have no path set of their own and "
      "appear inside the paths of the units that call them",
      non_unit_functions);
  if (sc_sites_over_cap > 0)
    log_warning(
      "--solidity-path-coverage: {} folded short-circuit/ternary site(s) have "
      "more than {} operands and were NOT treated as decisions; the paths "
      "through them are merged rather than enumerated (they stay coverable, "
      "but the decision set is incomplete at those sites)",
      sc_sites_over_cap,
      SC_DECISION_MAX);
  if (skipped_paths > 0)
    log_status(
      "--solidity-path-coverage: {} path(s) already witnessed in a previous "
      "round were not re-instrumented (covered-set {}); denominator remains "
      "the full {} path(s)",
      skipped_paths,
      covered_set_path,
      all_claims.size());

  // ---- THE SIGNAL-SAFE SNAPSHOT, published before a single claim is solved
  // ----
  //
  // A path-coverage run killed by SIGALRM/SIGTERM/SIGINT used to emit NOTHING:
  // the rescue in esbmc_parseoptions.cpp is gated on `branch_cov_active`, whose
  // only writer is branch_coverage(). This pass wrote none of the atomics, so
  // the handler returned at its first line and 27 killed runs across the corpus
  // contributed a zero that is indistinguishable in the gate table from a
  // measured zero.
  //
  // Set HERE, at the end of instrumentation, because that is the first moment
  // the denominator exists and the last moment before anything can be killed
  // mid-solve. A kill during symex then prints "0 of N decided", which is not a
  // result but IS the difference between "this run was cut short" and "this run
  // reached nothing".
  total_paths_atomic.store(all_claims.size(), std::memory_order_relaxed);
  live_F.store(0, std::memory_order_relaxed);
  live_decided.store(0, std::memory_order_relaxed);
  path_cov_active.store(true, std::memory_order_relaxed);

  goto_functions.update();
}

void goto_coveraget::insert_assert(
  goto_programt &goto_program,
  goto_programt::targett &it,
  const expr2tc &guard)
{
  insert_assert(goto_program, it, guard, from_expr(ns, "", guard));
}

/*
  convert
    1: DECL x   <--- it
    ASSIGN X 1
  to
    1: ASSERT(guard);
    DECL x      <--- it
    ASSIGN X 1
*/
void goto_coveraget::insert_assert(
  goto_programt &goto_program,
  goto_programt::targett &it,
  const expr2tc &guard,
  const std::string &idf)
{
  goto_programt::instructiont instruction;
  instruction.make_assertion(guard);
  instruction.location = it->location;
  instruction.function = it->function;
  instruction.location.property("instrumented assertion");
  instruction.location.comment(idf);
  instruction.location.user_provided(true);
  goto_program.insert_swap(it++, instruction);
  it--;
}

int goto_coveraget::get_total_instrument() const
{
  int total_instrument = 0;
  forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      const goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      forall_goto_program_instructions (it, goto_program)
      {
        if (
          it->is_assert() &&
          it->location.property().as_string() == "instrumented assertion" &&
          it->location.user_provided() == true)
        {
          total_instrument++;
        }
      }
    }
  return total_instrument;
}

// Count the total assertion instances in goto level via goto-unwind api
// run the algorithm on the copy of the original goto program
int goto_coveraget::get_total_assert_instance() const
{
  // 1. execute goto unwind
  bounded_loop_unroller unwind_loops;
  unwind_loops.run(goto_functions);
  // 2. calculate the number of assertion instance
  return get_total_instrument();
}

std::set<std::pair<std::string, std::string>>
goto_coveraget::get_total_cond_assert() const
{
  std::set<std::pair<std::string, std::string>> total_cond_assert = {};
  // Rebuilt in lockstep with the returned claim set: every call site assigns
  // it to all_claims, which report_dead_code indexes this map by.
  claim_negation.clear();
  forall_goto_functions (f_it, goto_functions)
  {
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      const goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      forall_goto_program_instructions (it, goto_program)
      {
        if (
          it->is_assert() &&
          it->location.property().as_string() == "instrumented assertion" &&
          it->location.user_provided() == true)
        {
          std::pair<std::string, std::string> claim_pair = std::make_pair(
            it->location.comment().as_string(), it->location.as_string());
          total_cond_assert.insert(claim_pair);
          claim_negation[claim_pair] =
            from_expr(ns, "", gen_not_expr(it->guard));
        }
      }
    }
  }
  return total_cond_assert;
}

/*
  Condition Coverage: fault injection
  1. find condition statements, this includes the converted for_loop/while
  2. insert assertion instances before that statement.
  e.g.
    if (a >1)
  =>
    assert(!(a>1))
    assert(a>1)
    if(a>1)
  then run multi-property
*/
void goto_coveraget::condition_coverage()
{
  // we need to skip the conditions within the built-in library
  // while keeping the file manually included by user
  // this filter, however, is unsound.. E.g. if the src filename is the same as
  // the builtin library name
  total_cond = {{}};

  std::unordered_set<std::string> location_pool = {};
  // cmdline.arg[0]
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      Forall_goto_program_instructions (it, goto_program)
      {
        std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->location.property().as_string() == "skipped")
          // this stands for the auxiliary condition/branch we added.
          continue;

        /*
          Places that could contains condition
          1. GOTO:          if (x == 1);
          2. ASSIGN:        int x = y && z;
          3. ASSERT
          4. ASSUME
          5. FUNCTION_CALL  test((signed int)(x != y));
          6. RETURN         return x && y;
          7. Other          1?2?3:4
          The issue is that, the side-effects have been removed
          thus the condition might have been split or modified.

          For assert, assume and goto, we know it contains GUARD
          For others, we need to convert the code back to expr and
          check there operands.
        */

        // Skip ASSUME instructions: __VERIFIER_assume / __ESBMC_assume
        // express path constraints, not program logic, so their guards
        // must not contribute to condition-coverage claims (issue #4291).
        if (it->is_assume())
          continue;

        // e.g. assert(a == 1);
        if (
          it->is_assert() &&
          it->location.property().as_string() != "replaced assertion")
        {
          if (!is_nil_expr(it->guard))
          {
            expr2tc guard = handle_single_guard(it->guard, true);
            gen_cond_cov_assert(guard, expr2tc(), goto_program, it);
            // after adding the instrumentation, we neutralize the original
            // assert
            if (cov_assume_asserts)
              replace_assert_to_assume(it);
            else
              replace_assert_to_guard(gen_true_expr(), it, false);
          }
        }

        // e.g. IF !(a > 1) THEN GOTO 3
        else if (it->is_goto() && !is_true(it->guard))
        {
          // e.g.
          //    GOTO 2;
          //    2: IF(...);
          if (it->is_target())
            target_num = it->target_number;

          // preprocessing: if(true) ==> if(true == true)
          expr2tc guard = handle_single_guard(it->guard, true);
          gen_cond_cov_assert(guard, expr2tc(), goto_program, it);
        }

        // e.g. bool x = (a>b);
        else if (it->is_assign())
        {
          const expr2tc &rhs = to_code_assign2t(it->code).source;
          if (!is_nil_expr(rhs))
            handle_operands_guard(rhs, goto_program, it);
        }

        // a>b;
        else if (it->is_other())
        {
          if (is_code_expression2t(it->code))
          {
            const expr2tc &other = to_code_expression2t(it->code).operand;
            if (!is_nil_expr(other))
              handle_operands_guard(other, goto_program, it);
          }
        }

        // e.g. RETURN a>b;
        else if (it->is_return())
        {
          const expr2tc &ret = to_code_return2t(it->code).operand;
          if (!is_nil_expr(ret))
            handle_operands_guard(ret, goto_program, it);
        }

        // e.g. func(a>b);
        else if (it->is_function_call())
        {
          for (const expr2tc &op : to_code_function_call2t(it->code).operands)
            if (!is_nil_expr(op))
              handle_operands_guard(op, goto_program, it);
        }

        // reset target number
        target_num = -1;
      }
    }

  total_cond = get_total_cond_assert();
  all_claims = total_cond;

  // recalculate line number/ target number
  goto_functions.update();
}

/*
  algo:
  if(b==0 && c > 90)
  => assert(b==0)
  => assert(!(b==0));
  => assert(!(b==0 && c>90))
  => assert(!(b==0 && !(c>90)))

  if(b==0 || c > 90)
  => assert(b==0)
  => assert((b==0));
  => assert(!(!b==0 && c>90))
  => assert(!(!(b==0) && !(c>90)))
*/
/// Recurse into all sub-expressions of @p ptr, calling
/// gen_cond_cov_assert on each.
void goto_coveraget::gen_cond_cov_assert(
  const expr2tc &ptr,
  const expr2tc &pre_cond,
  goto_programt &goto_program,
  goto_programt::instructiont::targett &it)
{
  if (is_nil_expr(ptr))
    return;
  const std::size_t n = ptr->get_num_sub_exprs();
  if (n == 0)
    return; // atom

  auto recurse_all = [&]() {
    for (std::size_t i = 0; i < n; ++i)
      gen_cond_cov_assert(*ptr->get_sub_expr(i), pre_cond, goto_program, it);
  };

  if (n == 1)
  {
    // (a!=0)++, !a, -a, (_Bool)(int)a
    recurse_all();
  }
  else if (n == 2)
  {
    if (is_comparison_expr(ptr))
    {
      recurse_all();
      add_cond_cov_assert(ptr, pre_cond, goto_program, it);
    }
    else if (is_and2t(ptr))
    {
      const expr2tc &lhs = *ptr->get_sub_expr(0);
      const expr2tc &rhs = *ptr->get_sub_expr(1);
      gen_cond_cov_assert(lhs, pre_cond, goto_program, it);

      // update pre-condition: pre_cond && lhs
      expr2tc new_pre =
        is_nil_expr(pre_cond) ? lhs : gen_and_expr(pre_cond, lhs);
      gen_cond_cov_assert(rhs, new_pre, goto_program, it);
    }
    else if (is_or2t(ptr))
    {
      const expr2tc &lhs = *ptr->get_sub_expr(0);
      const expr2tc &rhs = *ptr->get_sub_expr(1);
      gen_cond_cov_assert(lhs, pre_cond, goto_program, it);

      // update pre-condition: !(pre_cond && lhs)
      expr2tc new_pre =
        is_nil_expr(pre_cond) ? lhs : gen_and_expr(pre_cond, lhs);
      new_pre = gen_not_expr(new_pre);
      gen_cond_cov_assert(rhs, new_pre, goto_program, it);
    }
    else
    {
      // a+=b; a>>(b!=0);
      recurse_all();
    }
  }
  else if (n == 3)
  {
    // ternary if
    const expr2tc &cond = *ptr->get_sub_expr(0);
    const expr2tc &t_val = *ptr->get_sub_expr(1);
    const expr2tc &f_val = *ptr->get_sub_expr(2);

    gen_cond_cov_assert(cond, pre_cond, goto_program, it);

    // update pre-condition: pre_cond && cond
    expr2tc pre_cond_1 =
      is_nil_expr(pre_cond) ? cond : gen_and_expr(pre_cond, cond);
    gen_cond_cov_assert(t_val, pre_cond_1, goto_program, it);

    // update pre-condition: pre_cond && !cond
    expr2tc not_cond = gen_not_expr(cond);
    expr2tc pre_cond_2 =
      is_nil_expr(pre_cond) ? not_cond : gen_and_expr(pre_cond, not_cond);
    gen_cond_cov_assert(f_val, pre_cond_2, goto_program, it);
  }
  else
  {
    log_error("unexpected operand size");
    abort();
  }
}

void goto_coveraget::add_cond_cov_assert(
  const expr2tc &expr,
  const expr2tc &pre_cond,
  goto_programt &goto_program,
  goto_programt::instructiont::targett &it)
{
  expr2tc cond = is_nil_expr(pre_cond) ? expr : gen_and_expr(pre_cond, expr);

  // e.g. assert(!(a==1));  // a==1
  // the idf is used as the claim_msg
  // note that it's different from the actual guard.
  std::string idf = from_expr(ns, "", expr);
  expr2tc guard = gen_not_expr(cond);
  insert_assert(goto_program, it, guard, idf);

  // reversal
  expr2tc not_expr = gen_not_expr(expr);
  cond = is_nil_expr(pre_cond) ? not_expr : gen_and_expr(pre_cond, not_expr);
  idf = from_expr(ns, "", not_expr);
  guard = gen_not_expr(cond);
  insert_assert(goto_program, it, guard, idf);
}

expr2tc goto_coveraget::gen_not_eq_expr(const expr2tc &lhs, const expr2tc &rhs)
{
  expr2tc _lhs = (lhs->type == rhs->type) ? lhs : typecast2tc(rhs->type, lhs);
  return notequal2tc(_lhs, rhs);
}

expr2tc goto_coveraget::gen_and_expr(const expr2tc &lhs, const expr2tc &rhs)
{
  type2tc bt = get_bool_type();
  expr2tc _lhs = is_bool_type(lhs->type) ? lhs : typecast2tc(bt, lhs);
  expr2tc _rhs = is_bool_type(rhs->type) ? rhs : typecast2tc(bt, rhs);
  return and2tc(_lhs, _rhs);
}

expr2tc goto_coveraget::gen_not_expr(const expr2tc &guard) const
{
  if (is_not2t(guard))
    return to_not2t(guard).value;
  return not2tc(guard);
}

/*
  This function convert single guard to a non_equal_to_false expression
  e.g. if(true) ==> if(true!=false)
  rule:
  1. No-op: Do nothing. This means it's a symbol or constant
  2. Binary OP: for boolean expreession, e.g. a>b, a==b, do nothing
  3. Binary OP: for and/or expresson, add on both side, if possible. Do not add
  if it's already a binary boolean expression in 2. e.g. if(x==1 && a++) =>
  if(x==1 && a++ !=0)
  4. Others: for any other expresison, including unary, binary and teranry,
  traverse its op with handle_single_guard recursivly. convert it to not equal
  in the top level only. e.g. if((bool)a+b+c) => if((bool)(a+b+c)!=0) typecast
  <--- add not equal here
    - +
      - a
      - +
        - b
        - c
  e.g. if(a) => if(a!=0); if(true) => if(true != 0); if(a?b:c:d) =>
  if((a?b:c:d)!=0) if(a==b) => if(a==b); if(a&&b) => if(a != 0 && b!=0 )
*/
/// Recursively maps each operand of @p expr through handle_single_guard
/// (with the supplied @p sub_top_level), in place. Foreach_operand detaches
/// the irep_container before mutating, so this is safe even when @p expr
/// shares storage with its caller.
static void replace_operands(
  expr2tc &expr,
  bool sub_top_level,
  const std::function<expr2tc(const expr2tc &, bool)> &recurse)
{
  expr->Foreach_operand([&](expr2tc &op) { op = recurse(op, sub_top_level); });
}

expr2tc goto_coveraget::handle_single_guard(
  const expr2tc &expr,
  bool top_level /* = true */)
{
  if (is_nil_expr(expr))
    return expr;
  const std::size_t n = expr->get_num_sub_exprs();
  auto recurse = [this](const expr2tc &e, bool tl) {
    return handle_single_guard(e, tl);
  };

  // --- Rule 1: Atomic expressions ---
  // If the expression has no operands (a symbol or constant),
  // then if it's Boolean and we're at the outer guard, wrap it with
  // "!= false".
  if (n == 0)
  {
    if (top_level && is_bool_type(expr->type))
      return gen_not_eq_expr(expr, gen_false_expr());
    return expr;
  }

  // --- Special-case for "not" nodes ---
  // For a "not" operator, process its operand with top_level = true so that
  // even nested atomic expressions (like x in !(!(x))) get wrapped.
  if (is_not2t(expr))
  {
    expr2tc result = expr;
    replace_operands(result, /*sub_top_level=*/true, recurse);
    return result;
  }

  // --- Special-case for typecasts to bool ---
  // If we have (bool)(X) and X is not already a recognized guard
  // (comparison or logical AND/OR), unwrap the typecast and wrap X.
  if (is_typecast2t(expr) && is_bool_type(expr->type))
  {
    expr2tc inner = handle_single_guard(to_typecast2t(expr).from, top_level);
    if (!(is_comparison_expr(inner) || is_and2t(inner) || is_or2t(inner)))
      return gen_not_eq_expr(inner, gen_false_expr());
    return inner;
  }

  // --- Process Binary Operators (exactly 2 operands) ---
  if (n == 2)
  {
    expr2tc result = expr;
    if (is_and2t(expr) || is_or2t(expr))
    {
      // Process each operand as an independent guard (top_level = true).
      replace_operands(result, /*sub_top_level=*/true, recurse);
      return result;
    }
    if (is_comparison_expr(expr))
    {
      replace_operands(result, /*sub_top_level=*/false, recurse);
      return result;
    }
    // Other binary operators (e.g. arithmetic '+').
    replace_operands(result, /*sub_top_level=*/false, recurse);
    if (top_level)
      return gen_not_eq_expr(result, gen_false_expr());
    return result;
  }

  // --- Process Non-Binary Operators (Unary, Ternary, etc.) ---
  expr2tc result = expr;
  replace_operands(result, /*sub_top_level=*/false, recurse);

  // For any other expression producing a Boolean value, if at the outer
  // guard (top_level true) and its kind is not among our no-wrap set, then
  // wrap it with "!= false". This catches cases like member accesses.
  if (
    top_level && is_bool_type(result->type) && !is_and2t(result) &&
    !is_or2t(result) && !is_not2t(result) && !is_comparison_expr(result))
    return gen_not_eq_expr(result, gen_false_expr());
  return result;
}

/*
  add condition instrumentation for OTHER, ASSIGN, FUNCTION_CALL..
  whose operands might contain conditions
  we handle guards for each boolean sub-operand.
*/
void goto_coveraget::handle_operands_guard(
  const expr2tc &expr,
  goto_programt &goto_program,
  goto_programt::instructiont::targett &it)
{
  if (is_nil_expr(expr))
    return;
  const std::size_t n = expr->get_num_sub_exprs();
  if (n == 0)
    return;

  expr2tc pre_cond; // nil

  if (n == 1)
  {
    // e.g. RETURN ++(x&&y);
    handle_operands_guard(*expr->get_sub_expr(0), goto_program, it);
  }
  else if (n == 2)
  {
    expr2tc target = expr;
    if (is_and2t(expr) || is_or2t(expr))
    {
      // we do not need to add a !=false at top level
      // e.g. return x?1:0 != return (x?1:0)!=false
      target->Foreach_operand(
        [this](expr2tc &op) { op = handle_single_guard(op, false); });
    }
    gen_cond_cov_assert(target, pre_cond, goto_program, it);
  }
  else
  {
    // this could only be ternary boolean
    expr2tc rewrapped = handle_single_guard(expr, false);
    gen_cond_cov_assert(rewrapped, pre_cond, goto_program, it);
  }
}

// set the target function from "--function"
void goto_coveraget::set_target(const std::string &_tgt)
{
  target_function = _tgt;
}

// check if it's the target function
bool goto_coveraget::is_target_func(
  const irep_idt &f,
  const std::string &tgt_name) const
{
  const symbolt *sym = ns.lookup(f);
  if (sym == nullptr)
  {
    log_error("Cannot find target function");
    abort();
  }

  exprt symbol = symbol_expr(*ns.lookup(f));
  std::string sym_name = symbol.name().as_string();
  if (sym_name == tgt_name)
    return true;

  // For Solidity: modifier expansion renames functions from "func" to
  // "func_modifierName". Support prefix matching so that --function func
  // matches func_modifierName.
  if (
    config.language.lid == language_idt::SOLIDITY &&
    sym_name.size() > tgt_name.size() &&
    sym_name.substr(0, tgt_name.size()) == tgt_name &&
    sym_name[tgt_name.size()] == '_')
    return true;

  return false;
}

// Parse the --negating-property spec "[contract:]function[:line]".
//   1 token  -> function
//   2 tokens -> last all-digits: function:line ; else contract:function
//   3 tokens -> contract:function:line
// A malformed spec (>3 tokens or empty function) degrades to treating the
// whole string as the function name (backward compatible). `line` stays
// empty when no line is given; it is kept as a string so it can be compared
// directly against the instruction location, with no integer parsing.
static void parse_negate_spec(
  const std::string &spec,
  std::string &contract,
  std::string &fname,
  std::string &line)
{
  contract.clear();
  line.clear();

  auto all_digits = [](const std::string &s) {
    return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos;
  };

  std::vector<std::string> tok;
  size_t start = 0;
  for (size_t pos = spec.find(':'); pos != std::string::npos;
       pos = spec.find(':', start))
  {
    tok.push_back(spec.substr(start, pos - start));
    start = pos + 1;
  }
  tok.push_back(spec.substr(start));

  if (tok.size() == 1)
    fname = tok[0];
  else if (tok.size() == 2)
  {
    if (all_digits(tok[1]))
    {
      fname = tok[0];
      line = tok[1];
    }
    else
    {
      contract = tok[0];
      fname = tok[1];
    }
  }
  else if (tok.size() == 3)
  {
    contract = tok[0];
    fname = tok[1];
    if (all_digits(tok[2]))
      line = tok[2];
  }
  else
  {
    log_warning(
      "--negating-property: malformed spec '{}', treating it as a plain "
      "function name",
      spec);
    fname = spec;
  }

  if (fname.empty())
  {
    log_warning(
      "--negating-property: empty function name in spec '{}', treating it as "
      "a plain function name",
      spec);
    contract.clear();
    line.clear();
    fname = spec;
  }
}

// Extract the contract name from a Solidity mangled id of the form
// "sol:@C@<Contract>@F@...". Returns "" for non-contract / non-Solidity ids.
static std::string contract_of(const std::string &mangled_id)
{
  const std::string c_tag = "@C@";
  const std::string f_tag = "@F@";
  size_t cpos = mangled_id.find(c_tag);
  if (cpos == std::string::npos)
    return "";
  cpos += c_tag.size();
  size_t fpos = mangled_id.find(f_tag, cpos);
  if (fpos == std::string::npos || fpos <= cpos)
    return "";
  return mangled_id.substr(cpos, fpos - cpos);
}

// negate the condition inside the assertion
// The idea is that, if the claim is verified safe, and its negated claim is
// also verified safe, then we say this claim is unreachable
void goto_coveraget::negating_asserts(const std::string &tgt_spec)
{
  std::string contract, fname, target_line;
  parse_negate_spec(tgt_spec, contract, fname, target_line);

  std::string old = target_function;
  target_function = fname;

  std::unordered_set<std::string> location_pool = {};
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  // First pass: collect candidate asserts in functions matching the
  // function-name filter (and the optional case-sensitive contract filter),
  // restricted to the user source files.
  std::vector<goto_programt::instructiont::targett> candidates;
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;
      if (!contract.empty() && contract_of(f_it->first.as_string()) != contract)
        continue;

      Forall_goto_program_instructions (it, goto_program)
      {
        std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->is_assert())
          candidates.push_back(it);
      }
    }

  // Select the asserts to negate. When a line is given, keep only those on
  // that source line; if none match, silently fall back to whole-function
  // negation (all candidates).
  std::vector<goto_programt::instructiont::targett> matched;
  if (!target_line.empty())
  {
    for (auto &it : candidates)
      if (it->location.get_line().as_string() == target_line)
        matched.push_back(it);
    if (matched.empty())
    {
      log_debug(
        "coverage",
        "--negating-property: no assert at line {} in '{}', falling back to "
        "whole-function negation",
        target_line,
        fname);
      matched = candidates;
    }
  }
  else
    matched = candidates;

  for (auto &it : matched)
    replace_assert_to_guard(gen_not_expr(it->guard), it, false);

  target_function = old;
}

// return true if this function is skipped
bool goto_coveraget::filter(
  const irep_idt &func_name,
  const goto_programt &goto_program) const
{
  // "--function" mode
  if (target_function != "" && !is_target_func(func_name, target_function))
    return true;

  // Skip the function that is labelled with "__ESBMC_HIDE"
  // Extended to support Python in addition to Solidity
  if (
    goto_program.hide && (config.language.lid == language_idt::SOLIDITY ||
                          config.language.lid == language_idt::PYTHON))
    return true;
  return false;
}
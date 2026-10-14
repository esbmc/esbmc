#ifndef _ESBMC_SOLVERS_SOLVE_H_
#define _ESBMC_SOLVERS_SOLVE_H_

#include <string>
#include <util/config/config.h>
#include <util/symtab/namespace.h>
#include <util/message/message.h>

class array_iface;
class fp_convt;
class smt_convt;
class smt_solver_baset;
class tuple_iface;

typedef smt_solver_baset *(solver_creator)(const optionst &options,
                                           const namespacet &ns,
                                           tuple_iface **tuple_api,
                                           array_iface **array_api,
                                           fp_convt **fp_api);

smt_convt *create_solver(
  std::string solver_name,
  const namespacet &ns,
  const optionst &options);

/* ---- THE PER-QUERY SOLVER BUDGET (--path-cov-claim-timeout) --------------
 *
 * Path coverage decides one INDEPENDENT claim per job, so a single pathological
 * query costs the entire run and takes every already-decided result with it.
 * MEASURED: one run spent ~166 of its 180 s inside the FIRST solver query,
 * never got an answer, never asked the other nine claims, and produced nothing
 * -- on a few hundred MB against 40 GB free. Raising --memlimit or the outer
 * timeout routes AROUND that query; a per-query budget refuses to pay for it.
 *
 * ENFORCED BY THE SOLVER, NOT BY KILLING THE PROCESS. Each backend that has a
 * native per-check limit sets it at construction from `optionst`, which every
 * backend already receives. The limit is per check-sat, and under path coverage
 * a fresh solver is built per claim, so per-check == per-claim.
 *
 * A backend with no such facility must SAY SO rather than silently run
 * unbounded: `smt_timeout_mechanism()` is empty until a backend records one,
 * and the caller publishes "no enforcement" when it stays empty. A budget that
 * is quietly not applied is worse than no budget, because the report would
 * carry `claim_timeout_s: 120` for a run nothing bounded.
 *
 * Header-only on purpose: the three backends live in three separate static
 * libraries, and an inline function's local static is one object across all of
 * them under C++17 -- no new translation unit, no CMake change, no link-order
 * question.
 */
inline std::string &smt_timeout_mechanism_ref()
{
  static std::string mech;
  return mech;
}

inline const std::string &smt_timeout_mechanism()
{
  return smt_timeout_mechanism_ref();
}

/* Accumulates rather than overwrites. A run can legitimately use more than one
 * backend -- the CVC5 fallback fires once per process on an auto-selected
 * backend that ran out of memory -- and a reader asking "what enforced the
 * budget on this run?" must get both answers, not the last one. */
inline void smt_record_timeout_mechanism(const std::string &m)
{
  std::string &cur = smt_timeout_mechanism_ref();
  if (cur.find(m) != std::string::npos)
    return;
  cur += (cur.empty() ? "" : "; ") + m;
}

/* Milliseconds, 0 => unlimited. Reads the RESOLVED value the path-coverage
 * dispatch publishes, not the raw CLI option: boost's `defaulted()` values are
 * never pumped into `optionst` (util/options.cpp optionst::cmdline), so an
 * untouched `--path-cov-claim-timeout` would read as empty here and the default
 * of 120 s would silently never apply. */
inline uint64_t smt_per_query_timeout_ms(const optionst &options)
{
  const std::string v = options.get_option("path-cov-claim-timeout-ms");
  if (v.empty())
    return 0;
  const long long ms = atoll(v.c_str());
  return ms > 0 ? (uint64_t)ms : 0;
}
/// Abort early if the user explicitly selected an SMT solver that was not
/// built into this ESBMC binary. Safe to call before parsing the program;
/// returns silently when no explicit selection was made (the default picker
/// will choose from whatever is built in at solver-creation time).
void check_solver_availability(const optionst &options);

#endif

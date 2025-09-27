#include <esbmc/bmc.h>
#include <esbmc/document_subgoals.h>
#include <fstream>
#include <goto-programs/goto_loops.h>
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/features.h>
#include <goto-symex/xml_goto_trace.h>
#include <langapi/language_util.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <sstream>
#include <util/i2string.h>
#include <irep2/irep2.h>
#include <util/location.h>

#include <util/migrate.h>
#include <util/show_symbol_table.h>
#include <util/time_stopping.h>
#include <util/cache.h>
#include <atomic>
#include <goto-symex/witnesses.h>

std::unordered_set<std::string> goto_functionst::reached_claims;
std::unordered_multiset<std::string> goto_functionst::reached_mul_claims;
std::unordered_set<std::string> goto_functionst::verified_claims;

std::mutex goto_functionst::reached_claims_mutex;
std::mutex goto_functionst::reached_mul_claims_mutex;
std::mutex goto_functionst::verified_claims_mutex;

void report_coverage(
  const optionst &,
  std::unordered_set<std::string> &,
  const std::unordered_multiset<std::string> &)
{
}

bmct::bmct(goto_functionst &, optionst &opts, contextt &_context)
  : options(opts), context(_context), ns(context)
{
}

void bmct::successful_trace(const symex_target_equationt &)
{
}

void bmct::error_trace(smt_convt &, const symex_target_equationt &)
{
}

void bmct::generate_smt_from_equation(smt_convt &, symex_target_equationt &)
  const
{
}

void bmct::keep_alive_function() const
{
}

smt_convt::resultt
bmct::run_decision_procedure(smt_convt &, symex_target_equationt &) const
{
  return {};
}

void bmct::report_success()
{
}

void bmct::report_failure()
{
}

void bmct::show_program(const symex_target_equationt &)
{
}

void bmct::report_trace(smt_convt::resultt &, const symex_target_equationt &)
{
}

void bmct::clear_verified_claims_in_ssa(
  symex_target_equationt &,
  const claim_slicer &,
  const bool &)
{
}

void bmct::clear_verified_claims_in_goto(const claim_slicer &, const bool &)
{
}

void bmct::report_multi_property_trace(
  const smt_convt::resultt &,
  smt_convt *&,
  const symex_target_equationt &,
  const std::atomic<size_t>,
  const goto_tracet &,
  const std::string &)
{
}

void bmct::report_coverage_verbose(
  const claim_slicer &,
  const std::string &,
  const bool &,
  const bool &,
  const bool &,
  const bool &,
  const std::unordered_set<std::string> &,
  const std::unordered_multiset<std::string> &)
{
}

void bmct::report_result(smt_convt::resultt &)
{
}

smt_convt::resultt bmct::start_bmc()
{
  return smt_convt::resultt::P_ERROR;
}

smt_convt::resultt bmct::run(std::shared_ptr<symex_target_equationt> &)
{
  return smt_convt::resultt::P_ERROR;
}

void bmct::bidirectional_search(smt_convt &, const symex_target_equationt &)
{
}

smt_convt::resultt bmct::run_thread(std::shared_ptr<symex_target_equationt> &)
{
  return smt_convt::resultt::P_ERROR;
}

int bmct::ltl_run_thread(symex_target_equationt &) const
{
  return 0;
}

smt_convt::resultt
bmct::multi_property_check(const symex_target_equationt &, size_t, smt_convt &)
{
  return smt_convt::resultt::P_ERROR;
}

void bmct::report_simple_summary(const SimpleSummary &) const
{
}

#include <esbmc/esbmc_parseoptions.h>
#include <esbmc/ts/transition_system.h>
#include <esbmc/ts/ts_btor2.h>
#include <esbmc/ts/ts_engines.h>
#include <esbmc/ts/ts_pdr.h>

#include <util/base/time_stopping.h>
#include <util/message/message.h>

#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

int esbmc_parseoptionst::do_ts_strategy(
  optionst &options,
  goto_functionst &goto_functions)
{
  transition_systemt ts;
  std::string reason;
  fine_timet start = current_time();
  const bool extracted = extract_transition_system(
    goto_functions,
    context,
    options,
    !cmdline.isset("ts-no-live-filter"),
    ts,
    reason);

  if (!extracted)
  {
    log_status("Not a transition system: {}", reason);
    if (
      cmdline.isset("ts-no-fallback") || cmdline.isset("ts-dump") ||
      cmdline.isset("ts-btor2"))
    {
      log_result("VERIFICATION UNKNOWN");
      return 0;
    }
    if (!cmdline.isset("ts-k-induction") && !cmdline.isset("ts-pdr"))
      options.set_option("disable-inductive-step", true);
    return do_bmc_strategy(options, goto_functions);
  }

  log_status(
    "Transition system extracted in {}s",
    time2string(current_time() - start));

  if (cmdline.isset("ts-dump"))
  {
    std::ostringstream oss;
    ts.dump(oss, namespacet(context));
    log_result("{}", oss.str());
    return 0;
  }

  if (cmdline.isset("ts-btor2"))
  {
    std::ostringstream btor2;
    try
    {
      write_btor2(ts, btor2);
    }
    catch (const std::runtime_error &e)
    {
      log_error("BTOR2 export: {}", e.what());
      return 6;
    }
    std::ofstream(cmdline.getval("ts-btor2")) << btor2.str();
    log_status("BTOR2 written to {}", cmdline.getval("ts-btor2"));
    log_result("VERIFICATION UNKNOWN");
    return 0;
  }

  const uint64_t max_k =
    cmdline.isset("unlimited-k-steps")
      ? std::numeric_limits<uint64_t>::max()
      : strtoul(cmdline.getval("max-k-step"), nullptr, 10);

  namespacet ns(context);

  if (cmdline.isset("ts-pdr"))
  {
    ts_pdrt pdr(ts, ns, options);
    unsigned depth = 0;
    switch (pdr.run(max_k, depth))
    {
    case ts_pdrt::resultt::safe:
      log_success(
        "\nSolution found by PDR (invariant checked)\nVERIFICATION SUCCESSFUL");
      return 0;
    case ts_pdrt::resultt::unsafe:
    {
      log_status("PDR found a violation at step {}; replaying with BMC", depth);
      ts_enginet replay(ts, ns, options, true);
      const int rc = replay.bmc(depth, false);
      if (rc != 1)
        log_error("BMC did not reproduce the PDR counterexample");
      return rc;
    }
    case ts_pdrt::resultt::unknown:
      log_status("PDR: {}", pdr.reason);
      log_fail("\nVERIFICATION UNKNOWN");
      return 0;
    }
  }

  if (cmdline.isset("ts-bmc") && cmdline.isset("ts-fresh-solver"))
  {
    if (!ts.prefix_bad.empty())
    {
      ts_enginet prefix(ts, ns, options, true, false);
      if (prefix.solve_prefix() == smt_resultt::P_SATISFIABLE)
      {
        log_fail("\nVERIFICATION FAILED");
        return 1;
      }
    }
    if (ts.bad.empty())
    {
      log_success("\nVERIFICATION SUCCESSFUL");
      return 0;
    }
    for (uint64_t k = 0; k <= max_k; k++)
    {
      fine_timet start = current_time();
      ts_enginet fresh(ts, ns, options, true, false);
      smt_resultt res = fresh.solve_bound(k);
      log_status(
        "TS BMC step {} (fresh solver): {} ({}s)",
        k,
        res == smt_resultt::P_SATISFIABLE     ? "violated"
        : res == smt_resultt::P_UNSATISFIABLE ? "safe"
                                              : "unknown",
        time2string(current_time() - start));
      if (res == smt_resultt::P_SATISFIABLE)
      {
        log_fail("\nVERIFICATION FAILED");
        return 1;
      }
      if (res != smt_resultt::P_UNSATISFIABLE)
        return 6;
    }
    log_fail("\nVERIFICATION UNKNOWN");
    return 0;
  }

  ts_enginet engine(ts, ns, options, !cmdline.isset("ts-k-induction"));
  if (cmdline.isset("ts-k-induction"))
    return engine.k_induction(max_k, !cmdline.isset("ts-no-simple-path"));
  return engine.bmc(max_k, cmdline.isset("ts-push-pop"));
}

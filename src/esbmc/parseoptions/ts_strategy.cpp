#include <esbmc/esbmc_parseoptions.h>
#include <esbmc/ts/transition_system.h>
#include <esbmc/ts/ts_engines.h>
#include <esbmc/ts/ts_pdr.h>

#include <util/base/time_stopping.h>
#include <util/message/message.h>

#include <cstdlib>
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
    if (cmdline.isset("ts-no-fallback") || cmdline.isset("ts-dump"))
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

  ts_enginet engine(ts, ns, options, !cmdline.isset("ts-k-induction"));
  if (cmdline.isset("ts-k-induction"))
    return engine.k_induction(max_k, !cmdline.isset("ts-no-simple-path"));
  return engine.bmc(max_k, cmdline.isset("ts-push-pop"));
}

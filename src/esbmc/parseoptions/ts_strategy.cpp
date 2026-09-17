#include <esbmc/esbmc_parseoptions.h>

#include <transition-system/ts_engines.h>
#include <transition-system/ts_extraction.h>
#include <transition-system/ts_slice.h>
#include <util/base/time_stopping.h>
#include <util/message/message.h>

#include <cstdlib>
#include <limits>
#include <sstream>

bool esbmc_parseoptionst::ts_strategy_requested() const
{
  return cmdline.isset("ts-check") || cmdline.isset("ts-bmc");
}

int esbmc_parseoptionst::do_ts_strategy(
  optionst &options,
  goto_functionst &goto_functions)
{
  transition_systemt ts;
  fine_timet start = current_time();
  const bool extracted =
    extract_transition_system(goto_functions, context, options, ts);

  if (!extracted)
  {
    if (cmdline.isset("ts-check"))
      log_result("TS-CHECK rejected");
    else
      log_status("Not a transition system");
    if (cmdline.isset("ts-check") || cmdline.isset("ts-no-fallback"))
    {
      log_result("VERIFICATION UNKNOWN");
      return 0;
    }
    options.set_option("disable-inductive-step", true);
    return do_bmc_strategy(options, goto_functions);
  }
  if (!cmdline.isset("ts-no-live-filter"))
    slice_transition_system(ts);

  const std::string extract_s = time2string(current_time() - start);
  log_status("Transition system extracted in {}s", extract_s);

  if (cmdline.isset("ts-dump"))
  {
    std::ostringstream oss;
    ts.dump(oss, namespacet(context));
    log_result("{}", oss.str());
  }

  if (cmdline.isset("ts-check"))
  {
    log_result(
      "TS-CHECK accepted states={} inputs={} defs={} bad={} prefix_bad={} "
      "extract_s={}",
      ts.states.size(),
      ts.inputs.size(),
      ts.body_defs.size(),
      ts.bad.size(),
      ts.prefix_bad.size(),
      extract_s);
    log_result("VERIFICATION UNKNOWN");
    return 0;
  }

  const uint64_t max_k = cmdline.isset("unlimited-k-steps")
                           ? std::numeric_limits<uint64_t>::max()
                           : strtoul(cmdline.getval("max-k-step"), nullptr, 10);

  namespacet ns(context);
  ts_enginet engine(ts, ns, options);
  return engine.bmc(max_k, cmdline.isset("ts-push-pop"));
}

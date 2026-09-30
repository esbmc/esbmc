#include <esbmc/esbmc_parseoptions.h>

#include <transition-system/ts_extraction.h>
#include <transition-system/ts_slice.h>
#include <util/base/time_stopping.h>
#include <util/message/message.h>

#include <sstream>

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
    log_result("TS-CHECK rejected");
    log_result("VERIFICATION UNKNOWN");
    return 0;
  }
  slice_transition_system(ts);

  const std::string extract_s = time2string(current_time() - start);

  if (cmdline.isset("ts-dump"))
  {
    std::ostringstream oss;
    ts.dump(oss, namespacet(context));
    log_result("{}", oss.str());
  }

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

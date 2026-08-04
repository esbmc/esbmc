#ifndef _ESBMC_PROP_SMT_EXTERNAL_PROCESS_DIED_H_
#define _ESBMC_PROP_SMT_EXTERNAL_PROCESS_DIED_H_

#include <stdexcept>

/** Thrown when an external solver process can no longer provide a usable
 *  answer: a write hit EPIPE (the process died) or its response could not be
 *  parsed (typically EOF from a dead process).
 *
 *  Backends that talk to a solver over a pipe throw this instead of aborting,
 *  so a death mid-answer — notably while a counterexample is being read out
 *  via (get-value), past the point where a verdict was already returned — is
 *  reported as a clean failure. bmc_strategy.cpp catches it. */
struct external_process_died : std::runtime_error
{
  using std::runtime_error::runtime_error;
};

#endif /* _ESBMC_PROP_SMT_EXTERNAL_PROCESS_DIED_H_ */

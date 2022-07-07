/*******************************************************************\

Module: Message System. This system is used to send messages through
    ESBMC execution.
Author: Daniel Kroening, kroening@kroening.com

Maintainers:
- @2021: Rafael Sá Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/
#pragma once

//#include <fmt/format.h>
#include <util/location.h>

/**
   * @brief Verbosity refers to the max level
   * of which inputs are going to be printed out
   *
   * The level adds up to the greater level which means
   * that if the level is set to 3 all messages of value
   * 0,1,2,3 are going to be printed but 4+ will not be printed
   *
   * The number is where it appeared in the definition, in the
   * implementation below DEBUG is the highest value
   */
enum class VerbosityLevel : char
{
  None,     // No message output
  Error,    // fatal errors are printed
  Warning,  // warnings are printend
  Result,   // results of the analysis (including CE)
  Progress, // progress notifications
  Status,   // all kinds of esbmc is doing that may be useful to the user
  Debug     // messages that are only useful if you need to debug.
};

static inline void print(VerbosityLevel, const std::string&, const locationt&) {}


// Macro to generate log functions
#define log_message(name, verbosity)                                           \
  template <typename Arg, typename... Args>                                    \
  static inline void log_##name(Arg &&arg, Args &&...args)                     \
  {                                                                            \
                                                                                \
  }

log_message(error, VerbosityLevel::Error);
log_message(result, VerbosityLevel::Result);
log_message(warning, VerbosityLevel::Warning);
log_message(progress, VerbosityLevel::Progress);
log_message(status, VerbosityLevel::Status);
log_message(debug, VerbosityLevel::Debug);

#undef log_message

#include <stdio.h>
class messaget_state
{
public:
  static inline auto verbosity = VerbosityLevel::Status;
  static inline auto error_output = stderr;
  static inline auto standard_output = stdout;
};

// TODO: Eventually this will be removed
#ifdef ENABLE_OLD_FRONTEND
#define err_location(E) (E).location().dump();
#endif
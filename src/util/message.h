/*******************************************************************\

Module: Message System. This system is used to send messages through
    ESBMC execution.
Author: Daniel Kroening, kroening@kroening.com

Maintainers:
- @2021: Rafael Sá Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/
#pragma once

#include <cstdio>
#include <fmt/format.h>
#include <util/message/format.h>
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
 * implementation below Debug is the highest value
 */
enum class VerbosityLevel : char
{
  None,     // No message output
  Error,    // fatal errors are printed
  Warning,  // warnings are printend
  Result,   // results of the analysis (including CE)
  Progress, // progress notifications
  Status,   // all kinds of things esbmc is doing that may be useful to the user
  Debug     // messages that are only useful if you need to debug.
};

struct messaget
{
  static inline class
  {
    template <typename... Args>
    void println(FILE *f, Args &&...args) const
    {
      fmt::print(f, std::forward<Args>(args)...);
      fmt::print(f, "\n");
      if(flush_log)
        fflush(f);
    }

  public:
    VerbosityLevel verbosity;
    FILE *out;
    FILE *err;
    bool flush_log;

    FILE *target(VerbosityLevel lvl) const
    {
      return lvl > verbosity                ? nullptr
             : lvl == VerbosityLevel::Error ? err
                                            : out;
    }

    template <typename File, typename Line, typename... Args>
    bool logln(
      VerbosityLevel lvl,
      const File &file,
      const Line &line,
      Args &&...args) const
    {
      FILE *f = target(lvl);
      if(!f)
        return false;
      println(f, std::forward<Args>(args)...);
      return true;
      /* unused: */
      (void)file;
      (void)line;
    }
  } state = {VerbosityLevel::Status, stdout, stderr, false};
};

static inline void
print(VerbosityLevel lvl, std::string_view msg, const locationt &loc)
{
  messaget::state.logln(lvl, loc.get_file(), loc.get_line(), "{}", msg);
}

// Macro to generate log functions
#define log_message(name, verbosity)                                           \
  template <typename... Args>                                                  \
  static inline void log_##name(std::string_view fmt, Args &&...args)          \
  {                                                                            \
    messaget::state.logln(                                                     \
      verbosity, __FILE__, __LINE__, fmt, std::forward<Args>(args)...);        \
  }

log_message(error, VerbosityLevel::Error);
log_message(result, VerbosityLevel::Result);
log_message(warning, VerbosityLevel::Warning);
log_message(progress, VerbosityLevel::Progress);
log_message(status, VerbosityLevel::Status);
log_message(debug, VerbosityLevel::Debug);

#undef log_message

// TODO: Eventually this will be removed
#ifdef ENABLE_OLD_FRONTEND
#define err_location(E) (E).location().dump()
#endif

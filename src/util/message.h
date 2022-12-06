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
#include <fmt/color.h>
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
  Success,  // verification success/claim holds
  Fail,     // verification/claim fails
  Status,   // all kinds of things esbmc is doing that may be useful to the user
  Debug     // messages that are only useful if you need to debug.
};

struct messaget
{
  static inline class
  {
    template <typename... Args>
    static void println(FILE *f, VerbosityLevel lvl, Args &&...args)
    {
      if(config.options.get_bool_option("quiet"))
        return;

      if(config.options.get_bool_option("color"))
      {
        switch(lvl)
        {
        case VerbosityLevel::Error:
          fmt::print(
            f, fmt::fg(fmt::color::red) | fmt::emphasis::bold, "[ERROR] ");
          fmt::print(f, std::forward<Args>(args)...);
          break;
        case VerbosityLevel::Warning:
          fmt::print(
            f, fmt::fg(fmt::color::yellow) | fmt::emphasis::bold, "[WARNING] ");
          fmt::print(f, std::forward<Args>(args)...);
          break;
        case VerbosityLevel::Progress:
          fmt::print(
            f, fmt::fg(fmt::color::blue) | fmt::emphasis::bold, "[PROGRESS] ");
          fmt::print(f, std::forward<Args>(args)...);
          break;
        case VerbosityLevel::Fail:
          fmt::print(f, fmt::fg(fmt::color::red), std::forward<Args>(args)...);
          break;
        case VerbosityLevel::Success:
          fmt::print(
            f, fmt::fg(fmt::color::green), std::forward<Args>(args)...);
          break;
        default:
          fmt::print(f, std::forward<Args>(args)...);
          break;
        }
      }
      else
      {
        if(lvl == VerbosityLevel::Error)
          fmt::print(f, "[ERROR] ");
        if(lvl == VerbosityLevel::Warning)
          fmt::print(f, "[WARNING] ");
        fmt::print(f, std::forward<Args>(args)...);
      }
      fmt::print(f, "\n");
    }

  public:
    VerbosityLevel verbosity;
    FILE *out;
    FILE *err;

    FILE *target(VerbosityLevel lvl) const
    {
      return lvl > verbosity                ? nullptr
             : lvl == VerbosityLevel::Error ? err
                                            : out;
    }

    void set_flushln() const
    {
/* Win32 interprets _IOLBF as _IOFBF (and then chokes on size=0) */
#if !defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
      setvbuf(out, NULL, _IOLBF, 0);
      setvbuf(err, NULL, _IOLBF, 0);
#endif
    }

    template <typename... Args>
    bool logln(VerbosityLevel lvl, Args &&...args) const
    {
      FILE *f = target(lvl);
      if(!f)
        return false;
      println(f, lvl, std::forward<Args>(args)...);
      return true;
    }
  } state = {VerbosityLevel::Status, stdout, stderr};
};

static inline void
print(VerbosityLevel lvl, std::string_view msg, const locationt &)
{
  messaget::state.logln(lvl, "{}", msg);
}

// Macro to generate log functions
#define log_message(name, verbosity)                                           \
  template <typename... Args>                                                  \
  static inline void log_##name(std::string_view fmt, Args &&...args)          \
  {                                                                            \
    messaget::state.logln(verbosity, fmt, std::forward<Args>(args)...);        \
  }

log_message(error, VerbosityLevel::Error);
log_message(result, VerbosityLevel::Result);
log_message(warning, VerbosityLevel::Warning);
log_message(progress, VerbosityLevel::Progress);
log_message(success, VerbosityLevel::Success);
log_message(fail, VerbosityLevel::Fail);
log_message(status, VerbosityLevel::Status);
log_message(debug, VerbosityLevel::Debug);

#undef log_message

// TODO: Eventually this will be removed
#ifdef ENABLE_OLD_FRONTEND
#define err_location(E) (E).location().dump()
#endif

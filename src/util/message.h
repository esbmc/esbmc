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
    static void println(FILE *f, VerbosityLevel lvl, fmt::string_view format,
                        fmt::format_args args)
    {
      if(config.options.get_bool_option("color"))
      {
        switch(lvl)
        {
        case VerbosityLevel::Error:
          fmt::print(
            f, fmt::fg(fmt::color::red) | fmt::emphasis::bold, "[ERROR] ");
          fmt::vprint(f, format, args);
          break;
        case VerbosityLevel::Warning:
          fmt::print(
            f, fmt::fg(fmt::color::yellow) | fmt::emphasis::bold, "[WARNING] ");
          fmt::vprint(f, format, args);
          break;
        case VerbosityLevel::Progress:
          fmt::print(
            f, fmt::fg(fmt::color::blue) | fmt::emphasis::bold, "[PROGRESS] ");
          fmt::vprint(f, format, args);
          break;
        case VerbosityLevel::Fail:
          fmt::vprint(f, fmt::fg(fmt::color::red), format, args);
          break;
        case VerbosityLevel::Success:
          fmt::vprint(
            f, fmt::fg(fmt::color::green), format, args);
          break;
        default:
          fmt::vprint(f, format, args);
          break;
        }
      }
      else
      {
        if(lvl == VerbosityLevel::Error)
          fmt::print(f, "ERROR: ");
        fmt::vprint(f, format, args);
      }

      fmt::print(f, "\n");
    }

  public:
    VerbosityLevel verbosity;
    std::unordered_map<std::string, VerbosityLevel> modules;
    FILE *out;

    FILE *target(const char *mod, VerbosityLevel lvl) const
    {
      VerbosityLevel l = verbosity;
      if(mod)
        if(auto it = modules.find(mod); it != modules.end())
          l = it->second;
      return lvl > l ? nullptr : out;
    }

    void set_flushln() const
    {
/* Win32 interprets _IOLBF as _IOFBF (and then chokes on size=0) */
#if !defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
      setvbuf(out, NULL, _IOLBF, 0);
#endif
    }

    template <typename... Args>
    bool logln(
      const char *mod,
      VerbosityLevel lvl,
      const char *file,
      int line,
      fmt::format_string<Args...> format,
      Args &&...args) const
    {
      FILE *f = target(mod, lvl);
      if(!f)
        return false;
      println(f, lvl, format, fmt::make_format_args(args...));
      return true;
      (void)file;
      (void)line;
    }
  } state = {VerbosityLevel::Status, {}, stderr};
};

#define log_error(fmt, ...)                                                    \
  messaget::state.logln(                                                       \
    nullptr, VerbosityLevel::Error, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define log_result(fmt, ...)                                                   \
  messaget::state.logln(                                                       \
    nullptr, VerbosityLevel::Result, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define log_warning(fmt, ...)                                                  \
  messaget::state.logln(                                                       \
    nullptr, VerbosityLevel::Warning, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define log_progress(fmt, ...)                                                 \
  messaget::state.logln(                                                       \
    nullptr, VerbosityLevel::Progress, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define log_success(fmt, ...)                                                  \
  messaget::state.logln(                                                       \
    nullptr, VerbosityLevel::Success, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define log_fail(fmt, ...)                                                     \
  messaget::state.logln(                                                       \
    nullptr, VerbosityLevel::Fail, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define log_status(fmt, ...)                                                   \
  messaget::state.logln(                                                       \
    nullptr, VerbosityLevel::Status, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define log_debug(mod, fmt, ...)                                               \
  messaget::state.logln(                                                       \
    mod, VerbosityLevel::Debug, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

// TODO: Eventually this will be removed
#ifdef ENABLE_OLD_FRONTEND
#define err_location(E) (E).location().dump()
#endif


#include <util/config.h>
#include <util/message.h>

void messaget::statet::println(
  FILE *f,
  VerbosityLevel lvl,
  fmt::string_view format,
  fmt::format_args args)
{
  if (config.options.get_bool_option("color"))
  {
    switch (lvl)
    {
    case VerbosityLevel::Error:
      fmt::print(f, fmt::fg(fmt::color::red) | fmt::emphasis::bold, "[ERROR] ");
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
      fmt::vprint(f, fmt::fg(fmt::color::green), format, args);
      break;
    default:
      fmt::vprint(f, format, args);
      break;
    }
  }
  else
  {
    if (lvl == VerbosityLevel::Error)
      fmt::print(f, "ERROR: ");
    if (lvl == VerbosityLevel::Warning)
      fmt::print(f, "WARNING: ");
    fmt::vprint(f, format, args);
  }

  fmt::print(f, "\n");
}

FILE *messaget::statet::target(const char *mod, VerbosityLevel lvl) const
{
  VerbosityLevel l = verbosity;
  if (mod)
    if (auto it = modules.find(mod); it != modules.end())
      l = it->second;
  return lvl > l ? nullptr : out;
}

void messaget::statet::set_flushln() const
{
/* Win32 interprets _IOLBF as _IOFBF (and then chokes on size=0) */
#if !defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
  setvbuf(out, NULL, _IOLBF, 0);
#endif
}

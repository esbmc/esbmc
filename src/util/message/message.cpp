
#include <util/config/config.h>
#include <util/message/message.h>
#include <iterator>

/* The prefix, the message and the newline go out in a single fwrite. Emitting
 * them one at a time lets a concurrent writer sharing the descriptor land
 * between two of them: --k-induction-parallel forks three children onto
 * stderr, which is unbuffered, so each part is its own write(2). A verdict cut
 * in half stops matching a regression's `^VERIFICATION FAILED$` even though
 * the run was correct. This is what fmt's own vprintln does. */
void messaget::statet::println(
  FILE *f,
  VerbosityLevel lvl,
  fmt::string_view format,
  fmt::format_args args)
{
  fmt::memory_buffer line;
  auto out = std::back_inserter(line);

  if (config.options.get_bool_option("color"))
  {
    switch (lvl)
    {
    case VerbosityLevel::Error:
      fmt::format_to(
        out, fmt::fg(fmt::color::red) | fmt::emphasis::bold, "[ERROR] ");
      fmt::vformat_to(out, format, args);
      break;
    case VerbosityLevel::Warning:
      fmt::format_to(
        out, fmt::fg(fmt::color::yellow) | fmt::emphasis::bold, "[WARNING] ");
      fmt::vformat_to(out, format, args);
      break;
    case VerbosityLevel::Progress:
      fmt::format_to(
        out, fmt::fg(fmt::color::blue) | fmt::emphasis::bold, "[PROGRESS] ");
      fmt::vformat_to(out, format, args);
      break;
    case VerbosityLevel::Fail:
      fmt::vformat_to(out, fmt::fg(fmt::color::red), format, args);
      break;
    case VerbosityLevel::Success:
      fmt::vformat_to(out, fmt::fg(fmt::color::green), format, args);
      break;
    default:
      fmt::vformat_to(out, format, args);
      break;
    }
  }
  else
  {
    if (lvl == VerbosityLevel::Error)
      fmt::format_to(out, "ERROR: ");
    if (lvl == VerbosityLevel::Warning)
      fmt::format_to(out, "WARNING: ");
    fmt::vformat_to(out, format, args);
  }

  line.push_back('\n');
  fwrite(line.data(), 1, line.size(), f);
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

#include <util/message/fmt_message_handler.h>
#include <fmt/core.h>

void fmt_message_handler::print(
  VerbosityLevel level,
  const std::string &message) const
{
  fmt::print(files.at(level), "{}\n", message);
}

fmt_message_handler::fmt_message_handler()
{
  files[VerbosityLevel::None] = NULL;
  files[VerbosityLevel::Error] = stderr;
  files[VerbosityLevel::Warning] = stdout;
  files[VerbosityLevel::Result] = stdout;
  files[VerbosityLevel::Progress] = stdout;
  files[VerbosityLevel::Status] = stdout;
  files[VerbosityLevel::Debug] = stdout;
}
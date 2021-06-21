#include <util/message/fmt_message_handler.h>
#include <fmt/core.h>

void fmt_message_handler::print(
  message_handlert::VERBOSITY level,
  const std::string &message) const
{
  fmt::print(files.at(level), "{}\n", message);
}

fmt_message_handler::fmt_message_handler()
{
  files[message_handlert::NONE] = NULL;
  files[message_handlert::ERROR] = stderr;
  files[message_handlert::WARNING] = stdout;
  files[message_handlert::RESULT] = stdout;
  files[message_handlert::PROGRESS] = stdout;
  files[message_handlert::STATUS] = stdout;
  files[message_handlert::DEBUG] = stdout;
}
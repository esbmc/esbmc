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
  initialize(stdout, stderr);
}

fmt_message_handler::fmt_message_handler(FILE *out, FILE *err)
{
  initialize(out, err);
}

void fmt_message_handler::initialize(FILE *out, FILE *err)
{
  files[VerbosityLevel::None] = nullptr;
  files[VerbosityLevel::Error] = err;
  files[VerbosityLevel::Warning] = out;
  files[VerbosityLevel::Result] = out;
  files[VerbosityLevel::Progress] = out;
  files[VerbosityLevel::Status] = out;
  files[VerbosityLevel::Debug] = out;
}
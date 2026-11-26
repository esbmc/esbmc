#include <llm/cli_llm_client.h>

#include <boost/version.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/readable_pipe.hpp>
#include <boost/asio/writable_pipe.hpp>
#if __has_include(<boost/process/v2.hpp>)
#  include <boost/process/v2.hpp>
#else
#  include <boost/process.hpp>
#endif
#if BOOST_VERSION < 108800
// Boost.Process v2 is compiled into a library only from 1.88 onwards.
#  include <boost/process/v2/src.hpp>
#endif

#include <csignal>
#include <sstream>
#include <stdexcept>

#include <fmt/format.h>

#include <util/message/message.h>

namespace llm
{
namespace asio = boost::asio;
namespace bp = boost::process::v2;

cli_clientt::cli_clientt(configt config) : clientt(std::move(config))
{
  if (cfg.executable.empty())
  {
    log_error("CLI LLM backend requires an executable");
    abort();
  }
}

std::string cli_clientt::complete(const std::vector<message_t> &messages)
{
  std::ostringstream prompt;
  for (const auto &m : messages)
    prompt << m.role << ": " << m.content << "\n";
  prompt << "assistant: ";

  asio::io_context ctx;
  asio::writable_pipe in(ctx);
  asio::readable_pipe out(ctx);

  const bp::filesystem::path named(cfg.executable);
  const auto exe = named.has_parent_path()
                     ? named
                     : bp::environment::find_executable(cfg.executable);
  if (exe.empty() || !bp::filesystem::exists(exe))
    throw std::runtime_error(
      fmt::format("CLI LLM executable '{}' not found", cfg.executable));

  bp::process child(
    ctx, exe, cfg.extra_args, bp::process_stdio{in, out, nullptr});

  const std::string input = prompt.str();
#ifndef _WIN32
  // A command that exits without reading its input must not kill ESBMC;
  // asio may write as soon as async_write is called.
  const auto old_sigpipe = signal(SIGPIPE, SIG_IGN);
#endif
  asio::async_write(
    in, asio::buffer(input), [&in](boost::system::error_code, std::size_t) {
      in.close();
    });

  int exit_code = 0;
  bool timed_out = false, read_done = false, child_done = false;
  asio::steady_timer timer(ctx, std::chrono::milliseconds(cfg.timeout_ms));
  timer.async_wait([&](boost::system::error_code ec) {
    if (ec)
      return;
    timed_out = true;
    boost::system::error_code ignored;
    child.terminate(ignored);
    // A grandchild can still hold the pipes open; closing them ends the reads.
    in.close();
    out.close();
  });

  std::string result;
  asio::async_read(
    out, asio::dynamic_buffer(result), [&](boost::system::error_code, size_t) {
      read_done = true;
      if (child_done)
        timer.cancel();
    });
  child.async_wait([&](boost::system::error_code, int code) {
    exit_code = code;
    child_done = true;
    if (read_done)
      timer.cancel();
  });

  ctx.run();
#ifndef _WIN32
  signal(SIGPIPE, old_sigpipe);
#endif

  if (timed_out)
    throw std::runtime_error(
      fmt::format("CLI LLM process timed out after {} ms", cfg.timeout_ms));
  if (exit_code != 0)
    throw std::runtime_error(
      fmt::format("CLI LLM process exited with {}", exit_code));

  while (!result.empty() && result.back() == '\n')
    result.pop_back();
  return result;
}

} // namespace llm

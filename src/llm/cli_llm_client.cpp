#include <llm/cli_llm_client.h>

// Use boost::process v1 on macOS or when Boost >= 1.87.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#if defined(__APPLE__) || (BOOST_VERSION == 108700)
#  include <boost/process/v1.hpp>
namespace bp = boost::process::v1;
#elif BOOST_VERSION >= 108800
#  include <boost/process/v1/child.hpp>
#  include <boost/process/v1/io.hpp>
#  include <boost/process/v1/search_path.hpp>
namespace bp = boost::process::v1;
#else
#  include <boost/process.hpp>
namespace bp = boost::process;
#endif
#pragma GCC diagnostic pop

#include <sstream>

#include <util/message/message.h>

namespace llm
{
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

  bp::ipstream out;
  bp::opstream in;

  std::vector<std::string> args = cfg.extra_args;
  bp::child c(
    bp::search_path(cfg.executable),
    args,
    bp::std_in<in, bp::std_out> out,
    bp::std_err > bp::null);

  in << prompt.str();
  in.flush();
  in.pipe().close();

  std::string result;
  std::string line;
  while (std::getline(out, line))
  {
    if (!result.empty())
      result.push_back('\n');
    result += line;
  }

  c.wait();
  if (c.exit_code() != 0)
  {
    log_error("CLI LLM process exited with {}", c.exit_code());
    abort();
  }

  return result;
}

} // namespace llm

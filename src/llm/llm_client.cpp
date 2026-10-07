#include <llm/llm_client.h>

#include <llm/cli_llm_client.h>
#include <llm/openai_llm_client.h>

#include <util/message/message.h>

namespace llm
{
namespace
{
class stub_clientt : public clientt
{
public:
  explicit stub_clientt(configt config) : clientt(std::move(config))
  {
  }

  std::string complete(const std::vector<message_t> &messages) override
  {
    return "[" + cfg.model + "] got " + std::to_string(messages.size()) +
           " message(s)";
  }
};
} // namespace

std::unique_ptr<clientt> make_client(configt config)
{
  switch (config.backend)
  {
  case backendt::stub:
    return std::make_unique<stub_clientt>(std::move(config));
  case backendt::openai:
    return std::make_unique<openai_clientt>(std::move(config));
  case backendt::cli:
    return std::make_unique<cli_clientt>(std::move(config));
  case backendt::selfhosted:
    log_error("self-hosted LLM backend is not implemented");
    abort();
  }

  log_error("unknown LLM backend");
  abort();
}

} // namespace llm

#include <llm/openai_llm_client.h>

#include <nlohmann/json.hpp>

#include <sstream>

#include <util/message/message.h>

namespace llm
{
static std::string default_endpoint()
{
  return "https://api.openai.com/v1";
}

static nlohmann::json
build_payload(const configt &cfg, const std::vector<message_t> &messages)
{
  nlohmann::json msgs = nlohmann::json::array();
  for (const auto &m : messages)
  {
    msgs.push_back({{"role", m.role}, {"content", m.content}});
  }

  nlohmann::json payload;
  payload["model"] = cfg.model;
  payload["messages"] = msgs;
  payload["temperature"] = cfg.temperature;
  if (cfg.max_tokens > 0)
    payload["max_tokens"] = cfg.max_tokens;

  return payload;
}

openai_clientt::openai_clientt(configt config) : clientt(std::move(config))
{
  if (cfg.endpoint.empty())
    cfg.endpoint = default_endpoint();
}

std::string openai_clientt::complete(const std::vector<message_t> &messages)
{
  log_error("OpenAI LLM backend requires a solver or HTTP engine");
  abort();
}

} // namespace llm

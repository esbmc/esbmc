#pragma once

#include <llm/llm_client.h>

namespace llm
{
/**
 * @brief OpenAI-compatible HTTP LLM client.
 *
 * Sends the conversation to the configured endpoint's /chat/completions
 * route. This is currently a stub: the actual HTTP transport is not wired in,
 * so complete() logs an error and aborts.
 */
class openai_clientt : public clientt
{
public:
  explicit openai_clientt(configt config);

  std::string complete(const std::vector<message_t> &messages) override;
};
} // namespace llm

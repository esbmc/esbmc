#pragma once

#include <llm/llm_client.h>

namespace llm
{
/**
 * @brief LLM client that invokes an external CLI tool.
 *
 * The tool is started as a child process for every completion request. The
 * full conversation is written to its stdin; the tool's stdout is returned as
 * the assistant response.
 */
class cli_clientt : public clientt
{
public:
  explicit cli_clientt(configt config);

  std::string complete(const std::vector<message_t> &messages) override;
};
} // namespace llm

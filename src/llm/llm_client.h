#pragma once

#include <memory>
#include <string>
#include <vector>

namespace llm
{
struct message_t
{
  std::string role;
  std::string content;
};

enum class backendt
{
  stub,
  cli
};

struct configt
{
  backendt backend = backendt::stub;
  std::string model;
  std::string endpoint;
  std::string api_key;
  double temperature = 0.7;
  unsigned max_tokens = 1024;
  unsigned timeout_ms = 30000;

  // For the CLI backend.
  std::string executable;
  std::vector<std::string> extra_args;
};

/**
 * @brief Abstract LLM client.
 *
 * Implementations provide a single chat-completion interface that consumes a
 * message history and returns the assistant's response as plain text.
 */
class clientt
{
public:
  virtual ~clientt() = default;

  // Single chat completion; returns the assistant message content.
  virtual std::string complete(const std::vector<message_t> &messages) = 0;

  const configt &config() const noexcept
  {
    return cfg;
  }

protected:
  explicit clientt(configt config) : cfg(std::move(config))
  {
  }

  configt cfg;
};

std::unique_ptr<clientt> make_client(configt config);

} // namespace llm

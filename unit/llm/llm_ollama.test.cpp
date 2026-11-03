/*******************************************************************\
Module: Manual integration test: ask a local Ollama model to simplify IRep2
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2_utils.h>
#include <llm/llm_client.h>
#include <util/arith/arith_tools.h>
#include <util/config/config.h>

#include <sstream>
#include <string>

namespace
{
std::string expr_to_string(const expr2tc &expr)
{
  return expr->pretty(0);
}

std::string expr_to_prompt(const expr2tc &expr)
{
  std::ostringstream s;
  s << "Simplify the following arithmetic expression and answer with "
       "only the integer result, no explanation.\n";
  s << "Expression: " << expr_to_string(expr) << "\n";
  s << "Answer: ";
  return s.str();
}
} // namespace

// Pinned Ollama model: qwen2.5-coder:3b-instruct
// Digest:
// sha256:f72c60cabf6237b07f6e632b2c48d533cef25eda2efbd34bed21c5e9c01e6225
// Quantization: Q4_K_M, parameter size: 3.1B.
TEST_CASE(
  "ollama qwen2.5-coder:3b-instruct simplifies 1+1",
  "[.manual][llm][ollama]")
{
  config.ansi_c.set_data_model(configt::LP64);

  expr2tc one = constant_int2tc(get_int32_type(), BigInt(1));
  expr2tc expr = add2tc(get_int32_type(), one, one);

  llm::configt cfg;
  cfg.backend = llm::backendt::cli;
  cfg.model = "qwen2.5-coder:3b-instruct"; // pinned, see digest above
  cfg.executable = "ollama";
  cfg.extra_args = {"run", cfg.model};
  cfg.timeout_ms = 120000;

  auto client = llm::make_client(cfg);
  REQUIRE(client != nullptr);

  std::string answer = client->complete(
    {{"system", "You output a single integer."},
     {"user", expr_to_prompt(expr)}});

  // Trim whitespace and punctuation.
  size_t first = answer.find_first_not_of(" \t\n\r");
  size_t last = answer.find_last_not_of(" \t\n\r.");
  std::string trimmed =
    first == std::string::npos ? "" : answer.substr(first, last - first + 1);

  REQUIRE(trimmed == "2");
}

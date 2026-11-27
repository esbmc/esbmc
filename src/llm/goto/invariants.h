#pragma once

#include <string>
#include <vector>

namespace llm
{
/// System prompt asking for loop invariants with chain-of-thought reasoning
/// followed by an <invariants> block that parse_invariant_answer reads.
extern const char *const invariant_system_prompt;

/// One `function L<n>: expression` line of a model's answer.
struct invariant_linet
{
  std::string function;
  std::string loop;
  std::string expression;
};

/// The lines of the last <invariants> block in @p answer. Text outside the
/// block is the model's reasoning and is ignored, as are malformed lines.
std::vector<invariant_linet> parse_invariant_answer(const std::string &answer);
} // namespace llm

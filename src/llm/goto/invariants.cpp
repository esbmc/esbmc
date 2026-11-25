#include <llm/goto/invariants.h>

#include <regex>
#include <sstream>

namespace llm
{
const char *const invariant_system_prompt =
  "You are an expert in formal program verification. You are given C "
  "functions in which every loop carries a label such as L1 and a comment "
  "listing the variables it modifies. Propose loop invariants that a bounded "
  "model checker will check with this schema: (1) the invariant holds when "
  "the loop is first reached; (2) if it holds and the loop condition is true, "
  "it still holds after one iteration of the body; (3) together with the "
  "negated loop condition it implies the assertions that follow the loop.\n"
  "\n"
  "Think step by step before answering. For each loop: (a) state what each "
  "modified variable is before the loop; (b) describe how each changes in one "
  "iteration; (c) relate the variables to each other and to their values "
  "before the loop; (d) bound every counter; (e) check your invariant against "
  "(1), (2) and (3), including C integer overflow and the loop exit value.\n"
  "\n"
  "Then end your answer with exactly one block of this form:\n"
  "<invariants>\n"
  "main L1: i <= n\n"
  "main L1: s == 2 * i\n"
  "</invariants>\n"
  "Each line is `<function> <loop>: <expression>`, one conjunct per line. An "
  "expression is a side-effect-free C expression over the variables the "
  "function shows, using the names exactly as printed: no function calls, no "
  "quantifiers, no assignments.";

std::vector<invariant_linet> parse_invariant_answer(const std::string &answer)
{
  static const std::string open = "<invariants>", close = "</invariants>";
  const size_t begin = answer.rfind(open);
  if (begin == std::string::npos)
    return {};
  const size_t body = begin + open.size();
  const size_t end = answer.find(close, body);

  static const std::regex line_re(
    R"(^\s*`?\s*([A-Za-z_]\w*)\s+(L\d+)\s*:\s*(.*?)\s*;?\s*`?\s*$)");
  std::vector<invariant_linet> lines;
  std::istringstream in(answer.substr(
    body, end == std::string::npos ? std::string::npos : end - body));
  std::smatch m;
  for (std::string line; std::getline(in, line);)
    if (std::regex_match(line, m, line_re) && m[3].length() > 0)
      lines.push_back({m[1], m[2], m[3]});
  return lines;
}
} // namespace llm

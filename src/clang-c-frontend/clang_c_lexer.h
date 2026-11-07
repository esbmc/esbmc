#ifndef CLANG_C_FRONTEND_CLANG_C_LEXER_H_
#define CLANG_C_FRONTEND_CLANG_C_LEXER_H_

#include <irep2/irep2.h>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>

class clang_c_lexert
{
public:
  clang_c_lexert();
  ~clang_c_lexert();

  /**
   * @brief Parse a C scalar-expression string into an IRep2 expression tree.
   *
   * Uses the Clang raw lexer and NumericLiteralParser. Supports comparison
   * operators (==, !=, <, <=, >, >=) composed with && and || and parenthesised
   * sub-expressions.
   *
   * The special token \result is emitted as symbol2tc(int64, "\\result").
   * Integer constants are emitted as constant_int2tc(int64, N). Callers are
   * responsible for retyping these nodes to the target type before use.
   *
   * @param expr The expression string to parse.
   * @return The parsed IRep2 expression, or a null expr2tc on parse error.
   */
  expr2tc parse_expr(const std::string &expr);

  // Maps annotated unbraced body lines to exclusive end byte offsets.
  // Returns nullopt when token boundaries cannot be determined safely.
  std::optional<std::map<size_t, size_t>> find_control_body_ends(
    const std::string &source,
    const std::set<size_t> &invariant_lines);

  struct LexerContext;

private:
  std::unique_ptr<LexerContext> ctx;
};

#endif

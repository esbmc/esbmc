#ifndef CLANG_C_FRONTEND_CLANG_C_LEXER_H_
#define CLANG_C_FRONTEND_CLANG_C_LEXER_H_

#include <irep2/irep2.h>
#include <memory>
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
   * Integer constants retain their C literal types. Callers are responsible
   * for applying arithmetic conversions after substituting the result.
   *
   * @param expr The expression string to parse.
   * @return The parsed IRep2 expression, or a null expr2tc on parse error.
   */
  expr2tc parse_expr(const std::string &expr);

  struct LexerContext;

private:
  std::unique_ptr<LexerContext> ctx;
};

#endif

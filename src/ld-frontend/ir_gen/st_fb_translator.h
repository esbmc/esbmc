#pragma once

#include <util/std_code.h>
#include <util/std_expr.h>
#include <functional>
#include <string>

// -----------------------------------------------------------------------
// st_fb_translator — minimal IEC 61131-3 Structured Text -> codet translator
// for user-defined function-block bodies referenced by a Ladder Diagram.
//
// Supports the ST subset used by FB bodies in practice:
//   assignment      IDENT := expr ;
//   conditional     IF cond THEN stmt* END_IF ;
//   loop            WHILE cond DO stmt* END_WHILE ;     (kept as a real loop so
//                                                        ESBMC's unwinding
//                                                        assertions detect a
//                                                        non-terminating LLB)
//   expr            IDENT | INT_LIT | TRUE | FALSE | expr <op> expr
//   relop           =  <  <=  >  >=  <>
//   arith           +  -  *  /
// Keywords are case-insensitive.  This is deliberately small: it is enough to
// faithfully translate the function-block bodies that carry Ladder Logic Bombs
// (comparison/arithmetic guards plus a non-termination payload) so that the
// bomb logic reaches the GOTO IR.  Unsupported constructs throw.
// -----------------------------------------------------------------------

class st_fb_translator
{
public:
  // resolve(name) returns the typed symbol_exprt for an identifier; the caller
  // (ld_converter) declares one symbol per FB-local variable and supplies this.
  using resolver_t = std::function<symbol_exprt(const std::string &)>;

  explicit st_fb_translator(resolver_t resolve) : resolve_(std::move(resolve)) {}

  // Translate a whole FB body into a code_blockt.
  code_blockt translate(const std::string &body);

private:
  resolver_t resolve_;
  std::string src_;
  size_t pos_ = 0;
  int wd_counter_ = 0; // unique scan-watchdog counter id per loop

  // lexer
  void skip_ws();
  bool eof();
  std::string peek_word();          // next identifier/keyword (lowercased), no consume
  std::string next_word();          // consume identifier/keyword (original case)
  bool accept_kw(const char *kw);   // consume if next word == kw (case-insensitive)
  void expect_kw(const char *kw);
  bool accept_sym(const char *s);   // consume punctuation/operator if it matches
  void expect_sym(const char *s);

  // parser -> codet/exprt
  void parse_stmt_list(code_blockt &out, const char *const *terminators);
  codet parse_stmt();
  exprt parse_condition();
  exprt parse_expr();   // arithmetic (left-assoc + - * /)
  exprt parse_primary();
};

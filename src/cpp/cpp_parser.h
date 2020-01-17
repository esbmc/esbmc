/*******************************************************************\

Module: C++ Parser

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_PARSER_H
#define CPROVER_CPP_PARSER_H

#include <cassert>
#include <cpp/cpp_parse_tree.h>
#include <cpp/cpp_token_buffer.h>
#include <util/expr.h>
#include <util/parser.h>

class cpp_parsert : public parsert
{
public:
  cpp_parse_treet parse_tree;

  bool parse() override;

  void clear() override
  {
    parsert::clear();
    parse_tree.clear();
    token_buffer.clear();
  }

public:
  // internal state

  enum
  {
    LANGUAGE,
    EXPRESSION
  } grammar;
  enum
  {
    ANSI,
    GCC,
    MSC
  } mode;

  cpp_token_buffert token_buffer;

  cpp_tokent &current_token()
  {
    return token_buffer.current_token();
  }

  void set_location()
  {
    token_buffer.current_token().line_no = line_no - 1;
    token_buffer.current_token().filename = filename;
  }

  cpp_parsert() : mode(ANSI)
  {
  }

  // scanner
  unsigned parenthesis_counter;
};

extern cpp_parsert cpp_parser;
void cpp_scanner_init();

#endif

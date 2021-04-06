/*******************************************************************\

Module: type checking of Solidity AST

Author: Kunjian Song

\*******************************************************************/

#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_AST_CONVERT_H
#define SOLIDITY_AST_FRONTEND_SOLIDITY_AST_CONVERT_H

#include <solidity-ast-frontend/solidity_parse_tree.h>
#include <util/message_stream.h>
#include <util/std_code.h>

bool solidity_ast_convert(
  solidity_parse_treet &solidty_parse_tree,
  const std::string &module,
  message_handlert &message_handler);

#endif

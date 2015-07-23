/*
 * parse.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#include "llvm_parser.h"

class ASTParser
{
public:
  ASTParser(llvm_parsert *_parser)
    : parser(_parser)
  {
  }

  bool parse();

private:
  llvm_parsert *parser;
};

bool ASTParser::parse()
{
  // Use diagnostics to find errors, rather than the return code.
  for (const auto &astunit : parser->ASTs) {
    if (astunit->getDiagnostics().hasErrorOccurred()) {
      std::cerr << std::endl;
      return true;
    }
  }

  return false;
}

bool parse_AST()
{
  ASTParser parser(&llvm_parser);
  return parser.parse();
}

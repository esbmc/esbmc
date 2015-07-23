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
  return true;
}

bool parse_AST()
{
  ASTParser parser(&llvm_parser);
  return parser.parse();
}

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
  // Iterate through each translation unit and their global symbols, creating
  // symbols as we go.

  for (auto &translation_unit : parser->ASTs) {
    clang::ASTUnit::top_level_iterator it;
    for (it = translation_unit->top_level_begin();
        it != translation_unit->top_level_end(); it++) {
      std::cerr << "Got decl kind " << (*it) << std::endl;
    }
  }

  return true;
}

bool parse_AST()
{
  ASTParser parser(&llvm_parser);
  return parser.parse();
}

//
// Created by rafaelsamenezes on 23/09/2021.
//
#include <jimple-frontend/jimple-converter.h>

bool jimple_converter::convert()
{
  exprt dummy_decl = AST.to_exprt(msg, context);
  dummy_decl.dump();
  return true;
}

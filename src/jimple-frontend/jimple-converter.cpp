//
// Created by rafaelsamenezes on 23/09/2021.
//
#include <jimple-frontend/jimple-converter.h>

bool jimple_converter::convert()
{
  AST.to_exprt(context);
  return false;
}

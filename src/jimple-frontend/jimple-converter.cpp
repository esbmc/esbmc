#include <jimple-frontend/jimple-converter.h>

bool jimple_converter::convert()
{
  AST.to_exprt(context);
  return false;
}

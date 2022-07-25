#ifndef ESBMC_JIMPLE_CONVERTER_H
#define ESBMC_JIMPLE_CONVERTER_H

#include <jimple-frontend/jimple-language.h>
class jimple_converter
{
public:
  jimple_converter(contextt &_context, jimple_file &_ASTs)
    : context(_context), AST(_ASTs)
  {
  }
  bool convert();

protected:
  contextt &context;
  const jimple_file &AST;
};

#endif //ESBMC_JIMPLE_CONVERTER_H

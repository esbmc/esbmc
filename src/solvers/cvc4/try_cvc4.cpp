#include <cvc4/cvc4.h>

int main()
{
  CVC4::ExprManager em;
  CVC4::SymbolTable sym_tab;
  printf("%s", CVC4::Configuration::getVersionString().c_str());
}
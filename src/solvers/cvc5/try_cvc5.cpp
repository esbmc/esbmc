#include <cvc5/cvc5.h>

int main()
{
  cvc5::Solver slv;
  printf("%s", slv.getVersion().c_str());
}
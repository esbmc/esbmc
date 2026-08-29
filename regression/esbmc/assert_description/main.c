#include <assert.h>

int nondet_int();

int main()
{
  int x = nondet_int();
  assert(x > 5);
  return 0;
}

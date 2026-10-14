#include <cassert>

extern int nondet_int();

/* The VCC dump must render through expr2cpp, not expr2c: C++ has built-in
   boolean constants, so `flag` reads as true/false rather than 1/0. */
int main()
{
  bool flag = true;
  int x = nondet_int();
  if (x > 0)
    flag = false;
  assert(flag || x > 0);
  return 0;
}

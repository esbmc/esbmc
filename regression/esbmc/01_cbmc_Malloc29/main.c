#include <stdlib.h>

int main ()
{
  int      i;
  char  *words2;

  // Either assertion here should fail; due to a mis-encoded pointer boundry
  // arrangement in the past, an alloc of 0 would result in an overal unsat
  // formula.
  assert(0);
  words2 = malloc (0);
  words2[i] = 0;
}

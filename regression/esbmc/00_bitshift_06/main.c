#include <stdio.h>

void test_shift_fail()
{
  int x = 1;
  x <<= -1; // undefined behavior in C: negative shift count
}

int main()
{
  test_shift_fail();
  return 0;
}

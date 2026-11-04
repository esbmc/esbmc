/* #9371: the G.5.1p4 recovery is for floating element types only; integer
   complex * and / keep the usual formulas. */
#include <assert.h>

int main(void)
{
  int _Complex a, b, r;
  __real__ a = 2;
  __imag__ a = 3;
  __real__ b = 1;
  __imag__ b = 1;
  r = a * b;
  assert(__real__ r == -1 && __imag__ r == 5);
  r = a / b;
  assert(__real__ r == 2 && __imag__ r == 0);
  return 0;
}

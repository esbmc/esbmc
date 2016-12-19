#include <math.h>
#include <float.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main(void)
{
  float f = 123.45;

  float f3;
  float f2 = modff(f, &f3);
  __VERIFIER_assert(f3 == 123.f);
  __VERIFIER_assert(f2 == 0x1.ccccp-2);

  float param, fractpart, intpart;

  param = 3.14159265;
  fractpart = modff(param , &intpart);
  __VERIFIER_assert(intpart == 3.f);
  __VERIFIER_assert(fractpart == 0x1.21fb6p-3);
}


#include <assert.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main(void)
{
  union {
    float a;
    long b;
  } from_union = { .a = 1.0f };

  __VERIFIER_assert(from_union.a == 1.0f);
  __VERIFIER_assert(from_union.b == 1065353216);

  from_union.b++;
  __VERIFIER_assert(from_union.a == 0x1.000002p+0);
}


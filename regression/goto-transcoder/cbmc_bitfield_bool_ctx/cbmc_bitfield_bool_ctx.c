#include <assert.h>
// A _Bool:1 bitfield read in a boolean context (&&, ||, if) must be coerced to
// bool; keeping it as an N-bit value leaves a non-bool operand in the guard.
struct S
{
  _Bool a : 1;
  _Bool b : 1;
};
int main()
{
  struct S s; // nondet
  _Bool r = s.a || s.b;
  if (s.a)
    assert(r);
  if (!s.a && !s.b)
    assert(!r);
  return 0;
}

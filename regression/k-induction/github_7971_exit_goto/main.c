// #7971: the GOTO in front of the ASSERT loop head exits past the loop, so a
// havoc placed there never reaches the loop and the inductive step starts from
// the concrete x == 0. The bug at x == 50 is past the unwinding bound.
#include <assert.h>

int nondet_int();

int main()
{
  int x = 0;
  if (nondet_int())
  {
    x = 7;
    goto out;
  }
  while (1)
  {
    assert(x != 50);
    ++x;
  }
out:
  return 0;
}

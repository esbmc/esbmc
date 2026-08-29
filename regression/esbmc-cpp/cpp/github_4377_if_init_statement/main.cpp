// esbmc/esbmc#4377 (C++17): a variable declared in the init-statement of `if`
// used to lose its initialiser and read as nondeterministic -- a spurious
// counterexample on a deterministic program. Nothing pinned the fix.
#include <cassert>

int main()
{
  if (int x = 7; true)
  {
    assert(x == 7);
  }

  int taken = 0;
  if (int y = 3; y > 2)
    taken = y;
  else
    taken = -y;
  assert(taken == 3);

  if (int z = 1; z > 2)
    taken = 0;
  else
    taken = z * 10;
  assert(taken == 10);
  return 0;
}

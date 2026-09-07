#include <assert.h>

// GCSE inserts a cse symbol into calc, which leaves location numbers stale
// (insert_swap does not renumber). goto_loopst then classified the forward
// branch as a back edge and rewrote it to assume(!guard), so reach_error
// became unreachable and --gcse "proved" this failing program safe.
int g1, g2, out;

void reach_error()
{
  assert(0);
}

void calc(int in)
{
  if (in > 5)
  {
    reach_error();
  }
  out = g1 + g2;
  out = g1 + g2 + 1;
  out = g1 + g2 + 2;
}

int main()
{
  g1 = 1;
  g2 = 2;
  calc(9);
  return 0;
}

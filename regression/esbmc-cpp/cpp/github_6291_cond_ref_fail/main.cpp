// Negative companion to github_6291_cond_ref: writing through the conditional
// reference updates the SELECTED operand (a), not the other (b), so asserting
// the write landed on b is violated.
#include <cassert>

int main()
{
  int a = 1, b = 2;
  int &r = (a < b) ? a : b; // binds to a
  r = 9;
  assert(b == 9); // wrong on purpose: the write lands on a, so b stays 2
  return 0;
}

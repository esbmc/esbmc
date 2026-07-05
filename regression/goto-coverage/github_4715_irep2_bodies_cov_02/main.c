#include <assert.h>

// SUCCESSFUL counterpart to github_4715_irep2_bodies_cov_01: the assert lives in
// a function that main never calls, so under --assertion-coverage no instance is
// reachable (Reached: 0, SUCCESSFUL). It still must be COUNTED -- "Total Asserts:
// 1" -- which only holds if the assert kept its source location across the
// --irep2-bodies body round-trip. Before the convert_expression fix the
// location-less assert was filtered out and the run reported "Total Asserts: 0".
void dead()
{
  int x = 0;
  assert(x == 1);
}

int main()
{
  return 0;
}

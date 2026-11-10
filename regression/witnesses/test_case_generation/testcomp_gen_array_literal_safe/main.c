#include <assert.h>

extern unsigned char __VERIFIER_nondet_uchar(void);
extern int __VERIFIER_nondet_int(void);

unsigned char flags[3] = {1, 0, 1};
int grid[2][2] = {{5, 6}, {7, 8}};

int main()
{
  /* The SUCCESSFUL twin of testcomp_gen_array_literal: same writes, but the
   * assertion is unreachable, so no test case is written. */
  flags[0] = __VERIFIER_nondet_uchar();
  grid[1][0] = __VERIFIER_nondet_int();
  flags[2] = __VERIFIER_nondet_uchar();
  if (flags[2] == 3 && grid[1][0] == 4 && flags[2] == grid[1][0])
    assert(0);
  return 0;
}

#include <assert.h>

extern unsigned char __VERIFIER_nondet_uchar(void);
extern int __VERIFIER_nondet_int(void);

unsigned char flags[3] = {1, 0, 1};
int grid[2][2] = {{5, 6}, {7, 8}};

int main()
{
  /* The initialisers let constant propagation fold each write into an array
   * literal, e.g. `flags == { nondet0, 0, 1 }`. Every call is still a
   * harness input, in call order, including the one the property never
   * reads. */
  flags[0] = __VERIFIER_nondet_uchar();
  grid[1][0] = __VERIFIER_nondet_int();
  flags[2] = __VERIFIER_nondet_uchar();
  if (flags[2] == 3 && grid[1][0] == 4)
    assert(0);
  return 0;
}

/* Past symbolic_chain_bound the `with` chain is dropped, which is silent and
 * indistinguishable from an unexplained hang when the dropped chain also
 * holds the loop counter (#7597). Here the counter is a local, so the loop
 * still folds and the run only has to say once that it gave up. */
#include <assert.h>

int a[1025];

int nondet_int(void);

int main(void)
{
  int i;

  for (i = 0; i < 1025; i = i + 1)
    a[i] = nondet_int();

  assert(i == 1025);
  return 0;
}

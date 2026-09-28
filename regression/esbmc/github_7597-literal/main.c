/* The aggregate-literal arm of #7597, and the only one of the three not gated
 * off under --k-induction / --incremental-bmc. One symbolic element used to
 * drop the whole literal, so the sibling bounding this loop stayed symbolic
 * and the loop never folded. (Under k-induction the verdict is the same
 * either way -- it terminates by the forward condition, not by folding -- so
 * plain BMC is where this arm is observable.) */
#include <assert.h>

struct pair
{
  int n;
  int r;
};

int nondet_int(void);

int main(void)
{
  int v = nondet_int();
  struct pair s = {5, v};
  int c = 0;

  for (int i = 0; i < s.n; i++)
    c++;

  assert(c == 5);
  assert(s.r == v);
  return 0;
}

/* A bounded quantifier whose body indexes an array: the index is in range
 * only under the implication's antecedent, as in Kani's forall!. */
#include <assert.h>
struct seq { unsigned long n; unsigned char digits[8]; };
struct seq nondet_seq(void);
int main(void)
{
  struct seq ret = nondet_seq();
  unsigned char (*p)[8] = &ret.digits;
  __CPROVER_assume(__CPROVER_forall {
    unsigned long i;
    (i < 8) ==> ((*p)[i] <= 9)
  });
  assert(ret.digits[3] <= 8);
  return 0;
}

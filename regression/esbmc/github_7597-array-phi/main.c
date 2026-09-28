/* The merged object is a bare array rather than a struct: the counter is one
 * element and the conditional write targets another, so phi_function's
 * `if(g, then, else)` is over an array. It is decomposed element by element
 * exactly as a struct's members are -- pin_symbolic_updates has already capped
 * the element count, so the rebuild cannot be the expensive one
 * pinned_array_bound exists to refuse. No --unwind: the bound has to come from
 * propagation. */
#include <assert.h>

int A[4];
int IN;

int nondet_int(void);

int main(void)
{
  IN = nondet_int();
  /* A merge point, so IN carries no propagated constant afterwards, and a
   * bound that keeps the sums below from overflowing. */
  if (IN < -1000)
    IN = -1000;
  else if (IN > 1000)
    IN = 1000;

  for (A[0] = 1; A[0] <= 5; A[0] = A[0] + 1)
    if (IN > 0)
      A[1] = IN + 1;
  assert(A[0] == 6);
  /* The merged element still denotes what each branch left: A is a zero-
   * initialised global the loop writes only when IN > 0. */
  assert(A[1] == (IN > 0 ? IN + 1 : 0));

  for (A[0] = 1; A[0] <= 6; A[0] = A[0] + 1)
    if (IN > 0)
      A[2] = IN + 2;
  assert(A[0] == 7);
  assert(A[2] == (IN > 0 ? IN + 2 : 0));

  return 0;
}

/* Same defect through an array: the counter and the symbolically written
 * element are elements of the *same* object, so one refused element write
 * drops the array and the counter with it. #7605 carried a bare read here
 * too; an operator around it was still refused. */
#include <assert.h>

int A[2];

int nondet_int(void);

int main(void)
{
  int x = nondet_int();

  for (A[0] = 1; A[0] <= 5; A[0] = A[0] + 1)
    A[1] = x + 1;

  assert(A[0] == 6);
  return 0;
}

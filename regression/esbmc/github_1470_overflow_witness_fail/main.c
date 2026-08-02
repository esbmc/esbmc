// Regression for github #1470: the GraphML violation witness for an arithmetic
// overflow must record the assignment to the overflowing variable, otherwise
// external validators (UAutomizer/CPAchecker) cannot replay the counterexample.
// Signed `data + 1` overflows iff data == INT64_MAX, so the witness must emit
// the assumption `data == 9223372036854775807` on the assignment edge.
#include <stdint.h>

extern int64_t nondet_int64(void);

int main(void)
{
  int64_t data = nondet_int64();
  int64_t result = data + 1; // overflow when data == INT64_MAX
  return result == 0;
}

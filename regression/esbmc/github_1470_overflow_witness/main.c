// Companion to github_1470_overflow_witness_fail: when the input is constrained
// so that `data + 1` cannot overflow, ESBMC must report no overflow (no spurious
// violation witness). Pins that the overflow check is value-sensitive, not a
// blanket alarm on nondet operands.
#include <stdint.h>

extern int64_t nondet_int64(void);

int main(void)
{
  int64_t data = nondet_int64();
  __ESBMC_assume(data < 9223372036854775807LL);
  int64_t result = data + 1; // cannot overflow: data <= INT64_MAX - 1
  return result == 0;
}

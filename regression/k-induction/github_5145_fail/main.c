// Negative variant for issue #5145: ensures the __CPROVER_uninterpreted_*
// fix does not suppress genuine violations.  After ASSUME(hash != 0) is
// applied, hash is nondet in [1, UINT64_MAX]; the assertion below can be
// violated (e.g. hash == 0x42), so ESBMC must still report FAILED.
#include <assert.h>
#include <stdint.h>

uint64_t __CPROVER_uninterpreted_hasher(const void *a)
{
  return (uint64_t)a;
}

int main(void)
{
  void *key;
  uint64_t hash = __CPROVER_uninterpreted_hasher(key);
  // hash is nondet non-zero after the fix; 0x42 is a valid non-zero value,
  // so this assertion is genuinely reachable.
  assert(hash != 0x42ULL);
  return 0;
}

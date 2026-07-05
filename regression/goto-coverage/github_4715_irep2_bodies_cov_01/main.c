#include <assert.h>

// Assertion coverage under --irep2-bodies (esbmc#4715). The two asserts back
// onto a side_effect_exprt("function_call") for __ESBMC_assert; sideeffect2t
// carries no source location, so the IREP2 body round-trip used to strip the
// ASSERT instruction's location. --assertion-coverage gates its instrumentation
// on the source filename, so a location-less assert was silently dropped and
// the run reported "Total Asserts: 0" / SUCCESSFUL. convert_expression now
// restores the statement location onto the side effect, so both asserts are
// counted and reached (FAILED, because assertion coverage negates them).
void test()
{
  int x = 1;
  for (int i = 0; i < 2; i++)
    assert(x == 1);
}

int main()
{
  test();
}

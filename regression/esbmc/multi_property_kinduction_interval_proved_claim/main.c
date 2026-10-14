/* The twin of multi_property_kinduction_interval_missed_claim with a loop
   assertion that holds: its PASSED row must survive, and assert(1==2) must
   still be reported (esbmc/esbmc#7900). */
#include <assert.h>
int main()
{
  for (int i = 0; i < 2; i++)
  {
    assert(i < 5);
  }
  assert(1 == 2);
  return 0;
}

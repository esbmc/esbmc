/* The twin of multi_property_incremental_final_verdict: no claim is violated
   within --max-k-step, so exhausting the bound must still report
   VERIFICATION UNKNOWN (esbmc/esbmc#7900). */
#include <assert.h>
int main()
{
  unsigned x = 0;
  while (1)
  {
    assert(x != 100);
    ++x;
  }
}

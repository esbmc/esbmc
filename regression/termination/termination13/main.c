/* SV-COMP termination property: CHECK(init(main()), LTL(F end))
 * where `end` is true at exit/abort/return-from-main.
 *
 * `while (1) assert(0)` runs `assert(0)` on the first iteration; the
 * assert aborts the execution, which counts as `end`. Every execution
 * is finite → termination property HOLDS → SUCCESSFUL. */
#include <assert.h>
int main()
{
  while (1)
    assert(0);
  return 0;
}

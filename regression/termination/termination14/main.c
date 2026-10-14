/* SV-COMP termination property: CHECK(init(main()), LTL(F end))
 * where `end` is true at exit/abort/return-from-main.
 *
 * The `break` exits the loop on the first iteration; control reaches
 * `assert(0)`, which aborts. Both are `end` events. Every execution
 * is finite → termination property HOLDS → SUCCESSFUL. */
#include <assert.h>
int main()
{
  while (1)
  {
    break;
  }
  assert(0);
  return 0;
}

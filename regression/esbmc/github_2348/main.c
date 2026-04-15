#include <assert.h>
int main(int argc, char *argv[])
{
  /* argv[0] and argv[1] must be non-null when in range (C11 §5.1.2.2.1).
   * The old model set argv = array_of(NULL), making these assertions
   * produce false negatives.  The fixed model backs argv[0] and argv[1]
   * with nondet char arrays so both assertions succeed. */
  if (argc > 0)
    assert(argv[0]);
  if (argc > 1)
    assert(argv[1]);
  return 0;
}

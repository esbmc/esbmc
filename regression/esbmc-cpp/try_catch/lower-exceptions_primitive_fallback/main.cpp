#include <cassert>

// Primitive throw/catch: the int is copied into the exception storage slot and
// the value-catch handler reads it back (e = *(int*)__ESBMC_exc_value). This
// also covers the case that once crashed address_of on a literal operand.
int main()
{
  try
  {
    throw 42;
  }
  catch (int e)
  {
    assert(e == 42);
    return e;
  }
  return 0;
}

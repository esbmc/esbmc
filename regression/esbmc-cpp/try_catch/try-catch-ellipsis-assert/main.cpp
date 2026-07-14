#include <cassert>

// Regression for a symex segfault: an assert as the *first* instruction of a
// catch(...) handler that catches a thrown object. The catch-all target points
// straight at the ASSERT goto instruction (nil `code` field); the pre-fix
// throw-dispatch dereferenced it and crashed. The assertion holds here.
int main()
{
  int x = 0;
  try
  {
    throw 42;
  }
  catch (...)
  {
    assert(x == 0);
  }
  assert(x == 0);
  return 0;
}

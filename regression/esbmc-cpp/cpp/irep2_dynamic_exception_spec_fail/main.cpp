#include <cassert>

// A dynamic `throw(T...)` specification reaches the adjust pass with its
// declared types unresolved. Left so, the specification permits nothing and the
// throw below reports "exception specification violated" instead of being
// caught. Removed in C++17, hence the pinned mode.
void thrower() throw(int)
{
  throw 1;
}

int main()
{
  int caught = 0;
  try
  {
    thrower();
  }
  catch (int i)
  {
    caught = i;
  }
  assert(caught == 2);
}

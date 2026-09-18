#include <cassert>

// A primitive's exception id is its `#cpp_type` spelling, which the IREP2 seam
// does not carry: computed after it, `throw 1` reads as `signedbv` while the
// handler still reads `signed_int`, and the throw escapes uncaught. A pointer
// throw is the same question one level down.
void thrower()
{
  throw 1;
}

void ptr_thrower(int *p)
{
  throw p;
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
  assert(caught == 1);

  int v = 7;
  int *seen = 0;
  try
  {
    ptr_thrower(&v);
  }
  catch (int *q)
  {
    seen = q;
  }
  assert(seen == &v);
}

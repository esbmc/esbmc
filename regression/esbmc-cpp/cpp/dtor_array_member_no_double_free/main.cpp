#include <cassert>

// A class with a class-type member array is constructed via the frontend's
// `array_init$` helper (a temporary copy of the object).  That helper must not
// be destroyed on its own: doing so runs each element's destructor a second
// time.  With a resource-owning element this is a double-free.  Here each
// element records how many times it is destroyed; every count must be exactly
// one.  Regression for dtor_array_member (was KNOWNBUG, g double-counted to 6).

int destroyed[3];

struct R
{
  int id;
  ~R() { destroyed[id]++; }
};

struct Holder
{
  R rs[3];
  Holder()
  {
    rs[0].id = 0;
    rs[1].id = 1;
    rs[2].id = 2;
  }
};

int main()
{
  {
    Holder h;
  }
  assert(destroyed[0] == 1);
  assert(destroyed[1] == 1);
  assert(destroyed[2] == 1);
  return 0;
}

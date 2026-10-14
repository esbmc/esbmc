#include <cassert>

// A union's implicitly-defined copy/move constructor copies the object
// representation ([class.copy.ctor]/14), which here is one assignment of the
// whole union. The body is generated, not converted, so the pass that replaces
// the generator has to generate it too -- and read the reference parameter
// *through*, since a reference is modelled as a pointer.
union U
{
  int value;
};

int main()
{
  U a;
  a.value = 10;

  U copied(a);
  assert(copied.value == 10);

  U moved(static_cast<U &&>(a));
  assert(moved.value == 10);
}

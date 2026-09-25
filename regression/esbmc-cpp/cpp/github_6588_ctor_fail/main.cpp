#include <cassert>

struct T
{
  int a;
  T()
  {
    a = 7;
  }
};

int main()
{
  // [dcl.init]/8: a class with a user-provided default constructor is
  // value-initialised by calling that constructor, with no prior zeroing --
  // the braced spelling must not be zero-filled on top of it.
  T *t = new T[2]{};
  assert(t[0].a == 0);
  delete[] t;
  return 0;
}

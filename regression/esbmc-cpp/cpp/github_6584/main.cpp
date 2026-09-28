// new T[n] must run one constructor per element, and delete[] one destructor
// per element. Both arms of the lowering were empty stubs, so neither ran and
// members initialised by T's constructor read back nondeterministically.
#include <cassert>

int ctor = 0, dtor = 0;

struct T
{
  int v;
  T()
  {
    ctor++;
    v = 7;
  }
  ~T()
  {
    dtor++;
  }
};

int main()
{
  T *p = new T[4];
  assert(ctor == 4);
  assert(p[0].v == 7);
  assert(p[3].v == 7);
  delete[] p;
  assert(dtor == 4);
  return 0;
}

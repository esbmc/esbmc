#include <cassert>

int n = 0;

struct A
{
  virtual ~A()
  {
    n += 1;
  }
};
struct C : A
{
  ~C()
  {
    n += 10;
  }
};

struct Holder
{
  A *q;
  ~Holder()
  {
    delete q;
  }
};

int main()
{
  {
    Holder h;
    h.q = new C();
  }
  // Virtual dispatch must also work for a delete inside a destructor body.
  assert(n == 11);
  return 0;
}

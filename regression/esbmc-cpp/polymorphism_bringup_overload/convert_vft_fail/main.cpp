#include <cassert>

struct C
{
  virtual int d()
  {
    return 11;
  }
  virtual int d(int)
  {
    return 12;
  }
  virtual int d(float)
  {
    return 13;
  }
  virtual int d(int *)
  {
    return 14;
  }
};

int main()
{
  C c;
  C *cptr = &c;
  assert(cptr->d() == 11);
  assert(cptr->d(1) == 12);
  assert(cptr->d(1.0f) == 13);
  assert(cptr->d(nullptr) == 1400);
}
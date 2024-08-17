#include <cassert>

struct Bottom
{
  virtual int d()
  {
    return 11;
  }
  virtual int d(int *)
  {
    return 14;
  }
};

struct Middle : Bottom
{
  virtual int d(int *) override
  {
    return 24;
  }
};

int main()
{
  Bottom c;
  Bottom *cptr = &c;
  assert(cptr->d() == 11);
  assert(cptr->d(nullptr) == 14);

  Middle m;
  Middle *mptr = &m;
  assert(mptr->d(nullptr) == 24);
}
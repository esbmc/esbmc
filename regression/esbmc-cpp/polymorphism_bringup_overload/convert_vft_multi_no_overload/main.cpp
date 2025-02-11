#include <cassert>

struct Bottom
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

struct Bottom2
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

struct Middle : Bottom, Bottom2
{
  virtual int d(int *) override
  {
    return 24;
  }
  virtual int d()
  {
    return 25;
  }
};

struct Top : Middle
{
  virtual int d() override
  {
    return 31;
  }
  virtual int d(int) override
  {
    return 32;
  }
  virtual int d(int *) override
  {
    return 34;
  }
};

struct Both : Top, Middle
{
  virtual int d() override
  {
    return 41;
  }
  virtual int d(int) override
  {
    return 42;
  }
  virtual int d(int *) override
  {
    return 44;
  }
};

int main()
{
  Bottom c;
  Bottom *cptr = &c;
  assert(cptr->d() == 11);
  assert(cptr->d(1) == 12);
  assert(cptr->d(1.0f) == 13);
  assert(cptr->d(nullptr) == 14);

  Top d;
  Top *dptr = &d;
  assert(dptr->d() == 31);
  assert(dptr->d(1) == 32);
  assert(dptr->d(1.0f) == 32);
  assert(dptr->d(nullptr) == 34);

  Both e;
  Both *eptr = &e;
  assert(eptr->d() == 41);
  assert(eptr->d(1) == 42);
  assert(eptr->d(1.0f) == 42);
  assert(eptr->d(nullptr) == 44);

  Middle f;
  Middle *fptr = &f;
  assert(fptr->d() == 25);
  assert(fptr->d(nullptr) == 24);
}
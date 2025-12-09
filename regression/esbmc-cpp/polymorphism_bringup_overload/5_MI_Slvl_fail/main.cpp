/*
 * multiple inheritance: late binding
 */
#include <cassert>

class Base1
{
public:
  virtual int f(void)
  {
    return 21;
  }
  virtual int f(int)
  {
    return 22;
  }
};

class Base2
{
public:
  virtual int f(void)
  {
    return 42;
  }
  virtual int f(int)
  {
    return 43;
  }
};

class Derived : public Base1, public Base2
{
public:
  virtual int g(void)
  {
    return 100;
  }
  virtual int g(int)
  {
    return 101;
  }
  int f(void)
  {
    return 1;
  }
  int f(int)
  {
    return 2;
  }
};

int main()
{
  Derived *d = new Derived();
  assert(d->Base1::f() == 21);
  assert(d->Base1::f(1) == 22);
  assert(d->Base2::f() == 42);
  assert(d->Base2::f(1) == 43);
  assert(d->g() == 100);
  assert(d->g(1) == 101);
  assert(d->f() == 21); // FAIL, should be 1
  delete d;

  return 0;
}

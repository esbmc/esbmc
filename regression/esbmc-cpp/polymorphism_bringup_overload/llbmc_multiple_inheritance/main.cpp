/*
 * multiple inheritance: able to free, late binding
 */
#include <cassert>

class Base1
{
public:
  virtual ~Base1()
  {
  }
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
  virtual ~Base2()
  {
  }
  virtual int g(void)
  {
    return 21;
  }
  virtual int g(int)
  {
    return 22;
  }
};

class Derived : public Base1, public Base2
{
public:
  virtual ~Derived()
  {
  }
  virtual int f(void)
  {
    return 42;
  }
  virtual int f(int)
  {
    return 43;
  }
  virtual int g(void)
  {
    return 42;
  }
  virtual int g(int)
  {
    return 43;
  }
};

int main()
{
  Base2 *o = new Derived();
  int r = o->g();
  int s = o->g(1);
  delete o;
  assert(r == 42);
  assert(s == 43);
  return r + s;
}

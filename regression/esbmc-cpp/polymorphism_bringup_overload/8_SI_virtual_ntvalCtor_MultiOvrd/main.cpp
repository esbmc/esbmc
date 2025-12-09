/*
 * Polymorphism:
 *  - derived class contains two overriding methods
 *  - one of the overriding methods has `virtual` keyword, but the other doesn't
 */
#include <cassert>

class Bird
{
public:
  virtual int do_something(void)
  {
    return 21;
  }
  virtual int do_something(int)
  {
    return 22;
  }
  virtual int do_other(void)
  {
    return 25;
  }
  virtual int do_other(int)
  {
    return 26;
  }
};

class Penguin : public Bird
{
public:
  virtual int do_something(void)
  {
    return 42;
  }
  int do_something(int)
  {
    return 43;
  }
  int do_other(void)
  {
    return 50;
  }
  virtual int do_other(int)
  {
    return 51;
  }
};

int main()
{
  Bird *b = new Bird();
  Bird *p = new Penguin();
  assert(b->do_something() == 21);
  assert(b->do_something(1) == 22);
  assert(p->do_something() == 42);
  assert(p->do_something(1) == 43);
  assert(b->do_other() == 25);
  assert(b->do_other(1) == 26);
  assert(p->do_other() == 50);
  assert(p->do_other(1) == 51);
  delete b;
  delete p;
  return 0;
}

/*
 * Polymorphism:
 *  - derived class contains an overriding method
 *  - the overriding method has `virtual` keyword
 *  - non-trivial dtor
 */
#include <cassert>

int a = 5;

class Bird
{
public:
  virtual int do_something(void)
  {
    return 10;
  }
  virtual int do_something(int)
  {
    return 11;
  }
  virtual ~Bird()
  {
    a = do_something(1);
    assert(a == 11);
    a = do_something();
    assert(a == 10);
  }
};

class Penguin : public Bird
{
public:
  virtual int do_something(void)
  {
    return 15;
  }
  virtual int do_something(int)
  {
    return 16;
  }
  ~Penguin()
  {
    a = do_something(1);
    assert(a == 16);
    a = do_something();
    assert(a == 15); // should be calling Penguin::do_something
  }
};

int main()
{
  Bird *b = new Bird();
  assert(b->do_something() == 10);
  assert(b->do_something(1) == 11);
  delete b;
  assert(a == 10);

  Bird *p = new Penguin();
  assert(p->do_something() == 15);
  assert(p->do_something(1) == 16);
  delete p;
  assert(a == 15); // should be 10, not 15

  return 0;
}

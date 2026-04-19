#include <cassert>

class Bird
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
  virtual int g(void)
  {
    return 21;
  }
  virtual int g(int)
  {
    return 22;
  }
};

class FlyingBird : public Bird
{
public:
  virtual void fly()
  {
  }
  virtual void fly(int)
  {
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

class Penguin : public Bird
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

int main()
{
  Bird *b = new Bird();
  FlyingBird *f = new FlyingBird();
  Penguin *p = new Penguin();
  assert(b->f() == 21);
  assert(b->f(1) == 22);
  assert(f->g() == 42);
  assert(f->g(1) == 43);
  assert(p->g() == 21);
  assert(p->g(1) == 22);
  assert(p->f() == 21); // FAIL, should be 42
  delete b;
  delete f;
  delete p;
  return 0;
}

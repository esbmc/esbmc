/*
 * dynamic cast, late binding, passing ptr to function
 */
#include <cassert>

int x = 0;
int y = 0;

struct A
{
  virtual void f()
  {
    x += 1;
  }
  virtual void f(int)
  {
    y += 1;
  }
};

struct B : A
{
  virtual void f()
  {
  }
  virtual void f(int)
  {
  }
};

struct C : A
{
  virtual void f()
  {
    x += 2;
  }
  virtual void f(int)
  {
    y += 2;
  }
};

void f(A *arg)
{
  B *bp = dynamic_cast<B *>(arg);
  C *cp = dynamic_cast<C *>(arg);

  if (bp)
  {
    bp->f();
    bp->f(1);
  }
  else if (cp)
  {
    cp->f();
    cp->f(1);
  }
  else
  {
    arg->f();
    arg->f(1);
  }
};

int main()
{
  A aobj;
  C cobj;
  A *ap = &cobj;
  A *ap2 = &aobj;
  f(ap);          // x += 2; y += 2
  f(ap2);         // x +=1; y += 1
  assert(x == 1); // FAIL, should be 3
}

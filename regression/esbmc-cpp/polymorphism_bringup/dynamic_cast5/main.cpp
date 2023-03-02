/*
 * dynamic cast, late binding, passing ptr to function
 */
#include <cassert>

int x = 0;

struct A {
  virtual void f() { x+=1;}
};

struct B : A {
  virtual void f() { }
};

struct C : A {
  virtual void f() { x+=2;}
};

void f(A* arg) {
  B* bp = dynamic_cast<B*>(arg);
  C* cp = dynamic_cast<C*>(arg);

  if (bp)
    bp->f();
  else if (cp)
    cp->f();
  else
    arg->f();
};

int main() {
  A aobj;
  C cobj;
  A* ap = &cobj;
  A* ap2 = &aobj;
  f(ap);  // x += 2;
  f(ap2); // x +=1
  assert(x==3);
}


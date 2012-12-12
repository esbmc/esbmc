#include <cassert>

class Base1 {
  public:
  virtual int f(void) { return 21; }
};

class Base2 {
  public:
    virtual int f(void) { return 42; }
};

class Derived: public Base1, public Base2 {
  public:
    virtual int g(void) { return 42; }
};

int main(){
  Derived *d = new Derived();
  assert(d->Base1::f() == 21);
  assert(d->Base2::f() == 42);
  assert(d->g() == 42);
  delete d;

  return 0;
}


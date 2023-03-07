/*
 * Polymorphism:
 *  - derived class contains an overriding method
 *  - the overriding method has `virtual` keyword
 *  - trival ctor
 */
#include <cassert>

class Bird {
  public:
  virtual int do_something(void) { return 21; }
};

class Penguin: public Bird {
  public:
    virtual int do_something(void) { return 42; }
};

int main(){
  Bird *b = new Bird();
  Bird *p = new Penguin();
  assert(b->do_something() == 21);
  assert(p->do_something() == 42);
  delete b;
  delete p;
  return 0;
}


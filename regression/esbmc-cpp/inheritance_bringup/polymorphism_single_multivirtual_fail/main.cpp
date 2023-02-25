/*
 * Polymorphism:
 *  - derived class contains an overriding method
 *  - the overriding method has `virtual` keyword
 */
#include <cassert>

class Bird {
  public:
  virtual int do_something(void) { return 21; }
  virtual int do_other(void) { return 25; }
};

class Penguin: public Bird {
  public:
    virtual int do_something(void) { return 42; }
    int do_other(void) { return 50; }
};

int main(){
  Bird *b = new Bird();
  Bird *p = new Penguin();
  assert(b->do_something() == 21);
  assert(p->do_something() == 42);
  assert(b->do_other() == 25);
  assert(p->do_other() == 25); // FAIL, should be 50
  delete b;
  delete p;
  return 0;
}


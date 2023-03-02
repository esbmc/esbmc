/*
 * Polymorphism:
 *  - derived class contains an overriding method
 *  - the overriding method has `virtual` keyword
 *  - non-trivial dtor
 */
#include <cassert>

int a = 5;

class Bird {
  public:
  virtual int do_something(void) { return 21; }
};

class Penguin: public Bird {
  public:
    virtual int do_something(void) { return 42; }
    ~Penguin() { a = 10; }
};

int main(){
  Bird *b = new Bird();
  Bird *p = new Penguin();
  assert(b->do_something() == 21);
  assert(p->do_something() == 42);
  delete b;
  delete p;
  assert(a == 10);
  return 0;
}


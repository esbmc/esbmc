/*
 * A simple test case of polymorphism
 */
#include <cassert>

class Bird {
  public:
  virtual int do_something(void) { return 21; }
  //virtual ~Bird(){}
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


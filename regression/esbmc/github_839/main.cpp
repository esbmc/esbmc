#include <cassert>

class Bird {
  public:
  virtual int f(void) { return 21; }
  virtual int g(void) { return 21; }
  virtual ~Bird(){}
};

class FlyingBird: public Bird {
  public:
    virtual void fly();
    virtual int g(void) { return 42; }
};

class Penguin: public Bird {
  public:
    virtual int f(void) { return 42; }
};

void FlyingBird::fly() { }

int main(){
  Bird *b = new Bird();
  FlyingBird *f = new FlyingBird();
  Penguin *p = new Penguin();
  assert(b->f() == 21);
  assert(f->g() == 42);
  assert(p->f() == 42);
  assert(p->g() == 21);
  delete b;
  delete f;
  delete p;
  return 0;
}


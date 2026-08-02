// C++ [class.cdtor]/4: a virtual call made from a constructor or destructor
// resolves to the override in the class currently being constructed/destroyed,
// NOT the most-derived override.  ESBMC used to set the vptr only at the end of
// the constructor body (and never in destructors), so calls during
// construction/destruction wrongly dispatched to the derived class.
#include <cassert>

int ctor_seen;
int dtor_seen;

struct A
{
  virtual int who() { return 1; }
  A() { ctor_seen = who(); }        // must call A::who -> 1
  virtual ~A() { dtor_seen = who(); } // must call A::who -> 1
};

struct B : A
{
  int who() override { return 2; }
};

int main()
{
  B b;                       // A() runs with the A-subobject vptr
  assert(ctor_seen == 1);

  A *p = new B();            // heap object: single destruction on delete
  delete p;                  // ~B then ~A; ~A runs with the A vptr
  assert(dtor_seen == 1);

  // Post-construction dispatch is still most-derived.
  A *q = new B();
  assert(q->who() == 2);
  delete q;
  return 0;
}

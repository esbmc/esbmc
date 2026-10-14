// github #6310: typeid on a polymorphic glvalue must reflect the *dynamic*
// type. It used to be keyed on the operand's static type, so typeid(*p) with
// p an A* answered "A" whatever p pointed at.
#include <cassert>
#include <cstring>
#include <typeinfo>

struct A
{
  virtual ~A()
  {
  }
};
struct B : A
{
};
struct C : A
{
};
struct D : B
{
};

struct M1
{
  virtual ~M1()
  {
  }
};
struct M2
{
  virtual ~M2()
  {
  }
};
struct MD : M1, M2
{
};

extern bool nondet_bool();

int main()
{
  A *p = new B();
  A *q = new C();
  A *r = new D();
  A a;
  B b;
  A &ref = b;

  assert(typeid(*p) == typeid(B));
  assert(typeid(*p) != typeid(A));
  assert(typeid(*q) == typeid(C));
  assert(typeid(*p) != typeid(*q));
  assert(typeid(*r) == typeid(D)); // two levels down
  assert(typeid(*r) != typeid(B));
  assert(typeid(ref) == typeid(B)); // through a base reference
  assert(typeid(ref) == typeid(*p));
  assert(typeid(*p) != typeid(int));

  // A most-derived operand still answers its own type.
  assert(typeid(a) == typeid(A));
  assert(typeid(b) == typeid(B));
  assert(typeid(a) != typeid(b));

  // name() follows the dynamic type too.
  assert(strcmp(typeid(*p).name(), typeid(B).name()) == 0);

  delete p;
  delete q;
  delete r;

  // Either base subobject's vptr answers with the most-derived type.
  MD md;
  M1 *m1 = &md;
  M2 *m2 = &md;
  assert(typeid(*m1) == typeid(MD));
  assert(typeid(*m2) == typeid(MD));
  assert(typeid(*m1) == typeid(*m2));
  assert(typeid(*m2) != typeid(M2));

  // The dynamic type is not a compile-time constant here.
  bool c = nondet_bool();
  A *s = c ? (A *)new C() : (A *)new B();
  assert((typeid(*s) == typeid(C)) == c);
  assert(typeid(*s) != typeid(A));
  delete s;

  return 0;
}

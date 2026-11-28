// &f() for a call returning T& is the address of the referent.
#include <cassert>

bool nondet_bool();

int &ref(int *p)
{
  return *p;
}

struct Box
{
  int v;
  int &get()
  {
    return v;
  }
  int &operator*()
  {
    return v;
  }
};

struct Holder
{
  int &ref;
};

int main()
{
  int a = 0, b = 0;
  int *q = &ref(&a);
  assert(q == &a);
  *q = 1;
  assert(a == 1);

  int &r = ref(&b);
  assert(&r == &b);
  assert(&ref(&r) == &b);

  Holder h{a};
  assert(&h.ref == &a);

  int *p = &a;
  int *&rp = p;
  assert(&rp == &p);

  Box box{0};
  assert(&box.get() == &box.v);
  assert(&*box == &box.v);
  *&box.get() = 2;
  assert(box.v == 2);

  bool c = nondet_bool();
  int *s = &(c ? ref(&a) : ref(&b));
  assert(s == (c ? &a : &b));
  return 0;
}

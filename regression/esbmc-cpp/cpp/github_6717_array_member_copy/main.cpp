// Copying a class that holds an array of class type goes through clang's
// ArrayInitLoopExpr, whose sub-expression is the element's copy constructor
// rather than the indexed read a scalar element yields. The frontend cast it
// to an index anyway, walked off the end of the expression and segfaulted
// during conversion. Reduced from g++.dg/init/array37.C and init/array4.C
// (issue #6717).
struct A
{
  int i;
};

struct B
{
  A ar[3];
};

int check(B x)
{
  return x.ar[1].i;
}

struct C
{
  A ar[2];
  C()
  {
  }
  C(const C &) = default;
};

int main()
{
  B b;
  b.ar[0].i = 7;
  b.ar[1].i = 8;
  b.ar[2].i = 9;

  B b2(b);
  __ESBMC_assert(
    b2.ar[0].i == 7 && b2.ar[1].i == 8 && b2.ar[2].i == 9,
    "every element is copied, not just the first");

  // Passing by value takes the same path.
  __ESBMC_assert(check(b) == 8, "by-value parameter copies the array");

  // So does an explicitly defaulted copy constructor.
  C c;
  c.ar[1].i = 4;
  C c2(c);
  __ESBMC_assert(c2.ar[1].i == 4, "defaulted copy constructor");
  return 0;
}

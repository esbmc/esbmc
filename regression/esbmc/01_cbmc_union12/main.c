// Test union assignment, nondeterministically.

union cheese {
  int a;
  struct {
    short a;
    short b;
  } b;
};

struct face {
  union cheese bees;
};

struct foo {
  int *bar;
  int baz[2];
};

struct qux {
  struct face quux;
  struct foo xyzzy;
};

struct qux youwhat;

int
main()
{
  union cheese a = { 0 };
  if (nondet_bool())
    a.a = 1;

  youwhat.quux.bees = a;
  if (nondet_bool())
    youwhat.quux.bees.a = 2;
  assert(youwhat.quux.bees.a == 0);
  return 0;
}

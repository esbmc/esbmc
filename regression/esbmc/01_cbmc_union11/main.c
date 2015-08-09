// Test union assignment, through pointer dereference.
// Take a union in a struct; create a pointer to it; dereference.
// The result will be an array assignment to an array.

union cheese {
  int a;
  struct {
    short a;
    short b;
  } b;
};

struct qux {
  union cheese bees;
};

struct qux youwhat;

int
main()
{
  struct qux *uwotm8 = &youwhat;
  union cheese a = { 0 };
  if (nondet_bool()) // Knacker constant propagation
    a.a = 1;

  uwotm8->bees = a;
  if (nondet_bool())
    uwotm8->bees.a = 2;
  assert(uwotm8->bees.a == 0);
  return 0;
}

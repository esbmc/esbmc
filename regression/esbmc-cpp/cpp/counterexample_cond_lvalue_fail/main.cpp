// A conditional lvalue is not a chain of component accesses over the object
// symex rewrote, so the step reports the value it assigned.
extern "C" bool nondet_bool();

int a, b;

int main()
{
  bool c = nondet_bool();
  (c ? a : b) = 5;
  __ESBMC_assert(a != 5 && b != 5, "one of them is 5");
  return 0;
}

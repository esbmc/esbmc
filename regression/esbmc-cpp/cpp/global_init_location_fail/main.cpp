/* Negative counterpart: the located global initializer must still be a real
 * nondet assignment the solver can violate. */
int nondet_int();

int A = nondet_int();

int main()
{
  __ESBMC_assert(A == 42, "global is not always 42");
  return 0;
}

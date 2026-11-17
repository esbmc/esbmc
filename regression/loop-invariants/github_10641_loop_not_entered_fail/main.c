/* The loop is entered only when c holds, and the model that overflows x takes
   the path around it: the violation does not run the havoc, so it is a real
   bug rather than a weak invariant (#10641). */
extern int __VERIFIER_nondet_int(void);
int main()
{
  int c = __VERIFIER_nondet_int();
  int x = __VERIFIER_nondet_int();
  int i = 0;
  if (c)
  {
    __ESBMC_loop_invariant(0 <= i && i <= 4);
    while (i < 4)
      i++;
  }
  else
    x = x + 1;
  return x;
}

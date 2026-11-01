/* Issue #9656: a termination proof by ranking function must write the
 * correctness witness, as the forward-condition proof does. */

extern int __VERIFIER_nondet_int(void);

int main()
{
  int n = __VERIFIER_nondet_int();
  while (n > 0)
    n = n - 1;
  return 0;
}

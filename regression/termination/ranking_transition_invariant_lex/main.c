/* Neither y - x nor 2147483647 - z falls on every iteration, but the pair
 * falls lexicographically: the transition invariant is a disjunction with
 * one disjunct per measure. */
extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  int y = __VERIFIER_nondet_int();
  int z = __VERIFIER_nondet_int();
  while (x < y && z < 2147483647)
  {
    if (x < z)
      x++;
    else
      z++;
  }
  return 0;
}

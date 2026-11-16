/* The ranking proof's measure goes into the correctness witness as a
 * transition invariant on the outer loop; the inner loop restarts on every
 * outer iteration, so its measure alone would not hold and it gets none. */
extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  int y;
  while (x > 0)
  {
    y = 10;
    while (y > 0)
      y--;
    x--;
  }
  return 0;
}

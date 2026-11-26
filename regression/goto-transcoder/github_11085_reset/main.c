/* The contract holds at the outer head only, not inside the inner loop that
 * goto-cc gives its own head (#11085). */
int main(void)
{
  unsigned x = 0, n = 0;
  for (;;)
    __CPROVER_loop_invariant(x == 0 && n <= 3)
    {
      while (x < 5)
        x++;
      x = 0;
      if (n >= 3)
        break;
      n++;
    }
  __CPROVER_assert(n == 3, "n is 3");
  return 0;
}

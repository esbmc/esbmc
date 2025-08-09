_Bool __VERIFIER_nondet_bool();

int main()
{
  unsigned int x1 = 0, x2 = 0;
  int s = 1;

  __ESBMC_loop_invariant(s >= 1 && s <= 4);
  while (__VERIFIER_nondet_bool())
  {
    if (s == 1)
      x1++;
    else if (s == 2)
      x2++;

    s++;
    if (s == 5)
      s = 1;
  }
  if (s >= 4)
  {
    // Invalid safety property (s may be 4)
    assert(0);
  }
}

/* Safe: i == 0 whenever L is reached. Without --multi-property the entry jump
   past assert(0) must stay unhavoced, or this proof is lost (esbmc/esbmc#7900). */
int nondet_int();
int main()
{
  int i = 0, x = 0;
  if (nondet_int())
  {
    x = 1;
    goto L;
  }
  while (nondet_int())
  {
    i = 0;
    continue;
  L:
    if (i == 0)
      goto out;
    __ESBMC_assert(0, "v");
  }
out:
  return x;
}

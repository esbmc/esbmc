/* #11006: the invariant holds on entry but is not preserved by the body. */
int main()
{
  unsigned i = 0, s = 0;
  while (i < 2)
    __CPROVER_loop_invariant(s == 0)
  {
    s += i;
    i++;
  }
  __CPROVER_assert(s <= 1, "s stays small");
  return 0;
}

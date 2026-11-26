/* A call in the arguments of the invariant's call is lifted too. */
unsigned id(unsigned x)
{
  return x;
}
_Bool closure(unsigned x)
{
  return x <= 4;
}
int main()
{
  unsigned i = 0;
  while (i < 4)
    __CPROVER_loop_invariant(closure(id(i)))
  {
    i++;
  }
  return 0;
}

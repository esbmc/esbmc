int main()
{
  int x;
  __CPROVER_assume(x > 0);
  __CPROVER_assume(x < 0);
  __CPROVER_assert(0, "unreachable under contradictory assumptions");
  return 0;
}

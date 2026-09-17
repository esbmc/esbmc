int main()
{
  int x;
  __CPROVER_assume(x > 0 && x < 10);
  int y = (x > 5) ? x : x + 1;
  __CPROVER_assert(y > 0, "y positive");
  __CPROVER_assume(y != 3);
  __CPROVER_assert(y != 3, "assume constrains later");
  return 0;
}

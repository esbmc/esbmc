int g = 0;
int bump(void) { g = g + 1; return 2; }
int main()
{
  int x = bump() + bump();
  __CPROVER_assert(x == 4, "both calls evaluated");
  __CPROVER_assert(g == 2, "both side effects took place");
  return 0;
}

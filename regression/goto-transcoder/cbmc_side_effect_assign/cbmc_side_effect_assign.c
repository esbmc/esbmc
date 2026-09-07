int main()
{
  int y = 0;
  int x = (y = 3) + 1;
  __CPROVER_assert(x == 4, "outer value");
  __CPROVER_assert(y == 3, "the embedded assignment took effect");
  return 0;
}

int main()
{
  struct Local { unsigned a : 3; unsigned b : 5; };
  struct Local s;
  s.a = 9;
  s.b = 2;
  __CPROVER_assert(s.a == 1, "3-bit field truncates 9 to 1");
  __CPROVER_assert(s.b == 2, "second field intact");
  return 0;
}

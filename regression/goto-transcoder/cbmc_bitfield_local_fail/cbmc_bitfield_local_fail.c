int main()
{
  struct Local { unsigned a : 3; unsigned b : 5; };
  struct Local s;
  s.a = 9;
  s.b = 2;
  __CPROVER_assert(s.a == 9, "wrong: no truncation");
  __CPROVER_assert(s.b == 2, "second field intact");
  return 0;
}

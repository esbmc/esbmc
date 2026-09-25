int main()
{
  struct Outer {
    int x;
    struct { unsigned p : 3; unsigned q : 5; };  /* anonymous, function-local */
    union { int i; float f; };
  } s;
  s.x = 1;
  s.p = 9;
  s.i = 7;
  __CPROVER_assert(s.x == 1, "named member");
  __CPROVER_assert(s.p == 9, "wrong");
  __CPROVER_assert(s.i == 7, "anonymous union member");
  return 0;
}

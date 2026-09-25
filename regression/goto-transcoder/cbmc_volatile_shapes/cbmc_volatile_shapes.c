volatile int g;
struct S { volatile int m; };
int main()
{
  struct S s; s.m = 1;
  volatile int *p = &g;
  g = 5;
  int a = *p;
  __CPROVER_assert(a == 5, "volatile pointer read sees the write");
  __CPROVER_assert(s.m == 1, "volatile member keeps its value");
  return 0;
}

// f() writes a or b through a conditional lvalue, so the return must not put
// either back.
int a, b;
int c;
void f()
{
  (c ? a : b) = 5;
}
int main()
{
  a = 0;
  b = 0;
  c = nondet_int();
  f();
  __ESBMC_assert(a == 0 && b == 0, "ab");
}

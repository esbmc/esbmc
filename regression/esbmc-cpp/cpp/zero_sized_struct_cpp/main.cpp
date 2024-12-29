extern "C" void memmove(void *, void *, int);
// This struct is zero-sized and cpp-only
class a
{
  // int a;
};
void __assert_fail(char *, int, int, int);
struct b
{
  int c;
  a d;
};
int main()
{
  b e;
  memmove(&e.d, 0, 1);
  if (e.c)
    __assert_fail("", 0, 0, 0);
}

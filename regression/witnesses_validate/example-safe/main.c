extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *) __attribute__((__nothrow__, __leaf__)) __attribute__((__noreturn__));
void reach_error() { __assert_fail("0", "multivar_1-1.c", 3, "reach_error"); }

int main()
{
  unsigned n = __VERIFIER_nondet_uint();
  unsigned x = __VERIFIER_nondet_uint();
  unsigned y = n - x;
  while (x > y)
  {
    x--;
    y++;
    if (x + y != n)
      reach_error();
  }
  return 0;
}

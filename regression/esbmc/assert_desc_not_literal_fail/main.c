// The assertion is still checked when its description is not a literal, and
// the report falls back to the guard rather than an empty message. #1557
int nondet_int(void);

static void checked(const char *why, int ok)
{
  __ESBMC_assert(ok, why);
}

int main(void)
{
  int x = nondet_int();
  checked("x must be positive", x > 0);
  return 0;
}

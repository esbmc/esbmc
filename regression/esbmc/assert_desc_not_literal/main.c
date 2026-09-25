// An __ESBMC_assert whose description is a runtime pointer rather than a string
// literal used to abort the whole run out of get_string_constant. Only the
// assertion's description is read from there, so it degrades to a warning and
// verification proceeds. #1557
int nondet_int(void);

static void checked(const char *why, int ok)
{
  __ESBMC_assert(ok, why);
}

int main(void)
{
  int x = nondet_int();
  checked("x equals itself", x == x);
  return 0;
}

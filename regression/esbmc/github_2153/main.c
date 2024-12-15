int some_var;

int main()
{
  void *bar;
  char *ptr = &bar;
  ptr[0] = nondet_char();

  __ESBMC_assert(bar != &some_var, "");
}

int main()
{
  char buf[4] = "abc";
  char *p = nondet_bool() ? "abc" : buf;
  __ESBMC_assume(p == buf);
  __ESBMC_init_object(p);
  return 0;
}

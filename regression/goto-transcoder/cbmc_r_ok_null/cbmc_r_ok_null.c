int main()
{
  char *p = 0;
  __CPROVER_assert(!__CPROVER_r_ok(p, 1), "NULL is not readable");
  return 0;
}

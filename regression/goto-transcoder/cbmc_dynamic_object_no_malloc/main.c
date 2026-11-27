/* A binary that never allocates still reads a zero-initialised
 * __ESBMC_is_dynamic. */
int global;

int main(void)
{
  int local;
  __CPROVER_assert(!__CPROVER_DYNAMIC_OBJECT(&local), "local is not dynamic");
  __CPROVER_assert(!__CPROVER_DYNAMIC_OBJECT(&global), "global is not dynamic");
  return 0;
}

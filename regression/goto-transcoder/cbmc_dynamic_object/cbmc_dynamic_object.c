/* __CPROVER_DYNAMIC_OBJECT lowers to an is_dynamic_object expr with no
   migrate handler. ESBMC's faithful equivalent is its symex-managed
   __ESBMC_is_dynamic array, but wiring that up on the --binary path is future
   work (roadmap §4.4); until then ESBMC must decline cleanly, not abort(). */
int main(void)
{
  int x = 5;
  __CPROVER_assert(!__CPROVER_DYNAMIC_OBJECT(&x), "stack object is not dynamic");
  return 0;
}

/* __CPROVER_array_set / __CPROVER_havoc_object lower to whole-object OTHER
   codet statements (ARRAY_SET / HAVOC_OBJECT) that carry no explicit length --
   the extent comes from the pointee type, which ESBMC's memset/array machinery
   does not reconstruct here. migrate would abort(); the adapter now declines
   cleanly instead (roadmap §4.4). CBMC's own memset lowering is retargeted to
   __ESBMC_memset before its ARRAY_SET body runs, so it is unaffected. */
int main(void)
{
  int a[4];
  __CPROVER_array_set(a, 7);
  __CPROVER_assert(a[0] == 7, "array_set");
  return 0;
}

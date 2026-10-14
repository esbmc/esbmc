/* __CPROVER_array_set lowers to a whole-object ARRAY_SET codet that carries no
   explicit length -- the extent is the pointee array's own. When that array is
   an entire object the adapter rewrites the instruction into an array_of fill
   of exactly that extent (roadmap §4.4); shapes whose object is not statically
   recoverable are still declined, see cbmc_array_set_member. CBMC's own memset
   lowering is retargeted to __ESBMC_memset before its ARRAY_SET body runs, so
   it is unaffected. */
int main(void)
{
  int a[4];
  __CPROVER_array_set(a, 7);
  __CPROVER_assert(a[0] == 7, "array_set");
  return 0;
}

/* CBMC constrains each object's base address to its ABI alignment; ESBMC left
   scalar/stack objects unconstrained, so a natural-alignment cast check was
   satisfiable-false and ESBMC reported a spurious misalignment where CBMC
   proves it (roadmap §4.3, the corpus NonNull::as_ref mismatch). The reader now
   sets the alignment attribute the SMT layer honours. */
int g_int;

int main(void)
{
  int x;
  long y;
  int a[4];
  struct S { int m; double n; } s;

  __CPROVER_assert((unsigned long)&x % 4 == 0, "int stack alignment");
  __CPROVER_assert((unsigned long)&y % 8 == 0, "long stack alignment");
  __CPROVER_assert((unsigned long)&a[1] % 4 == 0, "array element alignment");
  __CPROVER_assert((unsigned long)&s.n % 8 == 0, "struct field alignment");
  __CPROVER_assert((unsigned long)&g_int % 4 == 0, "global alignment");
  return 0;
}

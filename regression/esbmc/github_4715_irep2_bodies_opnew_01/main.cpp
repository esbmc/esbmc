// Exercises a bare `operator new(sizeof(T))` call under --irep2-bodies
// (V.4.4, esbmc#4715). The size argument is a folded `sizeof` constant whose
// element type rides the "#c_sizeof_type" attribute; that attribute is dropped
// when the body round-trips through IREP2, leaving the allocation a nil-subtype
// `new void` that crashes the scalar zero-initializer. The operator-new
// lowering now falls back to a zero-initialised integer spanning the requested
// bytes, so the freshly allocated storage still reads back as zero.
#define NULL 0
int *p;
int main()
{
  p = (int *)operator new(sizeof(int));
  __ESBMC_assert(p != NULL, "allocation is non-null");
  __ESBMC_assert(*p == 0, "operator new storage reads back as zero");
  return 0;
}

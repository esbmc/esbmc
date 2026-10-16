/* C-suite KNOWNBUG: realloc fails to preserve byte content from old block.
 *
 * Focused reproducer for the realloc byte-content propagation gap identified
 * via Option gamma chain probe
 * (notes/napp/heap_byte_provenance/option_gamma_chain_probe.md). No dependence
 * on heap-byte-CONCAT-on-pointer-array -- purely tests realloc's byte
 * preservation semantics.
 *
 * Per C99 7.22.3.5: realloc must preserve the contents of the old object up
 * to the lesser of the old and new sizes.
 *
 * ESBMC's symex_realloc (src/goto-symex/builtin_functions.cpp:156) invokes
 * copy_memory_content which iterates min(old_elem_count, new_elem_count) via
 * index2tc(elem_type, ...). When elem_type does not equal uint8 (e.g. inferred
 * as size_t or wider from a (size_t*) cast site or pointer alignment), the
 * element-stride copy may miss bytes inside the strided element boundary.
 * SMT model leaves them unconstrained, picks any non-equal value, and the
 * post-realloc byte read can violate the assertion.
 *
 * KNOWNBUG until realloc encoding is fixed (Stage 2 zeta).
 */
#include <assert.h>
#include <stdlib.h>

int main(void)
{
  char *p = (char *)malloc(8);
  __ESBMC_assume(p != 0);
  p[0] = 42;
  char *q = (char *)realloc(p, 16);
  __ESBMC_assume(q != 0);
  assert(q[0] == 42);
  return 0;
}

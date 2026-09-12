/* Reading through a struct's zero-length array member (the GCC extension) drove
 * dereferencet::stitch_together_from_byte_array with num_bytes == 0. Its only
 * guard was an assert, which NDEBUG compiles out of the shipping build, so the
 * stitching loop read bytes[-1] and ESBMC died with SIGSEGV instead of
 * producing a verdict. It now fails loudly in every build.
 *
 * The refusal is not the end state -- clang runs this program and the assertion
 * holds -- but a diagnosable error beats an out-of-bounds read inside the
 * verifier. */
#include <assert.h>
#include <stdlib.h>

struct e
{
  void *k, *v;
};
struct s
{
  void *a;
  struct e slots[0];
};

int main(void)
{
  struct s tmpl;
  tmpl.a = (void *)0x1234;
  struct s *p = calloc(1, sizeof(struct s) + 2 * sizeof(struct e));
  if (!p)
    return 0;
  *p = tmpl;
  assert(((unsigned char *)&p->slots[0])[1] == 0);
  return 0;
}

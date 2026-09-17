/* The tagged-slot idiom, as the dereference layer actually presents it:
 * union accesses get normalized to the first member, so a value written
 * and read through the SAME field still reaches the simplifier as a
 * cross-member read of the literal. Same-width integer members share
 * their two's-complement representation, so the read must fold — before
 * the fix the receiver index below never resolved and the loop unrolled
 * to the bound. */
typedef union { short s; unsigned short r; } slot_t;
struct fr { slot_t stack[16]; unsigned char sp; };
static struct fr f;
static unsigned short tbl[4] = { 0, 3, 0, 0 };

static int walk(struct fr *fr)
{
  unsigned short ref = fr->stack[0].r;
  int sum = 0;
  for (unsigned short i = 0; i < tbl[ref]; i++)
    sum++;
  return sum;
}

int main(void)
{
  f.stack[0].r = 1;
  f.sp = 1;
  __ESBMC_assert(walk(&f) == 3, "the slot value resolved and bounded the loop");
  return 0;
}

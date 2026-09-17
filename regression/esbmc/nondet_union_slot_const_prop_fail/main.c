/* Negative twin of nondet_union_slot_const_prop: the union member
 * that actually holds the nondet must still be symbolic — asserting a
 * concrete value for it has to be refuted, proving the relaxed union
 * propagation folds only what is truly constant and never invents
 * values for symbol-valued members. */
typedef unsigned short u2;
typedef union
{
  short s;
  u2 r;
} slot_t;
u2 nondet_u2(void);

struct fr
{
  unsigned char sp;
  slot_t stack[16];
};
static struct fr f;

int main(void)
{
  f.stack[0].r = nondet_u2();
  f.sp = 1;
  __ESBMC_assert(f.stack[0].r == 5, "a nondet slot is not 5");
  return 0;
}

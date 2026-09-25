/* What rooting the path at its parameter does NOT reach. The check holds every
 * other field of *o unchanged, which catches a write to o->x, but the pointee
 * of o->sub is not a parameter, so Phase 2C has nothing to root a snapshot at
 * and a write to o->sub->b goes unreported.
 *
 * Recording `sub` as assignable is also weaker than the clause, which names
 * only o->sub->a: a body that reassigns o->sub itself verifies too. */
typedef struct { int a; int b; } Inner;
typedef struct { Inner *sub; int x; } Outer;

void write_sub_a(Outer *o, int v)
{
  __ESBMC_requires(o != (void *)0 && o->sub != (void *)0);
  __ESBMC_assigns(o->sub->a);
  __ESBMC_ensures(1);
  o->sub->a = v;
  o->sub->b = 42; /* not in __ESBMC_assigns */
}

int main()
{
  return 0;
}

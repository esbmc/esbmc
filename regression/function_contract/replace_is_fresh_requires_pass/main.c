/* replace_is_fresh_requires_pass: a callee whose requires contains
 * __ESBMC_is_fresh must be dischargeable under --replace-call-with-contract.
 *
 * Regression for #6380: the replace path asserted the callee's requires clause
 * verbatim, but __ESBMC_is_fresh was only lowered on the assume/enforce side.
 * On the assert side its return-value temp was left undefined, so the requires
 * assertion failed vacuously ("Could not find definition for temporary
 * variable return_value$___ESBMC_is_fresh"), no matter what the caller passed.
 *
 * Here touch_outer is enforced (its own is_fresh(o) allocates a valid Outer),
 * and touch_inner is replaced at both call sites. Its is_fresh(&o->a) /
 * is_fresh(&o->b) preconditions are now lowered to valid_object() checks on the
 * actual arguments, which hold because o->a and o->b are valid sub-objects of
 * the fresh o. Expected: SUCCESSFUL.
 */
typedef struct { int x; } Inner;
typedef struct { Inner a; Inner b; } Outer;

void touch_inner(Inner *p)
{
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(Inner)));
  __ESBMC_assigns(p->x);
  __ESBMC_ensures(p->x == 0);
  p->x = 0;
}

void touch_outer(Outer *o)
{
  __ESBMC_requires(__ESBMC_is_fresh(o, sizeof(Outer)));
  __ESBMC_assigns(o->a, o->b);
  __ESBMC_ensures(o->a.x == 0 && o->b.x == 0);
  touch_inner(&o->a);
  touch_inner(&o->b);
}

int main(void) { return 0; }

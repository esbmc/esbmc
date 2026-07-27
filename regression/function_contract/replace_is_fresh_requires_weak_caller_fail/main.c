/* replace_is_fresh_requires_weak_caller_fail: the requires-side is_fresh lowered
 * at a replace call site must be a REAL check, not vacuously true (#6380).
 *
 * touch_outer promises only o != NULL -- strictly weaker than fresh -- and
 * cannot establish that o->a is a valid object, so touch_inner's is_fresh(&o->a)
 * precondition is genuinely not discharged. Expected: FAILED, on the
 * "contract requires / VALID_OBJECT(&o->a)" property. If this ever reports
 * SUCCESSFUL the assert-side lowering has regressed to a vacuous pass.
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
  __ESBMC_requires(o != ((void *)0));
  __ESBMC_assigns(o->a);
  __ESBMC_ensures(1);
  touch_inner(&o->a);
}

int main(void) { return 0; }

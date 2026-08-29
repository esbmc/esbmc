/* is_fresh_two_ptrs_pass:
 * Two pointer parameters, each independently set and verified via ensures.
 *
 * Non-nullness alone does not make this contract provable. Pointer parameters
 * may alias (#6551), and with p == q the writes below leave p->x == 20, so the
 * first ensures fails. Separation has to be stated, which is what
 * __ESBMC_is_fresh does: it gives each parameter its own object, and a replace
 * site has to discharge that the caller's arguments really are separate.
 */

typedef struct { int x; } S;

void f(S *p, S *q)
{
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(S)));
  __ESBMC_requires(__ESBMC_is_fresh(q, sizeof(S)));
  __ESBMC_ensures(p->x == 10);
  __ESBMC_ensures(q->x == 20);

  p->x = 10;
  q->x = 20;
}

int main()
{
  S a, b;
  f(&a, &b);
  return 0;
}

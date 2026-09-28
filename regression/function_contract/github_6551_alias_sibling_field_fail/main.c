/* github_6551_alias_sibling_field_fail:
 * Pointer parameters may alias, so the assigns frame check exempts a
 * snapshotted parameter that aliases an assigns target's base. That exemption
 * is matched on the field name, and this pins how narrow it is: 'other' is not
 * 'coeffs', so writing it must still be reported however r and b alias. */
typedef struct { int coeffs[4]; int other; } P;

void f(P *r, P *b)
{
  __ESBMC_requires(r != 0);
  __ESBMC_requires(b != 0);
  __ESBMC_assigns(r->coeffs);
  __ESBMC_ensures(1);
  r->coeffs[0] = 1;
  r->other = 7;
}
int main(void) { return 0; }

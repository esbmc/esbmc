/* Test: assigns-only contract (no requires/ensures) — compliance FAIL
 *
 * Regression for Bug 1: functions with only __ESBMC_assigns (and no
 * __ESBMC_requires or __ESBMC_ensures) were previously skipped by
 * enforce_contracts because has_contracts() did not scan for assigns_target
 * sideeffect instructions.
 *
 * The function declares __ESBMC_assigns(g) but also writes to 'h', which
 * is not in the assigns clause.  Enforce mode must detect this violation.
 */

int g = 0;
int h = 0;

void foo(int v)
{
    __ESBMC_assigns(g);

    g = v;
    h = 99; /* VIOLATION: h is not in the assigns clause */
}

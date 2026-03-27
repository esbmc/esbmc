/* Test: struct field assigns compliance — FAIL
 *
 * Regression for Bug 2: the frame enforcer must generate a per-field
 * assertion for the field NOT in the assigns clause (global_pt.y), and
 * detect that the function body writes to that field.
 *
 * The function declares __ESBMC_assigns(global_pt.x) but also writes to
 * global_pt.y (set to val + 1, which is nondeterministic and non-zero in
 * general).  Enforce mode must detect this violation.
 */

typedef struct
{
    int x;
    int y;
} Point;

Point global_pt;

void set_x(int val)
{
    __ESBMC_requires(1);
    __ESBMC_assigns(global_pt.x);
    __ESBMC_ensures(global_pt.x == val);

    global_pt.x = val;
    global_pt.y = val + 1; /* VIOLATION: .y is not in the assigns clause */
}

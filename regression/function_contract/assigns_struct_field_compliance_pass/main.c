/* Test: struct field assigns compliance — PASS
 *
 * Regression for Bug 2: __ESBMC_assigns(global_pt.x) previously caused a
 * false positive because enforce_frame_rule generated a coarse whole-struct
 * assertion (ASSERT global_pt == snap_global_pt) instead of a per-field
 * assertion.  Only global_pt.x is permitted to change; global_pt.y must
 * remain unchanged.
 *
 * The function correctly writes only global_pt.x, so compliance must PASS.
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

    global_pt.x = val; /* only .x is modified — compliance must hold */
}

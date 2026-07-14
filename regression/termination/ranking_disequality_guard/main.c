// Termination via a `!=` countdown guard. Mirrors the Cairo shape from
// termination-crafted: an outer `if (x > 0)` makes x strictly positive
// at loop entry, then `while (x != 0) x = x - 1;` decrements to zero.
// The certifier sees the `!=` and emits two direction candidates;
// the `(x - 0)` candidate with the relaxed invariant `x >= 0`
// (weakened from the path-cond seed `x > 0`) discharges.

extern int __VERIFIER_nondet_int(void);

int main(void)
{
    int x = __VERIFIER_nondet_int();
    if (x > 0) {
        while (x != 0) {
            x = x - 1;
        }
    }
    return 0;
}

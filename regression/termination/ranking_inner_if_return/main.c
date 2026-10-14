// Loop body has an `if (cond) return val;` arm that exits the loop
// early. Previously Shape B rejected the loop because the THEN arm
// did not end in a forward GOTO to a merge label. The inner-if-return
// extension treats the early-return arm as a path that exits the
// loop, contributing no ranking obligation; only the continue arm
// (path-condition: the IF's jump-guard) generates a ranking path.
//
// Lifted from termination-memory-alloca/b.03_assume-alloca.

extern int __VERIFIER_nondet_int(void);

int main(void)
{
    int x = __VERIFIER_nondet_int();
    int y = __VERIFIER_nondet_int();
    while (x > y && y <= 2147483647 - x) {
        if (x <= 0) {
            return y;
        }
        y = y + x;
    }
    return y;
}

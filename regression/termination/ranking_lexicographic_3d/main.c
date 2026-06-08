// Three-counter loop where the lex tuple (x1, x2, x3) decreases on
// every body path. Guard has three && conjuncts (3 candidates). Body
// has three sequential ifs — exactly one fires per iteration because
// the guards partition the integer line. Path-conditions cover all
// 2^3 = 8 enumerated paths; the infeasible ones (no decrement, two
// decrements, etc.) get UNSAT path-cond and are vacuously discharged.

extern int __VERIFIER_nondet_int(void);

int test_fun(int c, int x1, int x2, int x3)
{
    while (x1 >= 1 && x2 >= 1 && x3 >= 1) {
        if (c == 1) {
            x1 = x1 - 1;
        }
        if (c == 2) {
            x2 = x2 - 1;
        }
        if (c != 1 && c != 2) {
            x3 = x3 - 1;
        }
    }
    return x1 + x2 + x3;
}

int main(void)
{
    return test_fun(
        __VERIFIER_nondet_int(),
        __VERIFIER_nondet_int(),
        __VERIFIER_nondet_int(),
        __VERIFIER_nondet_int());
}

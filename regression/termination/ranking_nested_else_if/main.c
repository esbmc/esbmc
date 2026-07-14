// Three-counter loop with a nested else-if body shape
// `if (c1) S1 else if (c2) S2 else S3`. Before multi-level if/else
// support, Shape B rejected the loop because the outer IF's else
// arm contained another IF (a forward GOTO inside the arm that the
// straight-line collector cannot handle).
//
// The three-way decomposition produces three paths:
//   Path 1: c1 fires        → x1 decreases
//   Path 2: !c1 ∧ c2        → x2 decreases
//   Path 3: !c1 ∧ !c2       → x3 decreases
// All paths lex-decrease (x1, x2, x3), so the 3-D lex pass discharges.

extern unsigned int __VERIFIER_nondet_uint(void);

int main(void)
{
    unsigned int x1 = __VERIFIER_nondet_uint();
    unsigned int x2 = __VERIFIER_nondet_uint();
    unsigned int x3 = __VERIFIER_nondet_uint();
    while (x1 > 0 && x2 > 0 && x3 > 0) {
        if (x1 > x2) {
            x1 = x1 - 1;
        } else if (x2 > x3) {
            x2 = x2 - 1;
        } else {
            x3 = x3 - 1;
        }
    }
    return 0;
}

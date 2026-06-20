// eca-shape benchmark where EVERY (state, input) hits a branch, so the
// fall-through `return -2` is unreachable. There's no period-1 fixpoint,
// the detector must return UNKNOWN and the program-level verdict
// follows from the ranking / k-induction pass (here we just need to
// confirm we don't get a wrong "VERIFICATION FAILED").

extern int __VERIFIER_nondet_int(void);

int update(int input)
{
    // input in {1, 2} → some branch always fires.
    if (input == 1) return 10;
    if (input == 2) return 20;
    return -2;
}

int main(void)
{
    int sink = 0;
    while (1) {
        int input = __VERIFIER_nondet_int();
        if (input != 1 && input != 2)
            return -2;
        sink = update(input);
    }
    return sink;
}

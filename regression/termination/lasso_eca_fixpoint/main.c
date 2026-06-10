// eca-rers2012 shape: while(1) reads validated input, calls an updater
// whose branches all transfer out via RETURN or exit(). The detector
// must find a (state, input) with NO branch firing — fall-through to
// `return -2` leaves state unchanged, demon picks that input forever,
// non-terminating. Period-1 fixpoint SMT discharge proves it.

int a = 1;
int b = 2;

extern int __VERIFIER_nondet_int(void);
extern void exit(int);

int update(int input)
{
    if (a == 99 && input == 1) {
        a = 0;
        return 0;
    }
    if (b == 99) {
        exit(0);
    }
    return -2;
}

int main(void)
{
    int sink = 0;
    while (1) {
        int input;
        input = __VERIFIER_nondet_int();
        if (input != 1 && input != 2)
            return -2;
        sink = update(input);
    }
    return sink;
}

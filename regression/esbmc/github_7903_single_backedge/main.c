int main(void) {
    unsigned r = 0;
_start:
    {
        unsigned t;
        if (r != 0) return 0;
        t = 1;
        if (nondet_int() != 0) goto after;
        r = t; goto _start;
    after:
        r = t;
        __ESBMC_assert(r == 1, "r == 1");
        return 0;
    }
}

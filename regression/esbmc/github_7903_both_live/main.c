int main(void) {
    unsigned r = 0;
_start:
    {
        unsigned t;
        if (r >= 2) { __ESBMC_assert(r == 2, "r == 2"); return 0; }
        t = r + 1;
        if (nondet_int() != 0) { r = t; goto _start; }
        r = t; goto _start;
    }
}

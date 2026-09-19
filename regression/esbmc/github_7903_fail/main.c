int main(void) {
    unsigned r = 0;
_start:
    {
        unsigned t = r + 5;
        if (r != 0) { __ESBMC_assert(r == 5, "r == 5"); return 0; }
        if (nondet_int() != 0) { r = t; goto _start; }
        r = 15 - t; goto _start;
    }
}

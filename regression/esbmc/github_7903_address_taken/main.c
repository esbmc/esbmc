// #7903: after the merge, *p writes the parked path's instance of t while the
// read of t uses the other path's, so the store through p is lost.
int main(void) {
    unsigned r = 0;
_start:
    {
        unsigned t;
        unsigned *p = &t;
        if (r >= 2) { __ESBMC_assert(r == 2, "r == 2"); return 0; }
        t = r + 1;
        if (nondet_int() != 0) { r = t; goto _start; }
        *p = t + 5;
        r = t - 5; goto _start;
    }
}

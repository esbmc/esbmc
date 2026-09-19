extern int nondet_int(void);
extern void __ESBMC_assert(_Bool, const char *);
int main(void) {
    unsigned r = 0;
_start:
    {
        unsigned t;
        if (r != 0) { __ESBMC_assert(r == 1, "r == 1"); return 0; }
        t = 1;
        if (nondet_int() != 0) { r = t; goto _start; }
        r = t; goto _start;
    }
}

typedef union{
    unsigned long long raw64;
    struct {
        unsigned raw32lo;
        unsigned raw32hi;
    };
    struct {
        unsigned long long    field1:1;
        unsigned long long    field2:1;
    };
} union_t;


unsigned long long foo(unsigned addr) {
    return addr + 100;
}

int main() {
  unsigned nondet = __VERIFIER_nondet_uint();
   union_t x = (union_t) foo(nondet);
   __ESBMC_assert(x.raw64 == (nondet+100), "Initialized correctly");
   return 0;
}
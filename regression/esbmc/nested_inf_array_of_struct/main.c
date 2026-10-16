// IR-near-equivalent of Solidity nested-mapping-of-struct.
// lvl2 has irep2 type array<array<Balance,1>,infinite> — a nested array
// with an infinite level and a struct (tuple) leaf, the same shape class
// that Solidity's mapping(K=>mapping(K=>Struct)) produces.  A write
// builds a with2t chain over it, exercising the SMT tuple_api->mk_struct_sort
// path on a nested-infinite-array-of-tuple, exactly as the Solidity case.
typedef struct { unsigned _BitInt(248) amount; unsigned char tokensCount; } Balance;

typedef Balance Row[1];
__attribute__((annotate("__ESBMC_inf_size")))
extern Row grid[1];          // array<array<Balance,1>, infinite>

void dock(unsigned long a, unsigned long keys[], unsigned n) {
  for (unsigned i = 0; i < n; i++) {
    if (grid[a][0].tokensCount == n)        // read struct field through nested-inf array
      grid[a][0].tokensCount = 0xff;        // write struct field -> with2t chain
  }
}
int main(){ unsigned long k[2]={1,2}; dock(7,k,2); return 0; }

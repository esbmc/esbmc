a;
typedef struct {
  int b;
  int c[];
} d;


main() {   
  d e = {a, {}};
  __ESBMC_assert(0, ""); }

/* #1507: ESBMC crashed parsing this. d.f has no elements, so d.f->b accesses
 * one, which C17 6.7.2.1p18 leaves undefined (#5393). */
struct a {
  long b
};
struct c d;
struct c {
  short e;
  struct a f[]
} main() {
  d.f->b;
}

/* The other side of the open lower bound: -1.0f truncates to -1, which no
   unsigned type represents, so this conversion stays undefined. Pinned so a
   fix for github_7572_trunc cannot be a blanket relaxation.
   clang -fsanitize=float-cast-overflow: "-1 is outside the range of
   representable values of type 'unsigned int'". */
extern float nondet_float(void);

int main(void)
{
  float a = nondet_float();
  __ESBMC_assume(a == -1.0f);
  unsigned int u = (unsigned int)a;
  (void)u;
  return 0;
}

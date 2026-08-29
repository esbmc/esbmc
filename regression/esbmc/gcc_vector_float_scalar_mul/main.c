typedef float v4f __attribute__((vector_size(16)));

/* vector-by-scalar, which gcc_vector_float_arith does not reach: the scalar
 * operand is broadcast, so a fold returning it would type the result wrongly.
 * Both operand orders, since a fold may inspect either side. */
int main()
{
  v4f a = {1.5f, -2.5f, 3.5f, 4.5f};
  v4f zr = a * 0.0f;
  v4f zl = 0.0f * a;
  v4f o = a * 1.0f;
  __ESBMC_assert(zr[0] == 0.0f && zr[2] == 0.0f, "scalar zero on the right");
  __ESBMC_assert(zl[0] == 0.0f && zl[2] == 0.0f, "scalar zero on the left");
  __ESBMC_assert(o[1] == -2.5f, "scalar one broadcasts");
  return 0;
}

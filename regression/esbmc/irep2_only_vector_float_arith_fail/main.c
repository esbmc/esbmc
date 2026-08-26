/* The promoted node carries the real float arithmetic per lane, so a wrong
   expected lane must be caught. */
typedef float v4f __attribute__((vector_size(16)));

int main(void)
{
  v4f a = {1.5f, 2.5f, 3.5f, 4.5f};
  v4f b = {0.5f, 0.5f, 0.5f, 0.5f};

  v4f m = a * b;
  __ESBMC_assert(m[1] == 2.5f, "m[1] is 1.25, not 2.5");
  return 0;
}

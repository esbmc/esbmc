unsigned int nondet_uint();
int nondet_int();

int main()
{
  unsigned int SIZE = 1;
  unsigned int j, k;
  int array[SIZE], menor;

  menor = nondet_int();

  __ESBMC_loop_invariant(j <= SIZE && menor <= array[0]);
  for (j = 0; j < SIZE; j++)
  {
    array[j] = nondet_int();

    if (array[j] <= menor)
      menor = array[j];
  }

  assert(array[0] >= menor);
}

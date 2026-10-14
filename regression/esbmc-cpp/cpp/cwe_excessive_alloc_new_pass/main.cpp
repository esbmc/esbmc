extern unsigned nondet_uint();

int main()
{
  // Passing companion to cwe_excessive_alloc_new: operator new[n] with a bounded
  // n must succeed. n <= 256 => 256 * sizeof(int) = 1024 bytes, under the 1 MiB
  // default. Guards the new[] byte scaling against over-counting.
  unsigned n = nondet_uint();
  if (n <= 256)
  {
    int *p = new int[n];
    delete[] p;
  }
  return 0;
}

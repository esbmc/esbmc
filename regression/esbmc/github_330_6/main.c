void g(void *a, unsigned n) {
  free(a); // unsafe
}

void f (unsigned n)
{
  void *a [8];
  for (int i = 0; i != 8; ++i)
    a [i] = __builtin_alloca (n);

  g(a, n);
}

unsigned nondet_uint();
int main() {
  f(nondet_uint());
  return 0;
}

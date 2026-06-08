// Loop body with one branch, adapted from regression/goto-coverage/k_path_cov_2/.
// Uses __VERIFIER_nondet_int() so the generated CTest cases compile.
extern "C" int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  for (int i = 0; i < 3; i++)
  {
    if (x > 0)
      x = x - 1;
    else
      x = x + 1;
  }
  return x;
}

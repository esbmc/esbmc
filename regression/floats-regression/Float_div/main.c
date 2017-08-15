extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

// Replace 1.6f to 2.5f to verification successful
#define X 2.5f

int main()
{
  float x = 1.0f;
  float x1 = x/X;

  while(x1 != x)
  {
    x = x1;
    x1 = x/X;
  }

  __VERIFIER_assert(x == 0);
  return 0;
}


// Regression test for GitHub issue #3854 (loop variant):
// Spurious "no body for function ~T#" warning with loop control flow
// even when main() has an explicit return 0.
class Num
{
private:
  int val;

public:
  explicit constexpr Num(int num) : val(num) {}

  Num() : val(nondet_int()) {}

  Num inc() const { return Num(val + 1); }

  bool operator!=(const Num &rhs) { return this->val != rhs.val; }
};

int main()
{
  Num n0;
  Num n1 = n0.inc();

  for (int idx = 0; idx < 2; ++idx)
  {
    n0 = n1.inc();
    n1 = n0.inc();
  }

  // After 2 iterations: n0.val = init+3, n1.val = init+4 — always differ
  __ESBMC_assert(n0 != n1, "n0 and n1 are never equal after loop");
  return 0;
}

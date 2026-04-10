// Regression test for GitHub issue #3854:
// Spurious "no body for function ~T#" warning for classes with
// implicit trivial destructors when main() lacks an explicit return.
class Num
{
private:
  int val;

public:
  explicit constexpr Num(int num) : val(num) {}

  Num() : val(nondet_int()) {}

  Num inc() const { return Num(val + 1); }

  bool operator==(const Num &rhs) { return this->val == rhs.val; }
};

int main()
{
  Num n0;
  Num n1 = n0.inc();

  // n0.val + 1 == n1.val, so n0 != n1: this assertion should always fail
  __ESBMC_assert(!(n0 == n1), "n0 and n1 are not equal");

  // No explicit return — triggers implicit destructor calls on scope exit.
  // This must not produce a "no body for function ~Num#" warning.
}

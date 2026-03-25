// Regression test for GitHub issue #3854 (expected failure):
// A class with an implicit trivial destructor where an assertion is violated.
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

  // n0.val + 1 == n1.val, so n0 == n1 is always false: violation expected
  __ESBMC_assert(n0 == n1, "n0 equals n1 after inc");

  // No explicit return — tests implicit destructor call path
}

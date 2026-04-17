// Regression test for GitHub issue #3854 (explicitly-defaulted destructor):
// Verify no "no body for function ~T#" warning for an explicitly-defaulted
// trivial destructor (~Num() = default), which Clang also doesn't synthesize
// a body for until it is ODR-used.
class Num
{
private:
  int val;

public:
  explicit constexpr Num(int num) : val(num) {}

  Num() : val(nondet_int()) {}

  ~Num() = default;

  Num inc() const { return Num(val + 1); }

  bool operator!=(const Num &rhs) { return this->val != rhs.val; }
};

int main()
{
  Num n0;
  Num n1 = n0.inc();

  __ESBMC_assert(n0 != n1, "n0 and n1 always differ after inc");

  // No explicit return — triggers implicit destructor calls on scope exit.
}

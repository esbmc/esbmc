/* github_6212_receiver_extent_pass:
 * The implicit receiver keeps a one-element backing when every other struct
 * pointer parameter lost one. C++ guarantees `this` addresses one complete
 * object, so the extent is the language's promise rather than one the
 * contract failed to state, and a method contract must not have to say it.
 *
 * __ESBMC_is_fresh is not the alternative here: it would also assert `this`
 * is separate from every other pointer parameter, which no contract states.
 */
class Counter
{
public:
  __ESBMC_contract
  int bump()
  {
    __ESBMC_requires(n_ >= 0 && n_ < 100);
    __ESBMC_ensures(__ESBMC_return_value == __ESBMC_old(n_) + 1);
    n_ = n_ + 1;
    return n_;
  }

private:
  int n_;
};

int main()
{
  return 0;
}

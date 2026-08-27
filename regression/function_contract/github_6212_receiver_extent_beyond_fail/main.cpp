/* github_6212_receiver_extent_beyond_fail:
 * Negative control for github_6212_receiver_extent_pass. The receiver keeps a
 * one-element backing, and one element is all C++ promises, so an access past
 * it must still be caught.
 */
class Counter
{
public:
  __ESBMC_contract
  void bump()
  {
    __ESBMC_requires(n_ >= 0);
    __ESBMC_ensures(1);
    this[3].n_ = 1;
  }

private:
  int n_;
};

int main()
{
  return 0;
}

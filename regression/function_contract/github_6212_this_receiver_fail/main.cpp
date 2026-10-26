/* github_6212_this_receiver_fail:
 *   `this` is backed by exactly one object, so reaching past it is caught.
 */
class Counter
{
public:
  __ESBMC_contract void bump()
  {
    __ESBMC_requires(n_ >= 0 && n_ < 100);
    __ESBMC_ensures(1);
    this[1].n_ = 0;
  }

private:
  int n_;
};

int main()
{
  return 0;
}

/* github_6212_this_receiver_pass:
 *   A member function may assume `this` addresses one object of its class, so
 *   reading a data member needs no __ESBMC_is_fresh on the receiver, unlike a
 *   struct pointer parameter (github_6212_struct_extent_fail).
 */
class Counter
{
public:
  __ESBMC_contract void bump()
  {
    __ESBMC_requires(n_ >= 0 && n_ < 100);
    __ESBMC_ensures(n_ == __ESBMC_old(n_) + 1);
    n_++;
  }

private:
  int n_;
};

int main()
{
  return 0;
}

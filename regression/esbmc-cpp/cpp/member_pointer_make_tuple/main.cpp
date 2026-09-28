// Two make_tuple specialisations over member pointers of different types
// collided and aborted symex in assert_type_compat_for_with (G14).
#include <vector>
#include <tuple>
class f
{
  int b;
  static constexpr auto c = std::make_tuple(&f::b);
};
class d
{
  std::vector<int> e;
  static constexpr auto c = std::make_tuple(&d::e);
};
int main()
{
}

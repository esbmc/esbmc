#include <cassert>
#include <cstdlib>
int order;
struct G
{
  int id;
  G(int i) : id(i) {}
  ~G() { order = order * 10 + id; assert(order != 21); }
};
G a(1), b(2);
int main()
{
  exit(0);
}

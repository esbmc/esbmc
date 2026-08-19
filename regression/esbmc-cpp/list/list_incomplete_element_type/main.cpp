// [container.requirements.general] (N4510, adopted for C++17) requires vector,
// list and forward_list -- and only those three -- to accept an incomplete
// element type. The models held their storage in by-value member arrays, which
// need the element complete at instantiation, so the recursive member that
// tree-shaped code declares did not parse. Contrast #7029: map is *not* obliged
// to accept one, and libstdc++ doing so is a QoI extension.
#include <list>
#include <forward_list>
#include <vector>
#include <cassert>

struct tree_node
{
  std::vector<tree_node> vec_kids;
  std::list<tree_node> list_kids;
  std::forward_list<tree_node> flist_kids;
  int id;
};

int main()
{
  tree_node n;
  n.id = 7;
  assert(n.id == 7);
  assert(n.vec_kids.empty());
  assert(n.list_kids.empty());
  assert(n.flist_kids.empty());
  assert(n.list_kids.size() == 0);
  return 0;
}

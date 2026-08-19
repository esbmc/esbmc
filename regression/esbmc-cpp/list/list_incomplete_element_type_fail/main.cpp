// Anti-vacuity twin of list_incomplete_element_type: the containers still hold
// elements once the type is complete, so a wrong claim about one must fail.
#include <list>
#include <cassert>

struct tree_node
{
  std::list<tree_node> kids;
  int id;
};

int main()
{
  std::list<int> l;
  l.push_back(1);
  l.push_back(2);
  assert(l.size() == 3);
  return 0;
}

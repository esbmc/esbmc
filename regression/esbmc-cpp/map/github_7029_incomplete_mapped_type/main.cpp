// esbmc/esbmc#7029: instantiating std::map<K, T> must not require T to be
// complete. A recursive `map<K, Node>` member is how tree-shaped code -- the
// `named_subt` of ESBMC's own irept among it -- declares its children.
#include <cassert>
#include <map>

struct node
{
  typedef std::map<int, node> named_subt;
  named_subt named_sub;
  int id;
};

int main()
{
  node n;
  n.id = 7;
  assert(n.id == 7);
  assert(n.named_sub.size() == 0);
  assert(n.named_sub.empty());
  return 0;
}

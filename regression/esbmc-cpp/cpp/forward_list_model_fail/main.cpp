// Negative counterpart of forward_list_model: the container really holds the
// elements pushed into it, so a claim contradicting the traversal order is
// refuted rather than vacuously held (issue #5868).
#include <forward_list>

int main()
{
  std::forward_list<int> l;
  l.push_front(2);
  l.push_front(1);
  __ESBMC_assert(l.front() == 2, "must not hold");
  return 0;
}

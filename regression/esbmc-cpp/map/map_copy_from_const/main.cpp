// [map.overview]: std::map's copy constructor takes `const map&`. The model
// took `map&`, so copying a const map was ill-formed -- which is what irept's
// `dt` copy constructor does when it initialises its std::map members from a
// `const dt &`, and the last thing between ESBMC and parsing its own irep.h
// after #7029.
#include <cassert>
#include <map>

struct dt
{
  std::map<int, int> named_sub;
  std::map<int, int> comments;
  int data;

  dt() : data(0)
  {
  }
  dt(const dt &other)
    : named_sub(other.named_sub), comments(other.comments), data(other.data)
  {
  }
};

int main()
{
  std::map<int, int> m;
  m[1] = 7;

  const std::map<int, int> &cref = m;
  std::map<int, int> copy(cref);
  assert(copy.size() == 1);
  assert(copy[1] == 7);

  dt d;
  d.data = 3;
  d.named_sub[2] = 9;
  const dt &cd = d;
  dt dup(cd);
  assert(dup.data == 3);
  assert(dup.named_sub[2] == 9);
  assert(dup.comments.empty());
  return 0;
}

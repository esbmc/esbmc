// Byte-granular access to a struct with a char-keyed map forces ESBMC to
// address a single byte of the map object. Under --ir the struct has no
// bit-vector representation, and flattening it used to abort the solver with
// "Z3 error operator is applied to arguments of the wrong sort".
#include <map>
#include <cassert>

int main()
{
  std::map<char, int> m;
  m['a'] = 1;
  m['b'] = 2;
  assert(m['a'] == 1);
  assert(m['b'] == 2);
  return 0;
}

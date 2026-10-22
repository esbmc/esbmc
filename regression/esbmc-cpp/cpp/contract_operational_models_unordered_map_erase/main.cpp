// Replaced by its contract, erase removes at most one entry: the size relation
// holds, an exact size does not follow.
#include <unordered_map>
#include <cassert>
int nondet_int();
int main(){ std::unordered_map<int,int> m; m[1]=2; int k=nondet_int(); size_t r = m.erase(k); assert(m.size() == 1 - r); assert(m.size() == 0); return 0; }

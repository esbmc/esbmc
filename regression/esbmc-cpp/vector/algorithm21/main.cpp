// replace_copy example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

template<class InIt, class OutIt, class Ty>
OutIt replace_copy(InIt first, InIt last, OutIt dest, const Ty& oldval,
		const Ty& newval) {
	for (; first != last; ++first, ++dest)
		*dest = (*first == oldval) ? newval : *first;
	return dest;
}

template<class InIt, class OutIt, class Ty>
OutIt replace_copy(InIt *first, InIt *last, OutIt dest, const Ty& oldval,
		const Ty& newval) {
	for (; first != last; ++first, ++dest)
		*dest = (*first == oldval) ? newval : *first;
	return dest;
}

int main () {
  int myints[] = { 10, 20, 30, 30, 20, 10, 10, 20 };

  vector<int> myvector (8);
  replace_copy (myints, myints+8, myvector.begin(), 20, 99);

  assert(myvector[1] == 99);
  cout << "myvector contains:";
  for (vector<int>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;
 
  return 0;
}

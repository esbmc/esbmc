// remove_copy example
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

template<class InIt, class OutIt, class Ty>
OutIt remove_copy(InIt first, InIt last, OutIt dest, const Ty& val) {
	for (; first != last; ++first)
		if (!(*first == val))
			*dest++ = *first;
	return dest;
}

template<class InIt, class OutIt, class Ty>
OutIt remove_copy(InIt *first, InIt *last, OutIt dest, const Ty& val) {
	for (; first != last; ++first)
		if (!(*first == val))
			*dest++ = *first;
	return dest;
}

int main () {
  int myints[] = {10,20,30,30,20,10,10,20};          // 10 20 30 30 20 10 10 20
  vector<int> myvector (8);
  vector<int>::iterator it;

  remove_copy (myints,myints+8,myvector.begin(),20); // 10 30 30 10 10 0 0 0
  assert(myvector[1] == 30);
  assert(myvector[7] == 0);
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;
 
  return 0;
}

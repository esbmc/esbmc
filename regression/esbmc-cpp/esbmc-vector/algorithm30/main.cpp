// remove_copy_if example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

bool IsOdd (int i) { return ((i%2)==1); }

template<class InIt, class OutIt, class Pr>
OutIt remove_copy_if(InIt first, InIt last, OutIt dest, Pr pred) {
	for (; first != last; ++first)
		if (!pred(*first))
			*dest++ = *first;
	return dest;
}

template<class InIt, class OutIt, class Pr>
OutIt remove_copy_if(InIt *first, InIt *last, OutIt dest, Pr pred) {
	for (; first != last; ++first)
		if (!pred(*first))
			*dest++ = *first;
	return dest;
}

int main () {
  int myints[] = {1,2,3,4,5,6,7,8,9};          
  vector<int> myvector (9);
  vector<int>::iterator it;

  remove_copy_if (myints,myints+9,myvector.begin(),IsOdd);
  assert(myvector[0] == 2);
  assert(myvector[6] == 0);
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;
 
  return 0;
}

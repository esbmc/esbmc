// rotate_copy algorithm example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

template<class InputIterator, class OutputIterator>
OutputIterator copy(InputIterator first, InputIterator last,
		OutputIterator result) {
	while (first!=last) *result++ = *first++;
	return result;
}

template<class InIt, class OutIt>
OutIt copy(InIt *first, InIt *last, OutIt dest) {
	while (first != last)
		*dest++ = *first++;
	return dest;
}

template<class FwdIt, class OutIt>
OutIt rotate_copy(FwdIt first, FwdIt mid, FwdIt last, OutIt dest) {
	dest = copy(mid, last, dest);
	return copy(first, mid, dest);
}

template<class FwdIt, class OutIt>
OutIt rotate_copy(FwdIt *first, FwdIt *mid, FwdIt *last, OutIt dest) {
	dest=copy (mid,last,dest);
	return copy (first,mid,dest);
}

int main () {
  int myints[] = {10,20,30,40,50,60,70};

  vector<int> myvector;
  vector<int>::iterator it;

  myvector.resize(7);

  rotate_copy(myints,myints+3,myints+7,myvector.begin());

  assert(myvector[0] == 40);

  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}

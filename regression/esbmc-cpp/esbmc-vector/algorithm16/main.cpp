// swap_ranges example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

template<class Ty>
void swap_esbmc(Ty& a, Ty& b) {
	Ty c(a);
	a = b;
	b = c;
}

template<class FwdIt1, class FwdIt2>
FwdIt2 swap_ranges_esbmc(FwdIt1 first1, FwdIt1 last1, FwdIt2 last2) {
	while (first1 != last1)
		swap_esbmc(*first1++, *last2++);
	return last2;
}

int main () {
  vector<int> first (5,10);        //  first: 10 10 10 10 10
  vector<int> second (5,33);       // second: 33 33 33 33 33
  vector<int>::iterator it;

  swap_ranges_esbmc(first.begin()+1, first.end()-1, second.begin());
  assert(first[2] == 33);
  assert(second[2] == 10);
  // print out results of swap:
  cout << " first contains:";
  for (it=first.begin(); it!=first.end(); ++it)
    cout << " " << *it;

  cout << "\nsecond contains:";
  for (it=second.begin(); it!=second.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}

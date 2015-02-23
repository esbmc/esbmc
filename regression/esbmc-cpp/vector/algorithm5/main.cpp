// find_first_of example
#include <iostream>
#include <cassert>
#include <cctype>
#include <vector>
using namespace std;

template<class FwdIt1, class FwdIt2>
FwdIt1 find_first_of(FwdIt1 first1, FwdIt1 last1, FwdIt2 first2, FwdIt2 last2) {
	for (; first1 != last1; ++first1)
		for (FwdIt2 it = first2; it != last2; ++it)
			if (*it == *first1)
				return first1;
	return last1;
}

template<class FwdIt1, class FwdIt2>
FwdIt1 find_first_of(FwdIt1 first1, FwdIt1 last1, FwdIt2 *first2,
		FwdIt2 *last2) {
	for (; first1 != last1; ++first1)
		for (FwdIt2 *it = first2; it != last2; ++it)
			if (*it == *first1)
				return first1;
	return last1;
}

template<class FwdIt1, class FwdIt2, class Pr>
FwdIt1 find_first_of(FwdIt1 first1, FwdIt1 last1, FwdIt2 first2, FwdIt2 last2,
		Pr pred) {
	for (; first1 != last1; ++first1)
		for (FwdIt2 it = first2; it != last2; ++it)
			if (pred(*it, *first1))
				return first1;
	return last1;
}

template<class FwdIt1, class FwdIt2, class Pr>
FwdIt1 find_first_of(FwdIt1 first1, FwdIt1 last1, FwdIt2 *first2, FwdIt2 *last2,
		Pr pred) {
	for (; first1 != last1; ++first1)
		for (FwdIt2 *it = first2; it != last2; ++it)
			if (pred(*it, *first1))
				return first1;
	return last1;
}

bool comp_case_insensitive (char c1, char c2) {
  return (tolower(c1)==tolower(c2));
}

int main () {
  int mychars[] = {97, 98, 99, 65, 66, 67};
  vector<int> myvector (mychars,mychars+6);
  vector<int>::iterator it;

  int match[] = {65, 66, 67};

  // using default comparison:
  it = find_first_of (myvector.begin(), myvector.end(), match, match+3);
  assert(*it == 65);
  if (it!=myvector.end())
    cout << "first match is: " << *it << endl;

  // using predicate comparison:
  it = find_first_of (myvector.begin(), myvector.end(),
                      match, match+3, comp_case_insensitive);
  assert(*it == 97);
  if (it!=myvector.end())
    cout << "first match is: " << *it << endl;
  
  return 0;
}

// lexicographical_compare example
#include <iostream>
#include <cassert>
#include <cctype>
using namespace std;

template<class InIt1, class InIt2>
bool lexicographical_compare(InIt1 *first1, InIt1 *last1, InIt2 *first2,
		InIt2 *last2) {
	while (first1 != last1) {
		if (first2 == last2 || *first2 < *first1)
			return false;
		else if (*first1 < *first2)
			return true;
		first1++;
		first2++;
	}
	return (first2 != last2);
}

// a case-insensitive comparison function:
bool mycomp (char c1, char c2)
{ return tolower(c1)<tolower(c2); }

int main () {
  char first[]="Apple";         // 5 letters
  char second[]="apartment";    // 9 letters
  assert(lexicographical_compare(first,first+5,second,second+9));
  cout << "Using default comparison (operator<): ";
  if (lexicographical_compare(first,first+5,second,second+9))
    cout << first << " is less than " << second << endl;
  else
    if (lexicographical_compare(second,second+9,first,first+5))
      cout << first << " is greater than " << second << endl;
  else
    cout << first << " and " << second << " are equivalent\n";

  return 0;
}

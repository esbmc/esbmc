#include <vector>
#include <cassert>
#include <iostream>
using namespace std;
int main()
{
    std::vector<char> letters;
	 letters.push_back('a');
	 letters.push_back('b');
	 letters.push_back('c');
	 letters.push_back('d');
	 letters.push_back('e');
	 letters.push_back('f');
	 letters.push_back('g');
 
    if (!letters.empty()) {
		  assert(letters.front() == 'a');
        cout << "The first character is: " << letters.front() << endl;
    }
	 assert(letters.back() == 'g');
	 letters.pop_back();
	 assert(letters.back() == 'f');
	 return 0;
}

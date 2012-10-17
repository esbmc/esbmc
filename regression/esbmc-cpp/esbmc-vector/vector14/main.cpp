#include <vector>
#include <cassert>
#include <iostream>
using namespace std;

int main( )
{
	 int i;
    std::vector<char> c;
	 c.push_back('f');
	 c.push_back('e');
	 c.push_back('l');
	 c.push_back('i');
	 c.push_back('p');
	 c.push_back('e');
    std::vector<char> d;
	 d.push_back('a');
	 d.push_back('x');
	 d.push_back('b');
	 d.push_back('b');
	 d.push_back('n');
	 d.push_back('m');
	 d.push_back('h');
	 d.push_back('h');
	 d.push_back('f');
	 assert(c.front() == 'f');
	 c.swap(d);
	 assert(c.front() != 'a');
	 return 0;
}

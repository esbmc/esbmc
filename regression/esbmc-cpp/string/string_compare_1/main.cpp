#include <string>
#include <cassert>
using namespace std;


int main(){
	string str1;
	int i, j, k;
	str1 = "Testing";
	i = str1.compare("Testing");
	j = str1.compare("Test");
	k = str1.compare("Testing this");
	assert( ( i == 0 ) && ( j > 0 ) && ( k < 0 ) );
	

}

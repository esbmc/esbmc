#include <string>
#include <cassert>
using namespace std;

int main(){
	string str1, str2;
	int i, j, k;
	str1 = string("Test");
	str2 = str1;

	i = str2.compare(str1);
	str1 = 'T';
	j = str2.compare(str1);
	str1 = "Testing";
	k = str2.compare(str1);
	assert( ( i == 0 ) && ( j > 0 ) && ( k < 0 ) );

}

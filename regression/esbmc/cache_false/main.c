#include <assert.h>
int main() {
	int arr[50];
	for(int i = 0; i < 50; i++) {
		arr[i] = 50 - i;
		assert(arr[i] == 50 - i); }
	assert(arr[23] == 72);
}

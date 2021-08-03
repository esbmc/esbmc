/* This test is just to check if the no-unroll
 * option is working properly
 * If the loop is unrolled this should give find the error
 *    with a small K
 *
 * If it is isn't, it will not be able to find the error.
*/

int main() {
    int arr[1001];

    for(int i = 0; i < 500; i++) arr[i] = i;
    for(int i = 0; i < 500; i++) __ESBMC_assert(arr[i] != i, "This is a violation");

    return 0;
}
// Check if the flag don't disable other pointer checks
int main(void) {
    int* p = (int*) malloc(sizeof(int) *10); // --assume-malloc-success
    int q;
    return p[q];
}

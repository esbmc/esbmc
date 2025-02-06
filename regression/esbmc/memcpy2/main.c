#include <stdio.h>
#include <string.h>

void test_memcpy_null() {
    char *src = "Hello";
    char dest[5];

     memcpy(dest, src, 5);  

}

int main() {
    test_memcpy_null();
    return 0;
}


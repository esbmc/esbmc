#include <stdio.h>
#include <string.h>

void test_memcpy_overlap() {
    char buffer[] = "Hello, World!";
    
    // Overlapping copy: Copy "World" into "Hello"
    memcpy(buffer + 2, buffer, 5);  // Unsafe because of overlap

    printf("Test 2 (Overlapping memcpy): %s\n", buffer);  
}

int main() {
    test_memcpy_overlap();
    return 0;
}


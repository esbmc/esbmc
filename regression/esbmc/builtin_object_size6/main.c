#include <assert.h>
#include <string.h>

int main() {
    char buffer[10];
    size_t buf_size = __builtin_object_size(buffer, 0);
    
    assert(buf_size == 10);
    
    if (buf_size >= 5) {
        memset(buffer, 'A', 5);
    }
    
    return 0;
}

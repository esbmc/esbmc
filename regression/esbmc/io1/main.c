#include <stdio.h>
#include <assert.h>

void test_fclose_null_pointer() {
    int close_status = fclose(NULL);
    assert(close_status != 0);
}

int main() {
    test_fclose_null_pointer();
    return 0;
}


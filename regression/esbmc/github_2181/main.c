#include <assert.h>

struct MyStruct {
    int values[5]; // Array as a struct member
};

int main() {
    struct MyStruct data;
    assert(data.values[3] !=2);
}


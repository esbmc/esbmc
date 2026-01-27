// Test: ensures clause with || and __ESBMC_old on struct array via pointer.
// Validates that inline_temporary_variables correctly reconstructs
// short-circuit OR expressions from Clang's control flow.
#include <assert.h>

typedef struct {
    int id;
    int value;
} Item;

void modify_first(Item* items, int which)
{
    __ESBMC_requires(items != 0);
    __ESBMC_requires(which == 0 || which == 1);

    __ESBMC_assigns(items[0].value, items[1].value);

    // which != 0 || (items[0] unchanged AND items[1] unchanged)
    // Without correct short-circuit reconstruction, the || right side is lost.
    __ESBMC_ensures(which != 0 || (
        items[0].value == __ESBMC_old(items[0].value) &&
        items[1].value == __ESBMC_old(items[1].value)
    ));

    // which != 1 || items[0].value == 42
    __ESBMC_ensures(which != 1 || items[0].value == 42);

    if (which == 1) {
        items[0].value = 42;
    }
}

int main()
{
    Item items[2];
    items[0].value = 10;
    items[1].value = 20;

    modify_first(items, 0); // which==0 so nothing changes

    assert(items[0].value == 10);
    assert(items[1].value == 20);

    return 0;
}

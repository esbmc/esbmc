#include <cheri/cheric.h>
#include <assert.h>

char *buffer = "hello";

int main(int argc, char **argv) {
    /* create valid capability: */
    char *__capability valid_cap = cheri_ptr(buffer, 6);
    /* create a copy with tag cleared: */
    char *__capability cleared_cap = cheri_cleartag(valid_cap);

    /* this does not fail, leaking the capability base */
    assert(cheri_getbase(cleared_cap));

    return 0;
}

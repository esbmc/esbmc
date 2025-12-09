#include <cheri/cheric.h>
#include <assert.h>

char *buffer = "hello";
char *secret = "secret";

int main(int argc, char **argv) {
    /* create valid capability: */
    char *__capability valid_cap = cheri_ptr(buffer, 6);
    /* create a copy with tag cleared: */
    char *__capability cleared_cap = cheri_cleartag(valid_cap);

    /* now try to dereference the capability with cleared tag */
    assert(cleared_cap[0]);  // this fails with tag violation exception

    return 0;
}

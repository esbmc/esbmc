#include <assert.h>
#include <stdio.h>

int main(void) {
    char buffer[32];
    FILE *fp = fmemopen(buffer, sizeof(buffer), "r"); // read-only
    if (fp == NULL)
        return -1;

    // No error at start
    assert(ferror_unlocked(fp) == 0);

    // Try writing to a read-only stream
    int ret = fputs("hello", fp);
    assert(ret >= 0); // This assertion will fail

    // Flush should never be reached if assert above fails
    fflush(fp);
    assert(ferror_unlocked(fp) == 0);

    fclose(fp);
    return 0;
}


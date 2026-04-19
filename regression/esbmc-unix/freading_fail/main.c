#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>

int main(void) {
    char buffer[32] = "abc";

    // Open read-only
    FILE *fp = fmemopen(buffer, sizeof(buffer), "r");
    if (fp == NULL)
        return -1;

    // This assertion will fail because the stream is in read mode
    assert(__freading(fp) == 0);

    char tmp[16];
    char *res = fgets(tmp, sizeof(tmp), fp);
    assert(res != NULL);

    // This one will pass (stream is in read mode)
    assert(__freading(fp) != 0);

    fclose(fp);
    return 0;
}


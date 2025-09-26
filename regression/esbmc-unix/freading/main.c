#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>

int main(void) {
    char buffer[32] = "hello world\n";

    FILE *fp = fmemopen(buffer, sizeof(buffer), "r");
    if (fp == NULL)
      return -1;

    // Initially in read mode
    assert(__freading(fp) != 0);

    // Perform a read
    char tmp[16];
    char *res = fgets(tmp, sizeof(tmp), fp);
    assert(res != NULL);

    // Still in read mode
    assert(__freading(fp) != 0);

    fclose(fp);
    return 0;
}


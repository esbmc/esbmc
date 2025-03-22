#include <stdio.h>
#include <assert.h>
#include <errno.h>

void test_fopen_fclose() {
    FILE *file = fopen("testfile.txt", "w");
    assert(file != NULL);

    int close_status = fclose(file);
    assert(close_status == 0 || close_status == EOF);
}

void test_fdopen() {
    FILE *file = fopen("testfile.txt", "w+");
    assert(file != NULL);
    if (file != NULL)
    {
      int fd = fileno(file);
      assert(fd >= 0 || fd == -1);

      FILE *dup_file = fdopen(fd, "r");
      assert(dup_file != NULL);
      fclose(file);
      fclose(dup_file);
   }
}

void test_feof_ferror() {
    FILE *file = fopen("testfile.txt", "w+");
    assert(file != NULL);

    if (file != NULL)
    {
      int c = fgetc(file);
      assert(c == EOF || (c >= 0 && c <= 255));
      feof(file);
    }

    fclose(file);
}

void test_fopen_invalid_file() {
    FILE *file = fopen("nonexistentfile.txt", "r");
    assert(file != NULL);
}

void test_fclose_null_pointer() {
    int close_status = fclose(NULL);
    assert(close_status != 0);
}

int main() {
    test_fopen_fclose();
    test_fdopen();
    test_feof_ferror();
    test_fopen_invalid_file();
    test_fclose_null_pointer();
    return 0;
}


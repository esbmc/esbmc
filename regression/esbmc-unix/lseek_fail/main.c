#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

int main(void) {
    int fd = open("testfile.txt", O_CREAT | O_RDWR | O_TRUNC, 0600);
    if (fd < 0)
        return -1;

    // Attempt an invalid seek: negative offset from start
    off_t pos = lseek(fd, -10, SEEK_SET);

    // This will fail because lseek returns -1
    assert(pos >= 0);

    close(fd);
    return 0;
}


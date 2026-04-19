#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

int main(void) {
    const char *msg = "hello";
    char buffer[16];

    int fd = open("testfile.txt", O_CREAT | O_RDWR | O_TRUNC, 0600);
    assert(fd >= 0);

    // Write something
    ssize_t written = write(fd, msg, strlen(msg));
    assert(written == (ssize_t)strlen(msg));

    // Seek back to beginning
    off_t pos = lseek(fd, 0, SEEK_SET);
    assert(pos == 0);

    // Read back
    ssize_t n = read(fd, buffer, sizeof(buffer));
    assert(n > 0);

    close(fd);
    return 0;
}


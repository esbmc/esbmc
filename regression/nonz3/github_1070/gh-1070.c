//FormAI DATASET v0.1 Category: Building a HTTP Client ; Style: high level of detail
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h> /* sockaddr_in */
#include <unistd.h>
#define BUFSIZE 4096

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <url>\n", argv[0]);
        return 1;
    }

    const char delimiter[2] = "/";
    char* host, *path;
    char url[strlen(argv[1])];
    strcpy(url, argv[1]);

    strtok(url, delimiter); // http:
    strtok(NULL, delimiter); // empty
    strtok(NULL, delimiter); // host

    host = strtok(NULL, delimiter); // path
    path = "/";
    while (host != NULL) {
        path = strcat(path, "/");
        path = strcat(path, host);
        host = strtok(NULL, delimiter);
    }

    char request[BUFSIZE];
    sprintf(request, "GET %s HTTP/1.1\r\n"
                      "Host: %s\r\n"
                      "Connection: close\r\n\r\n"
            , path, argv[1]);

    struct sockaddr_in server;
    memset(&server, 0, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = inet_addr(argv[1]);
    server.sin_port = htons(80);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Socket error.");
        return 1;
    }

    if (connect(sock, (struct sockaddr*) &server, sizeof(server)) < 0) {
        perror("Connection error.");
        return 1;
    }

    send(sock, request, strlen(request), 0);

    char buffer[BUFSIZE];
    while (recv(sock, buffer, BUFSIZE, 0) > 0) {
        printf("%s", buffer);
        memset(buffer, 0, BUFSIZE);
    }

    close(sock);
    return 0;
}

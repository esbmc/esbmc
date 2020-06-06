#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>

char Message[100];
char server_response[100];
char server_response2[100];
int key;

int main() {
    //create socket
    int Client_Socket, connect_status;
    Client_Socket = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(9002);
    server_address.sin_addr.s_addr = INADDR_ANY;
    connect_status = connect(Client_Socket, (struct sockaddr*) &server_address, sizeof(server_address));
    //connect to the socket and pass the server address,
    if (connect_status == -1) {
        printf("There is an error with the connection, Check your connection please\n");
        exit(0);
    }
    //send message to the server
    // send(Client_Socket,Message);
    //receive data from server
    recv(Client_Socket, &server_response, sizeof(server_response), 0);
    //print out the server's response
    printf("%s\n", server_response);
    printf("Connection Now is  established\n Please enter your message\n");
    //scanf("%s", Message);
    gets(Message);
    //printf("\nEnter the key\n");
   // scanf("%d",&key);
    key = rand() % 26 +1;
    //key = key % 26;
    printf("The key is %d",key);
    for(int i = 0; (i < 100 && Message[i] != '\0'); i++)
        Message[i] = Message[i] + key; //the key for encryption is 3 that is added to ASCII value
    printf("\nyour Encrypted Message is: %s\n", Message);
    send(Client_Socket, &key, sizeof(key),0);
    send(Client_Socket, &Message, sizeof(Message),0 );
    recv(Client_Socket, server_response2, sizeof(server_response2), 0);
    //print out the server's response
    printf("\nHi from the server, Your Decrypted Message is : %s\n", server_response2);
    // close the socket
    // close(socket);
    return 0;

}




// Created by Fatimah Aljaafari on 30/05/2020.
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
char client_response[100];
int key;
int main() {
    //create socket
    int Server_Socket, connect_status;
    Server_Socket = socket(AF_INET, SOCK_STREAM, 0);
    //define the server address
    //AF_INET=IPv4 protocol, SOCK_DGRAM = communicatoin type (TCP), Protocol value for (IP)= 0
    //Now we will specify the address and port # to connect to it.
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(9002);
    server_address.sin_addr.s_addr = INADDR_ANY;
    // bind the socket to our Ip and pord
    bind(Server_Socket, (struct sockaddr*) &server_address, sizeof(server_address));
    // listent for connection, 5 how many connections can wait for at the same time)
    listen(Server_Socket, 5);
    int Client_Socket;
    //accept connection, NULL --> address of client but we don't need it now because we will connect to local machine)
    Client_Socket = accept(Server_Socket, NULL, NULL);
    //send message
    printf("Waiting for the clients");
    char server_message[100]="You have reached the server, please tell us what you want to decrypt";
    send(Client_Socket, server_message, sizeof(server_message),0);
    //close socket
    //close(Server_Socket);
    recv(Client_Socket,&key,sizeof(key),0);
    recv(Client_Socket,client_response,sizeof(client_response),0 );

    printf("\nThe message you sent is : %s\n",client_response);

    for(int i = 0; (i < 100 && client_response[i] != '\0'); i++)// the decrypted message
        client_response[i] = client_response[i] - key;
    printf("\n Your decrypted Message is : %s \n", client_response);
    send(Client_Socket, client_response, sizeof(client_response),0);

    return 0;
};

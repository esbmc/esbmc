//
//  Created by Fatimah Aljaafari on 14/06/2020.
//  Copyright © 2020 Fatimah Aljaafari. All rights reserved.
//

#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define size 100
char Message[size];
int key, key1;
pthread_mutex_t mutex;
void *client(
  void *
    arg) //void pointer to pass any data type and returen any data type from thread function
{
  pthread_mutex_lock(&mutex);
  for(int i = 0; (i < size && Message[i] != '\0'); i++)
    Message[i] = Message[i] + key1;
  printf("\nyour Encrypted Message is: %s\n", Message);
  pthread_mutex_unlock(&mutex);
  pthread_exit(0);
}

void *server(void *arg)
{
  pthread_mutex_lock(&mutex);
  for(int i = 0; (i < size && Message[i] != '\0'); i++) // the decrypted message
    Message[i] = Message[i] - key1;
  printf("\n Your decrypted Message is : %s \n", Message);
  pthread_mutex_unlock(&mutex);
  pthread_exit(0);
}

int main()
{
  printf("\n Please enter your message\n");
  fgets(Message, sizeof(Message), stdin);
  printf("\nEnter the key\n");
  scanf("%d", &key);
  key1 = (sqrt(key));
  printf("The key is %d", key1);
  pthread_t encryption, decryption;
  pthread_create(&decryption, 0, server, 0);
  pthread_create(&encryption, 0, client, 0);
  pthread_join(encryption, 0);
  pthread_join(decryption, 0);
  assert(pthread_equal(encryption, decryption));
  printf("\n Thanks for using our program \n");
}


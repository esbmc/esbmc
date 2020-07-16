
//  Created by Fatimah Aljaafari on 14/06/2020.
//  Copyright © 2020 Fatimah Aljaafari. All rights reserved.
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#define N 10
char nondet_char();
unsigned short nondet_ushort();
unsigned int nondet_uint();
int i;
unsigned short key;
char str[N];
char copyf[12];
pthread_mutex_t mutex;
void initialise_array(char *array, unsigned short size)
{
  unsigned short i;
  for(i = 0; i < nondet_ushort() % size; ++i)
  {
    array[i] = nondet_char();
  }
  strcpy(copyf, array);
  printf("str is: %s and copy is %c \n", array, copyf[11]);
}
void *client(void *arg)
{
  pthread_mutex_lock(&mutex);
  for(i = 0; (i < N && str[i] != '\0'); i++)
    str[i] = str[i] + key;
  assert(
    str[i] ==
    copyf
      [i]); //assert if encrypted str + key == original str, if yes hit the position (abort).
  pthread_mutex_unlock(&mutex);
  pthread_exit(0);
}
void *server(void *arg)
{
  pthread_mutex_lock(&mutex);
  for(i = 0; (i < N && str[i] != '\0'); i++) // the decrypted message
    str[i] = str[i] - key;
  pthread_mutex_unlock(&mutex);
  pthread_exit(0);
}
int main()
{
  key = nondet_ushort() % 26 + 1;
  initialise_array(str, N);
  pthread_t encryption, decryption;
  pthread_create(&decryption, 0, server, 0);
  pthread_create(&encryption, 0, client, 0);
  pthread_detach(encryption);
  pthread_join(encryption, 0);
  pthread_join(decryption, 0);
}

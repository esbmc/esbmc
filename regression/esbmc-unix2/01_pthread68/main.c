//  Created by Fatimah Aljaafari on 14/06/2020.
//  Copyright © 2020 Fatimah Aljaafari. All rights reserved.
#include <pthread.h>
#include <assert.h>
#include <errno.h>
unsigned short key;
char str[1];
pthread_mutex_t mutex;
void *client(void *arg)
{
  pthread_mutex_lock(&mutex);
  str[0] = str[0] + key;
  pthread_mutex_unlock(&mutex);
  return NULL;
}
void *server(void *arg)
{
  pthread_mutex_lock(&mutex);
  str[0] = str[0] - key;
  pthread_mutex_unlock(&mutex);
  return NULL;
}
int main()
{
  key = nondet_ushort() % 26 + 1;
  pthread_t encryption, decryption;
  pthread_create(&decryption, 0, server, 0);
  pthread_create(&encryption, 0, client, 0);
  assert(pthread_detach(encryption) == 0);
  assert(pthread_detach(encryption) == EINVAL);
}

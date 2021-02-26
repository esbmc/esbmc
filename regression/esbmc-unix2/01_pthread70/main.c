//  Created by Fatimah Aljaafari on 14/06/2020.
//  Copyright © 2020 Fatimah Aljaafari. All rights reserved.
#include <pthread.h>
#include <assert.h>
#include <errno.h>
int i;
pthread_mutex_t mutex;
void *client(void *arg)
{
  pthread_mutex_lock(&mutex);
  i++;
  pthread_mutex_unlock(&mutex);
  return NULL;
}

int main()
{
  pthread_t encryption;
  pthread_create(&encryption, 0, client, 0);
  if(!pthread_join(encryption, NULL))
    assert(pthread_detach(encryption) == ESRCH);
  return 0;
}

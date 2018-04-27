#include <pthread.h>
#include <assert.h>

#define BUFFER_MAX  10
#define N 1
#define ERROR -1
#define FALSE 0
#define TRUE 1

static char  buffer[BUFFER_MAX];     /* BUFFER */

static unsigned int first;           /* Pointer to the input buffer   */
static unsigned int next;            /* Pointer to the output pointer */
static int  buffer_size;	         /* Max amount of elements in the buffer */

_Bool send, receive;

pthread_mutex_t m;

void initLog(int max) {
	buffer_size = max;
	first = next = 0;
}

int removeLogElement(void) {

	assert(first>=0);

	if (next > 0 && first < buffer_size) {
		first++;
		return buffer[first-1];
	}
	else {
		return ERROR;
	}
}  

int insertLogElement(int b) {

	if (next < buffer_size && buffer_size > 0) {
		buffer[next] = b;
		next = (next+1)%buffer_size;
		assert(next<buffer_size);
	} else {
		return ERROR;
	}

	return b;
}

void *t1(void *arg) {

  pthread_mutex_lock(&m);
//  if (send) {
    insertLogElement(0);
//    send=FALSE;
//	receive=TRUE;
//  }
  pthread_mutex_unlock(&m);
}

void *t2(void *arg) {

  pthread_mutex_lock(&m);
//  if (receive) {
//    assert(removeLogElement()==0);
	receive=FALSE;
	send=TRUE;
//  }
  pthread_mutex_unlock(&m);
}

int main() {

	pthread_t id1, id2;

	pthread_mutex_init(&m, 0);

	initLog(5);
	send=TRUE;
	receive=FALSE;

	pthread_create(&id1, NULL, t1, NULL);
	pthread_create(&id2, NULL, t2, NULL);

	return 0;
}


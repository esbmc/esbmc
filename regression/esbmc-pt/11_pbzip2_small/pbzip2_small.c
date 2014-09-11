#include "header.h"
#include "LinuxThreads.h"
#include <pthread.h> 
// pthread_mutex_t fifo_mut;
int fifo_mut_m_spinlock;
int fifo_mut_m_count;
// pthread_mutex_t ProgressIndicatorsMutex;
int ProgressIndicatorsMutex_m_spinlock;
int ProgressIndicatorsMutex_m_count;
// pthread_mutex_t TerminateFlagMutex = PTHREAD_MUTEX_INITIALIZER;
int TerminateFlagMutex_m_spinlock;
int TerminateFlagMutex_m_count;
// pthread_mutex_t ProducerDoneMutex;
int ProducerDoneMutex_m_spinlock;
int ProducerDoneMutex_m_count;
int producerDone = 0;
int terminateFlag = 0;

#define queue void
queue *fifo;
int fifo_full;
int fifo_empty;
int NumBlocks, InBytesProduced, inSize;
int __X__;

#define queue_add(fifo) \
	__X__ = 0; \
	assert(__X__<=0); \
	fifo_empty = 0

#define queue_remove(fifo) \
	__X__ = 1; \
	assert(__X__>=1); \
	fifo_full = 0; 

inline int syncGetProducerDone()
{
	int ret;
	pthread_mutex_lock(ProducerDoneMutex_m_spinlock, ProducerDoneMutex_m_count);
	ret = producerDone;
	pthread_mutex_unlock(ProducerDoneMutex_m_spinlock, ProducerDoneMutex_m_count);

	return ret;
}

inline int syncGetTerminateFlag()
{
	int ret;
	pthread_mutex_lock(TerminateFlagMutex_m_spinlock, TerminateFlagMutex_m_count);
	ret = terminateFlag;
	pthread_mutex_unlock(TerminateFlagMutex_m_spinlock, TerminateFlagMutex_m_count);

	return ret;
}

int producer()
{
	pthread_mutex_lock(ProgressIndicatorsMutex_m_spinlock, ProgressIndicatorsMutex_m_count);
	NumBlocks = 0;
	InBytesProduced = 0;
	pthread_mutex_unlock(ProgressIndicatorsMutex_m_spinlock, ProgressIndicatorsMutex_m_count);

	while (1)
	{
		if (syncGetTerminateFlag() != 0)
		{
			return -1;
		}
		pthread_mutex_lock(fifo_mut_m_spinlock, fifo_mut_m_count);
		while (fifo_full)
		{
			pthread_cond_wait(fifo_mut_m_spinlock, fifo_mut_m_count);

			if (syncGetTerminateFlag() != 0)
			{
				pthread_mutex_unlock(fifo_mut_m_spinlock, fifo_mut_m_count);
				return -1;
			}
		}
		queue_add(fifo);
		pthread_cond_signal();

		pthread_mutex_lock(ProgressIndicatorsMutex_m_spinlock, ProgressIndicatorsMutex_m_count);
		++NumBlocks;
		InBytesProduced += inSize;
		pthread_mutex_unlock(ProgressIndicatorsMutex_m_spinlock, ProgressIndicatorsMutex_m_count);
		
		pthread_mutex_unlock(fifo_mut_m_spinlock, fifo_mut_m_count);
	} // while
}

void *consumer()
{
	for (;;)
	{
		if (syncGetTerminateFlag() != 0)
		{
			return (NULL);
		}

		pthread_mutex_lock(fifo_mut_m_spinlock, fifo_mut_m_count);
		for (;;)
		{
		  if (!fifo_empty)
		    {
				queue_remove(fifo);
				// block retreived - break the loop and continue further
				break;
		    }
		  
		  if (fifo_empty && ((syncGetProducerDone() == 1) || (syncGetTerminateFlag() != 0)))
		    {
		      pthread_mutex_unlock(fifo_mut_m_spinlock, fifo_mut_m_count);
		      return (NULL);
		    }
		  
		  pthread_cond_wait(fifo_mut_m_spinlock, fifo_mut_m_count);
		}

		pthread_cond_signal();
		pthread_mutex_unlock(fifo_mut_m_spinlock, fifo_mut_m_count);

		// outputBufferAdd(&outBlock, "consumer");
	} // for
	
	return (NULL);
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, consumer, NULL);
  pthread_create(&t2, NULL, producer, NULL);
  return 0;
}

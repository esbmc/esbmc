// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2005-2021 University of Tartu & Technische Universität München
//
// SPDX-License-Identifier: MIT

#include <assert.h>
extern void abort(void);
void reach_error()
{
  assert(0);
} //6
void __VERIFIER_assert(int cond)
{
  if(!(cond))
  {
  ERROR:
  {
    reach_error();
    abort();
  }
  }
}

#include <pthread.h>

struct __anonstruct_PQUEUE_63
{
  int occupied;
  pthread_mutex_t mtx;
};
typedef struct __anonstruct_PQUEUE_63 PQUEUE;

PQUEUE pqb; //1

int pqueue_init(PQUEUE *qp)
{
  qp->occupied = 0; //4
  pthread_mutex_init(&qp->mtx, NULL);
  return (0);
}

int pqueue_put(PQUEUE *qp)
{
  pthread_mutex_lock(&qp->mtx);
  if(qp->occupied < 1000)
    (qp->occupied)++;
  pthread_mutex_unlock(&qp->mtx);
  return (1);
}

int pqueue_get(PQUEUE *qp)
{
  int got = 0; //4
  pthread_mutex_lock(&qp->mtx);
  while(qp->occupied <= 0)
  {
    __VERIFIER_assert(qp->occupied == 0);
  }
  __VERIFIER_assert(qp->occupied != 0);
  if(qp->occupied > 0)
  {
    (qp->occupied)--;
    got = 1;
    pthread_mutex_unlock(&qp->mtx);
  }
  else
  {
    pthread_mutex_unlock(&qp->mtx);
  }
  return (got);
}

void *worker(void *arg)
{
  while(1)
  {
    pqueue_get(&pqb);
  }
  return NULL;
}

int main(int argc, char **argv)
{
  pthread_t tid;

  PQUEUE *qp = &pqb; //2
  pqueue_init(&pqb);
  pthread_create(&tid, NULL, &worker, NULL);

  for(int i = 1; i < 3; i++)
  { //5
    pqueue_put(&pqb);
  }
  return 0;
}

/* Simplified version of SV-COMP 2023's
 * c/goblint-regression/09-regions_11-arraylist_nr.c
 */

// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2005-2021 University of Tartu & Technische Universität München
//
// SPDX-License-Identifier: MIT

extern int __VERIFIER_nondet_int();
extern void abort(void);
void assume_abort_if_not(int cond) {
  if(!cond) {abort();}
}

#include<pthread.h>
#include<stdlib.h>
#include<stdio.h>

struct s {
  int datum;
  struct s *next;
};

struct s *new(int x) {
  struct s *p = malloc(sizeof(struct s));
  p->datum = x;
  p->next = NULL;
  return p;
}

void list_add(struct s *node, struct s *list) {
  struct s *temp = list->next;
  list->next = node;
  node->next = temp;
}

#define N 2

pthread_mutex_t mutex[N];
struct s *slot[N];

void *t_fun(void *arg) {
  int i = __VERIFIER_nondet_int();
  assume_abort_if_not(0 <= i && i < N);
  pthread_mutex_lock(&mutex[i]);
  list_add(new(3), slot[i]);
  pthread_mutex_unlock(&mutex[i]);
  return NULL;
}

int main () {
  for (int i = 0; i < N; i++)
    pthread_mutex_init(&mutex[i], NULL);

  int j = __VERIFIER_nondet_int();
  assume_abort_if_not(0 <= j && j < N);
  struct s *p;
  pthread_t t1;

  for (int k = 0; k < N; k++) {
    slot[k] = new(1);
    list_add(new(2), slot[k]);
  }

  //pthread_create(&t1, NULL, t_fun, NULL);

  //pthread_mutex_lock(&mutex[j]);
  // p = slot[j]->next; // NORACE
  // printf("%d\n", p->datum);
  //pthread_mutex_unlock(&mutex[j]);

  // t_fun(NULL);

  //pthread_join(t1, NULL);

  return 0;
}

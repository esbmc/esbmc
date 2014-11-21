// #include <assert.h>
#include <pthread.h>

int g;
int g1 = 0;
int g2 = 0;

void* thr1() {
  while (g1 < g) {
    g1 = g1 + 1;
  }
}

void* thr2() {
  while (g2 < g) {
    g2 = g2 + 1;
  }
}

int main() {
  pthread_t t1, t2;
  glb_init(g>0);
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  assert(g1 <= g);
  assert(g2 <= g);
  assert(g1+g2 <= 2*g);
}

/* Minimal set of predicates (2-0/3-0/0-2): 
retractall(preds(_,_,_)), retractall(trans_preds(_,_,_,_)),
assert(preds(1, p(_,data(G,G1,_)), [G-G1>=0,G-G1>=1])),
assert(preds(2, p(_,data(G,G1,_)), [])),

assert(trans_preds(_-1, p(_,data(G,G1,_)), p(_,data(GP,G1P,_)), [G-G1-GP+G1P=<0,G-G1-GP+G1P=<1,GP-G1P>=1])),

assert(pc_preds(2, [[[inf|3],[5|sup]],[[inf|0],[2|sup]]], from)),
assert(pc_preds(2, [[[4|4]],[[1|1]]], to)).
*/

/* Set of predicates inferred automatically in 8 iterations (3-2/4-0/0-2):
retractall(preds(_,_,_)), retractall(trans_preds(_,_,_,_)),
assert(preds(1, p(_,data(G,G1,_)), [G-G1>=0,G-G1>=1,G>=1])),
assert(preds(2, p(_,data(G,G1,_)), [G1=<0,G-G1>=1])),

assert(trans_preds(_-1, p(_,data(G,G1,_)), p(_,data(GP,G1P,_)), [G-GP+G1P=<0, G-G1-GP+G1P=<1, G-G1-GP+G1P=<0, GP-G1P>=1])),

assert(pc_preds(2, [[[inf|3],[5|sup]],[[inf|0],[2|sup]]], from)),
assert(pc_preds(2, [[[4|4]],[[1|1]]], to)).
*/

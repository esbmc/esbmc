struct a {
 union {
   struct a *c;
 } d;
};


struct a Q[3];
struct a *b = &Q;
int main() {
  (b--)->d.c;
}

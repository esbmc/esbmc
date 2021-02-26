#ifndef SVC_H
#define SVC_H

void abort(void); 
void reach_error(){ assert(0); }

#undef assert
#define assert( X ) (!(X) ? reach_error() : (void)0)

#endif // SVC_H

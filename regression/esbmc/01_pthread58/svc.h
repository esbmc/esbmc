#ifndef SVC_H
#define SVC_H

void abort(void); 
void reach_error(){}

#undef assert
#define assert( X ) (!(X) ? reach_error() : (void)0)

#endif // SVC_H

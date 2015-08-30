#define bool int
#define uint16_t int
#define result_t int
#define uint8_t int
#define SUCCESS 0
#define FAIL 0

#define TRUE 1
#define FLASE 0


#define IDLE 0
#define SINGLE_CONVERSION 1
#define CONTINUOUS_CONVERSION 2



#define TXSTATE_IDLE 0
#define TXSTATE_PROTO 1
#define TXSTATE_INFO 2
#define TXSTATE_ESC 3
#define TXSTATE_FCS1 4
#define TXSTATE_FCS2 5
#define TXSTATE_ENDFLAG 6
#define TXSTATE_FINISH 7
#define TXSTATE_ERROR 8 

#include <pthread.h> 
int x=0;
uint8_t state;
uint8_t tosPort; // check dataraces on this variable

void __ESBMC_atomic_begin();
void __ESBMC_atomic_end();

inline result_t dataReady(uint16_t data) {
  uint8_t port;
  uint8_t oldState;
  result_t ok;
  { __ESBMC_atomic_begin();
      // port = tosPort;
    oldState = state;
    if (state == SINGLE_CONVERSION) {
      state = IDLE;
    }
   __ESBMC_atomic_end();
  }
    
  switch (oldState) {
  case IDLE:
    return SUCCESS;
    
  case SINGLE_CONVERSION:
    //call HPLADC.sampleStop();
    break;
    
  case CONTINUOUS_CONVERSION:
    //call HPLADC.sampleAgain();
    break;
  }
  
  //dbg(DBG_ADC, "adc_tick: port %d with value %i \n", port, (int)data);
  //ok = signal ADC.dataReady[port](data);
  if (ok == FAIL && oldState == CONTINUOUS_CONVERSION) {
    //call HPLADC.sampleStop();
    //  atomic {
      state = IDLE;
      //}
  }
  return SUCCESS;
}

inline result_t startGet(uint8_t newState, uint8_t port) {
    uint8_t oldState;
    {
    __ESBMC_atomic_begin();
	oldState = state;
      if (state == IDLE) {
	state = newState;
      }
     __ESBMC_atomic_end();
    }
    if (oldState == IDLE) {
      if ( x<=0 ) x++;
      //??atomic {
      // tosPort = port;
      //}
      return SUCCESS; //call HPLADC.samplePort(port);
    }
    return FAIL;
}

// Run time envoirment ????
/*
void runtime(){
  int port, data;
  startGet( SINGLE_CONVERSION, port);
  dataReady( data);
  assert(x < 2);
}
*/
void main(){
  int port, data;
  startGet( SINGLE_CONVERSION, port);
  dataReady( data);
  assert(x < 2);
}

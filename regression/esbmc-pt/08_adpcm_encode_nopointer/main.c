static int indexTable[16] = 
{
 -1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8,
};

static int stepsizeTable[89] = 
{
  7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
  19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
  50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
  130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
  337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
  876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
  2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
  5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
  15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
};

/* SPARK - made all arrays being passed by reference into global arrays */
#define N 40   /* obtained from adpcm_encode_driver.c */
/* so also constraint on len being 0 to 39 */
/* its not completely clear - at one point they use 64 to initialize indata */
int indata[N];
int outdata[N];
int state_valprev;
int state_index;
//#define len (10)
/* End SPARK */

void coder(int len)
{
 __ESBMC_assume(len>=0 && len<=39);
 /* SPARK - no more pointers */
 // short *inp;			// Input buffer pointer 
 /// signed char *outp;		// output buffer pointer 
 /* End SPARK */
 int val;			// Current input sample value 
 int sign;			// Current adpcm sign bit 
 int delta;			// Current adpcm output value 
 int diff;			// Difference between val and valprev 
 int step;			// Stepsize 
 int valpred;		// Predicted output value 
 int vpdiff;			// Current change to valpred 
 int index;			// Current step change index 
 int outputbuffer;		// place to keep previous 4-bit value 
 int bufferstep;		// toggle between outputbuffer/output 
 int i;   // SPARK
 
 //SPARK outp = outdata;
 //SPARK inp = indata;
 
 step = stepsizeTable[state_index];
 
 bufferstep = 1;
 
 for (i=0; i<len; i++)  // SPARK: changed from for ( ; len > 0 ; len-- )
  {			// for well defined boudaries
   val = indata[i];
  
   // Step 1 - compute difference with previous value 
   diff = val - state_valprev;
  
   //SPARK sign = (diff < 0) ? 8 : 0;
   if (diff < 0)
     sign = 8;
   else
     sign = 0;
   
   if ( sign ) diff = (-diff);
  
  // Step 2 - Divide and clamp 
  // Note:
  // This code *approximately* computes:
  //    delta = diff*4/step;
  //    vpdiff = (delta+0.5)*step/4;
  // but in shift step bits are dropped. The net result of this is
  // that even if you have fast mul/div hardware you cannot put it to
  // good use since the fixup would be too expensive.
  
   delta = 0;
   vpdiff = (step >> 3);
 
   if ( diff >= step ) 
    {
     delta = 4;
     diff -= step;
     vpdiff += step;
    }
   step >>= 1;
   if ( diff >= step  ) 
    {
     delta |= 2;
     diff -= step;
     vpdiff += step;
    }
   step >>= 1;
   if ( diff >= step ) 
    {
     delta |= 1;
     vpdiff += step;
    }
  
   // Step 3 - Update previous value 
   if ( sign )
     state_valprev -= vpdiff;
   else
     state_valprev += vpdiff;
   
   // Step 4 - Clamp previous value to 16 bits 
   if ( state_valprev > 32767 )
     state_valprev = 32767;
   else if ( state_valprev < -32768 )
     state_valprev = -32768;
   
   // Step 5 - Assemble value, update index and step values 
   delta |= sign;
   
   state_index += indexTable[delta];
   if ( state_index < 0 ) state_index = 0;
   if ( state_index > 88 ) state_index = 88;
   step = stepsizeTable[state_index];
   
   // Step 6 - Output value 
   if ( bufferstep ) 
    {
     outputbuffer = (delta << 4) & 0xf0;
     bufferstep = 0;
    } 
   else 
    {
     outdata[i] = (delta & 0x0f) | outputbuffer;
     bufferstep = 1;
    }
   //SPARK bufferstep = !bufferstep;
   if (bufferstep)
    bufferstep=0;
   else
    bufferstep=1;
  } /*  for (i=0; i<len; i++)  */
 
 // Output last step, if needed 
 if ( !bufferstep )
  {
   //SPARK *outp++ = outputbuffer;
   outdata[i] = outputbuffer;
  }

}

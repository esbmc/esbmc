#include<stdlib.h>
#include<stdio.h>
#include "original.h"
#include "quantized.h"

const float x[1][4] = {
 {
    0.8227847814559937,
    0.6818181872367859,
    0.8405796885490417,
    0.8799999952316284
 }
};

float y[1][3];
float y_q[1][3];

int main(){
    entry(x, y);
    entry_quantized(x, y_q);

    for(int i = 0; i < 3; i++) {
        printf("%f ", y[0][i]);
    }
    printf("\n");

    for(int i = 0; i < 3; i++) {
        printf("%f ", y_q[0][i]);
    }
    printf("\n");

    __ESBMC_assert(y[0][2] > y[0][0] && y[0][2] > y[0][1], "FAILED");
    __ESBMC_assert(y_q[0][2] > y_q[0][0] && y_q[0][2] > y_q[0][1], "FAILED");
    __ESBMC_assert(0, "FAILED");
}
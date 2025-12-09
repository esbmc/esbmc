extern void __VERIFIER_error(void);

extern unsigned char __VERIFIER_nondet_uchar(void);
extern unsigned short __VERIFIER_nondet_ushort(void);

long scast_helper(unsigned long i, unsigned char width){
    if((i & (1ULL << (width-1))) > 0){
        return ((long)i - (1LL<< width));
    }
    return i;
}

int main(){
        const unsigned short dut_offset__AT20 = __VERIFIER_nondet_ushort();
        const unsigned char valid__AT20 = __VERIFIER_nondet_uchar();
        const unsigned char in__AT20 = __VERIFIER_nondet_uchar();
        int flag = 0;
    if(!((unsigned char) (1U & ((unsigned char) (1U & (valid__AT20))))  ==  (unsigned char) (1U & ((unsigned char) (1U & ((unsigned char) (1U & (in__AT20))))  ==  (unsigned char) (1U & ((scast_helper((0U), 16)  <=  scast_helper(((unsigned short) ((unsigned short) (0U) << 8 | (unsigned short) (((unsigned char) (1U & ((unsigned char) (1U & (in__AT20))))  ==  (unsigned char) (1U & (1U))) ? (136U) : (0U)))  -  (unsigned short) ((unsigned short) (dut_offset__AT20))), 16)) ? (1U) : (0U))))))){
        __VERIFIER_error();
    }
}

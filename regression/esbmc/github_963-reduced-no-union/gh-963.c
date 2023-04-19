typedef unsigned char UINT8;
typedef unsigned short UINT16;
typedef unsigned int UINT32;
typedef unsigned long UINT64;

typedef struct _LIST_DETAILS {
    UINT8* ListPointer;
    UINT32 placeholder;
} LIST_DETAILS;

typedef struct _LIST_INFO {
    UINT32 tmp;
} LIST_INFO;

typedef struct _CONTEXT {
    UINT32 WorkingListIndex;
    #ifndef __PASS_WITHOUT_UNION
    union {
        LIST_INFO Arr[5];
        // LIST_INFO Element;
    };
    #endif
    LIST_DETAILS ListDetails[8];
} CONTEXT;

extern CONTEXT myCtx;

static __inline LIST_DETAILS* GetListDetails(UINT32 ListIndex)
{
    return &myCtx.ListDetails[ListIndex];
}

int main() {
        UINT8 tmp;

        #ifndef __PASS
        // GetListDetails(0)->ListPointer = &tmp;
        GetListDetails(1)->ListPointer = &tmp;
        #else
        myCtx.ListDetails[0].ListPointer = &tmp;
        myCtx.ListDetails[1].ListPointer = &tmp;
        #endif

    return 0;
}

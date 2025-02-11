#include <stdio.h>
#include <assert.h>
#define UINT8  unsigned char
#define UINT16 unsigned short int
#define UINT32 unsigned long
#define UINT64 unsigned long long

typedef struct {
	UINT32                       Offset;
	UINT32                       Width;
	UINT64                       Value;
	char* Name;
	UINT64                       Reserved;
} REGMAP_ENTRY; 

typedef struct {
    REGMAP_ENTRY* RegMap;
} REGMAP_TABLE; 

REGMAP_ENTRY m_PORT_RegMap[] = {
  {0x0,   32, 0x00000800, "REG0", 0xFFFF0000},
  {0x0,   32, 0x00000800, "REG1", 0xFFFF0000},
  {0x0,   32, 0x00000800, "REG2", 0xFFFF0000},
  {0x0,   32, 0x00000800, "REG3", 0xFFFF0000}
};

REGMAP_TABLE m_PORT_RegMapTable = {
  m_PORT_RegMap
};

typedef struct {
    void *       RegCntxtPortList1[4];
} IP_INST;

int main()
{
    IP_INST* pInst;
    pInst = malloc(sizeof(IP_INST));
    
    // SOMETHING WEIRD HERE
    int PortIdx = 0;    
    pInst->RegCntxtPortList1[PortIdx] = &m_PORT_RegMapTable;
    REGMAP_TABLE* RegMapTable = pInst->RegCntxtPortList1[PortIdx];
    REGMAP_ENTRY* pRegmap = &(RegMapTable->RegMap[3]);
    assert(pRegmap != 0);
    free(pInst);
    return 0;
}

int main_fixed()
{
    IP_INST* pInst;
    pInst = malloc(sizeof(IP_INST));
    
    // No PortIdx
    pInst->RegCntxtPortList1[0] = &m_PORT_RegMapTable;
    REGMAP_TABLE* RegMapTable = pInst->RegCntxtPortList1[0];
    REGMAP_ENTRY* pRegmap = &(RegMapTable->RegMap[3]);
    assert(pRegmap != 0);
    free(pInst);
    return 0;
}

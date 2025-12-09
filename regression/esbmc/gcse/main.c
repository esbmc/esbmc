#include <stdio.h>

#define UINT8  unsigned char
#define UINT16 unsigned short int
#define UINT32 unsigned long
#define UINT64 unsigned long long

typedef void (*REGISTER_HANDLER) (UINT64 Data64);

typedef struct {
	UINT32           Flags;
	UINT32           ReadCount;
	UINT32           WriteCount;
	REGISTER_HANDLER RegisterHandler;
} REGMAP_ENTRY_AUXILIARY; 

typedef struct {
	UINT32                       Offset;
	UINT32                       Width;
	UINT64                       Value;
	char* Name;
	UINT64                       Reserved;
	REGMAP_ENTRY_AUXILIARY Auxiliary;
} REGMAP_ENTRY; 

typedef struct {
    REGMAP_ENTRY* RegMap;
} REGMAP_TABLE; 

REGMAP_ENTRY m_RegMap[] = {
  {0x0,   32, 0x00000800, "REG0", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG1", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG2", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG3", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG4", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG5", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG6", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG7", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG8", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG9", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG10", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG11", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG12", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG13", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG14", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG15", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG16", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG17", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG18", 0xFFFF0000, {0} }
};

REGMAP_ENTRY m_PORT_RegMap[] = {
  {0x0,   32, 0x00000800, "REG0", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG1", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG2", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG3", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG4", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG5", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG6", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG7", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG8", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG9", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG10", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG11", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG12", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG13", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG14", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG15", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG16", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG17", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG18", 0xFFFF0000, {0} },
  {0x0,   32, 0x00000800, "REG19", 0xFFFF0000, {0} }
};

REGMAP_TABLE m_RegMapTable = {
  m_RegMap
};

REGMAP_TABLE m_PORT_RegMapTable = {
  m_PORT_RegMap
};

typedef struct {
    UINT8        PortIsDisabled[4];

    void *       RegCntxtHost;
    void *       RegCntxtPortList1[4];
    void *       RegCntxtPortList2[4];
} IP_INST;

UINT64 RegWrite(
    REGMAP_TABLE* RegMapTable,
    UINT32              EntryIndex,
    UINT64              Data64
)
{
    if (RegMapTable == NULL) {
        return 0;
    }

#if __REGMAP_DEFINED
    REGMAP_ENTRY* pRegmap = &(RegMapTable->RegMap[EntryIndex]);

    if (pRegmap->Auxiliary.RegisterHandler != NULL) {
        pRegmap->Auxiliary.RegisterHandler(Data64);
    }

    pRegmap->Auxiliary.Flags |= 1;
    pRegmap->Auxiliary.WriteCount++;
    pRegmap->Value = Data64;

    if (pRegmap->Auxiliary.RegisterHandler != NULL) {
        pRegmap->Auxiliary.RegisterHandler(Data64);
    }
#else
    if (RegMapTable->RegMap[EntryIndex].Auxiliary.RegisterHandler != NULL) {
        RegMapTable->RegMap[EntryIndex].Auxiliary.RegisterHandler(Data64);
    }

    RegMapTable->RegMap[EntryIndex].Auxiliary.Flags |= 1;
    RegMapTable->RegMap[EntryIndex].Auxiliary.WriteCount++;
    RegMapTable->RegMap[EntryIndex].Value = Data64;

    if (RegMapTable->RegMap[EntryIndex].Auxiliary.RegisterHandler != NULL) {
        RegMapTable->RegMap[EntryIndex].Auxiliary.RegisterHandler(Data64);
    }
#endif

    return Data64;
}

UINT64 RegRead(
    REGMAP_TABLE* RegMapTable,
    UINT32              EntryIndex
)
{
    if (RegMapTable == NULL) {
        return ~0;
    }

    #if __REGMAP_DEFINED
    REGMAP_ENTRY* pRegmap = &(RegMapTable->RegMap[EntryIndex]);

    if (pRegmap->Auxiliary.RegisterHandler != NULL) {
        pRegmap->Auxiliary.RegisterHandler(0);
    }

    pRegmap->Auxiliary.Flags |= 1;
    pRegmap->Auxiliary.ReadCount++;
    
    return pRegmap->Value;
    #else
    if (RegMapTable->RegMap[EntryIndex].Auxiliary.RegisterHandler != NULL) {
        RegMapTable->RegMap[EntryIndex].Auxiliary.RegisterHandler(0);
    }

    RegMapTable->RegMap[EntryIndex].Auxiliary.Flags |= 1;
    RegMapTable->RegMap[EntryIndex].Auxiliary.ReadCount++;

    return RegMapTable->RegMap[EntryIndex].Value;
    #endif
}

int main()
{
    IP_INST* pInst;
    pInst = __ESBMC_alloca(sizeof(IP_INST));
    pInst->RegCntxtHost = (void *)(&m_RegMapTable);

    for (UINT8 PortIdx = 0; PortIdx < 4; PortIdx++) {
        pInst->RegCntxtPortList1[PortIdx] = (void *)(&m_PORT_RegMapTable);
        pInst->RegCntxtPortList2[PortIdx] = (void *)(&m_PORT_RegMapTable);
    }

    for (UINT8 Port = 0; Port <= 3; Port++) {
        if (!pInst->PortIsDisabled[Port]) {
            RegWrite(pInst->RegCntxtPortList1[Port], 6, 0);
            RegWrite(pInst->RegCntxtPortList1[Port], 7, 0);
            RegWrite(pInst->RegCntxtPortList1[Port], 9, 0);
        }
    }

    UINT32 aux_index;
    __ESBMC_assume(aux_index < 4);
    UINT64 read_data1 = RegRead(pInst->RegCntxtPortList1[aux_index], 6);
    UINT64 read_data2 = RegRead(pInst->RegCntxtPortList1[aux_index], 7);
    UINT64 read_data3 = RegRead(pInst->RegCntxtPortList1[aux_index], 9);
    __ESBMC_assert(pInst->PortIsDisabled[aux_index] ||
        (
            read_data1 == 0 && 
            read_data2 == 0 &&
            read_data3 == 0),
        "Error: one of the registers is not set to 0");

    return 0;
}

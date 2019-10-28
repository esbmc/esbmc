extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume(int);
extern void abort (void) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
void avoid_zero(int y)
{
    if (!y) 
    {
        abort();
    }
}

struct Od_SubIndex;
struct Od_Index;
struct Od_IndexTable;
struct CanOpenNode;
struct _IO_FILE;
struct _IO_marker;
typedef struct _IO_FILE FILE;
typedef struct _IO_FILE __FILE;
typedef struct _IO_FILE _IO_FILE;
typedef long long  __quad_t;
typedef __quad_t __off64_t;
typedef void  _IO_lock_t;
typedef long  __off_t;
typedef struct CanOpenNode CanOpenNode;
typedef struct Od_IndexTable Od_IndexTable;
struct Od_IndexTable{
struct Od_Index* Id_MCDC_0;
struct Od_IndexTable* Id_MCDC_1;
} ;
typedef unsigned int  size_t;
struct _IO_FILE{
int  Id_MCDC_2;
char * Id_MCDC_3;
char * Id_MCDC_4;
char * Id_MCDC_5;
char * Id_MCDC_6;
char * Id_MCDC_7;
char * Id_MCDC_8;
char * Id_MCDC_9;
char * Id_MCDC_10;
char * Id_MCDC_11;
char * Id_MCDC_12;
char * Id_MCDC_13;
struct _IO_marker* Id_MCDC_14;
struct _IO_FILE* Id_MCDC_15;
int  Id_MCDC_16;
int  Id_MCDC_17;
__off_t Id_MCDC_18;
unsigned short  Id_MCDC_19;
signed char  Id_MCDC_20;
char  Id_MCDC_21 [1];
_IO_lock_t* Id_MCDC_22;
__off64_t Id_MCDC_23;
void * Id_MCDC_24;
void * Id_MCDC_25;
void * Id_MCDC_26;
void * Id_MCDC_27;
size_t Id_MCDC_28;
int  Id_MCDC_29;
char  Id_MCDC_30 [40];
} ;
typedef unsigned char  UINT8;
typedef unsigned short  UINT16;
typedef void  pEmcyCallBackFunc_t(UINT8, UINT16);
typedef struct {
UINT8 Id_MCDC_31;
pEmcyCallBackFunc_t* Id_MCDC_32;
} T5T5_261_0;
typedef T5T5_261_0 nozomi32nozomi32_2156_0;
typedef nozomi32nozomi32_2156_0 Emcy_buff;
typedef unsigned int  UINT32;
typedef struct {
UINT8 Id_MCDC_33;
UINT8 Id_MCDC_34;
UINT8 Id_MCDC_35;
UINT8 Id_MCDC_36;
UINT32 Id_MCDC_37;
UINT32 Id_MCDC_38;
void * Id_MCDC_39 [8];
UINT8 Id_MCDC_40 [8];
} T5T5_241_0;
typedef T5T5_241_0 nozomi32nozomi32_2074_0;
typedef nozomi32nozomi32_2074_0 Rxpdo;
typedef struct {
UINT8 Id_MCDC_33;
UINT8 Id_MCDC_34;
UINT8 Id_MCDC_35;
UINT8 Id_MCDC_36;
UINT32 Id_MCDC_37;
UINT32* Id_MCDC_39 [8];
} T5T5_231_0;
typedef T5T5_231_0 nozomi32nozomi32_2064_0;
typedef nozomi32nozomi32_2064_0 Txpdo;
typedef struct {
UINT32 Id_MCDC_41;
Txpdo Id_MCDC_42 [4];
Rxpdo Id_MCDC_43 [4];
} T5T5_253_0;
typedef T5T5_253_0 nozomi32nozomi32_2090_0;
typedef nozomi32nozomi32_2090_0 PDO_Buffer;
typedef struct {
UINT16 Id_MCDC_44;
UINT8 Id_MCDC_45;
UINT32 Id_MCDC_46;
UINT32 Id_MCDC_47;
UINT32 Id_MCDC_48;
UINT16 Id_MCDC_49;
UINT32 Id_MCDC_50;
UINT32 Id_MCDC_51;
UINT8 Id_MCDC_52 [252];
UINT16 Id_MCDC_53;
UINT8 Id_MCDC_54;
UINT8 Id_MCDC_55;
UINT32 Id_MCDC_56;
UINT32 Id_MCDC_57;
UINT32 Id_MCDC_58;
UINT32 Id_MCDC_59;
UINT32 Id_MCDC_60;
} T5T5_210_0;
typedef T5T5_210_0 nozomi32nozomi32_2004_0;
typedef nozomi32nozomi32_2004_0 SDO_Buffer;
typedef UINT32 pNmtCallBackFunc_t(UINT32);
typedef struct {
UINT32 Id_MCDC_61;
pNmtCallBackFunc_t* Id_MCDC_62;
} T5T5_204_0;
typedef T5T5_204_0 nozomi32nozomi32_2140_0;
typedef nozomi32nozomi32_2140_0 NMT_Buffer;
struct CanOpenNode{
UINT8 Id_MCDC_54;
UINT16 Id_MCDC_44;
UINT8 Id_MCDC_63;
NMT_Buffer* Id_MCDC_64;
SDO_Buffer* Id_MCDC_65;
PDO_Buffer* Id_MCDC_66;
Emcy_buff* Id_MCDC_67;
} ;
typedef unsigned long long  UINT64;
typedef char  INT8;
typedef char  CHAR;
typedef char * STRING;
typedef short  INT16;
typedef int  INT32;
typedef long  INT64;
typedef float  REAL32;
typedef unsigned char  BYTE;
struct Od_Index{
UINT16 Id_MCDC_68;
BYTE Id_MCDC_69;
BYTE Id_MCDC_70;
struct Od_SubIndex* Id_MCDC_71;
} ;
struct Od_SubIndex{
BYTE Id_MCDC_72;
UINT32 Id_MCDC_73;
CHAR Id_MCDC_74 [6];
REAL32 Id_MCDC_75;
BYTE Id_MCDC_76;
UINT32 Id_MCDC_77;
UINT32 Id_MCDC_78;
void * Id_MCDC_79;
} ;
extern int  strcmp(char  const* Id_MCDC_80, char  const* Id_MCDC_81);
UINT32 Id_MCDC_89(UINT16 Id_MCDC_82, UINT8 Id_MCDC_83, UINT32* Id_MCDC_84, UINT32* Id_MCDC_85, INT32* Id_MCDC_86, void * Id_MCDC_87, UINT8 Id_MCDC_88);
static UINT32 Id_MCDC_92(UINT32 Id_MCDC_90, UINT32 Id_MCDC_91);
static UINT32 Id_MCDC_95(UINT32 Id_MCDC_93, CHAR* Id_MCDC_94);
UINT32 Id_MCDC_96(UINT16 Id_MCDC_82, UINT8 Id_MCDC_83, UINT32* Id_MCDC_84, UINT32* Id_MCDC_85, INT32* Id_MCDC_86, void ** Id_MCDC_87);
UINT32 Id_MCDC_97();
extern void * malloc(size_t Id_MCDC_98);
void  Id_MCDC_99();
void  Id_MCDC_100();
extern FILE* fopen(char  const* Id_MCDC_101, char  const* Id_MCDC_102);
int  main();
void  __VERIFIER_assert(int  Id_MCDC_103);
unsigned int  Id_MCDC_104;
UINT16 Id_MCDC_105;
UINT16 Id_MCDC_106;
Od_IndexTable* Id_MCDC_107;
Od_IndexTable* Id_MCDC_108;
Od_IndexTable* Id_MCDC_109;
Od_IndexTable* Id_MCDC_110;
Od_IndexTable* Id_MCDC_111;
UINT32 Id_MCDC_112 [4];
CanOpenNode Id_MCDC_113;
UINT32 Id_MCDC_114=0;

UINT32 Id_MCDC_89(UINT16 Id_MCDC_82, UINT8 Id_MCDC_83, UINT32* Id_MCDC_84, UINT32* Id_MCDC_85, INT32* Id_MCDC_86, void * Id_MCDC_87, UINT8 Id_MCDC_88)
{
UINT32 Id_MCDC_115=0x80000000;
Od_IndexTable* Id_MCDC_116;
Id_MCDC_116 = Id_MCDC_111;
if(Id_MCDC_116 == (( void * ) 0))
{
return 0x08000023;
}
if(((( UINT16 ) Id_MCDC_82 > ( UINT16 ) Id_MCDC_105) && (( UINT16 ) Id_MCDC_82 <= ( UINT16 ) 0x19FF)) || ((( UINT16 ) Id_MCDC_82 > ( UINT16 ) Id_MCDC_106) && (( UINT16 ) Id_MCDC_82 <= ( UINT16 ) 0x15FF)))
{
if(Id_MCDC_83 == 0x01)
{
return 0x00;
}
}
switch(Id_MCDC_82 & 0xFF00)
{
case 0x1800: ;
case 0x1900: Id_MCDC_116 = Id_MCDC_107;
break;
case 0x1A00: ;
case 0x1B00: Id_MCDC_116 = Id_MCDC_108;
break;
case 0x1400: ;
case 0x1500: Id_MCDC_116 = Id_MCDC_109;
break;
case 0x1600: ;
case 0x1700: Id_MCDC_116 = Id_MCDC_110;
break;
default: Id_MCDC_116 = Id_MCDC_111;
break;
}
while(Id_MCDC_116->Id_MCDC_0->Id_MCDC_68 != Id_MCDC_82)
{
Id_MCDC_116 = Id_MCDC_116->Id_MCDC_1;
if(Id_MCDC_116 == (( void * ) 0))
{
return 0x06020000;
}
}
if(((Id_MCDC_83 == 0) || (Id_MCDC_83 < Id_MCDC_116->Id_MCDC_0->Id_MCDC_70)) && (strcmp((Id_MCDC_116->Id_MCDC_0->Id_MCDC_71 + Id_MCDC_83)->Id_MCDC_74, "rsvd")))
{
if((strcmp((Id_MCDC_116->Id_MCDC_0->Id_MCDC_71 + Id_MCDC_83)->Id_MCDC_74, "wo")))
;
else
{
return 0x06010001;
}
}
else
{
return 0x06090011;
}
return 0x00;
}
static UINT32 Id_MCDC_92(UINT32 Id_MCDC_90, UINT32 Id_MCDC_91)
{
UINT32 Id_MCDC_117=0;
UINT32 Id_MCDC_118=0;
UINT32 Id_MCDC_119=1;
UINT32 Id_MCDC_120=0;
if(Id_MCDC_90 > Id_MCDC_91)
{
Id_MCDC_117 = Id_MCDC_90;
Id_MCDC_118 = Id_MCDC_91;
}
else
if(Id_MCDC_90 < Id_MCDC_91)
{
Id_MCDC_117 = Id_MCDC_91;
Id_MCDC_118 = Id_MCDC_90;
}
else
{
return Id_MCDC_90;
}
while(Id_MCDC_119 != 0)
{
avoid_zero(Id_MCDC_118 != 0);
Id_MCDC_120 = ( UINT32 ) (Id_MCDC_117 / Id_MCDC_118);
Id_MCDC_119 = Id_MCDC_117 - (Id_MCDC_120 * Id_MCDC_118);
Id_MCDC_117 = Id_MCDC_118;
Id_MCDC_118 = Id_MCDC_119;
}
return Id_MCDC_117;
}
static UINT32 Id_MCDC_95(UINT32 Id_MCDC_93, CHAR* Id_MCDC_94)
{
UINT32 Id_MCDC_121=0;
UINT32 Id_MCDC_122=0;
if(Id_MCDC_93 < 2)
{
return 0;
}
if(Id_MCDC_93 != 2)
{
Id_MCDC_122 = (Id_MCDC_95(Id_MCDC_93 - 1, Id_MCDC_94 + 1));
}
else
{
Id_MCDC_122 = Id_MCDC_94[1];
}
Id_MCDC_121 = Id_MCDC_92(Id_MCDC_94[0], Id_MCDC_122);
avoid_zero(Id_MCDC_121 != 0);
return ("36_39854_4294972316" , __VERIFIER_assert(( long long  ) (Id_MCDC_94[0] * Id_MCDC_122) >= 0 && ( long long  ) (Id_MCDC_94[0] * Id_MCDC_122) <= 4294967295)) , ((Id_MCDC_94[0] * Id_MCDC_122) / Id_MCDC_121);
}
UINT32 Id_MCDC_96(UINT16 Id_MCDC_82, UINT8 Id_MCDC_83, UINT32* Id_MCDC_84, UINT32* Id_MCDC_85, INT32* Id_MCDC_86, void ** Id_MCDC_87)
{
Od_IndexTable* Id_MCDC_116;
Id_MCDC_116 = Id_MCDC_111;
if(Id_MCDC_116 == (( void * ) 0))
{
return 0x08000023;
}
while(Id_MCDC_116->Id_MCDC_0->Id_MCDC_68 != Id_MCDC_82)
{
Id_MCDC_116 = Id_MCDC_116->Id_MCDC_1;
if(Id_MCDC_116 == (( void * ) 0))
{
return 0x06020000;
}
}
if(((Id_MCDC_83 == 0) || (Id_MCDC_83 < Id_MCDC_116->Id_MCDC_0->Id_MCDC_70)) && (strcmp((Id_MCDC_116->Id_MCDC_0->Id_MCDC_71 + Id_MCDC_83)->Id_MCDC_74, "rsvd")))
{
if(strcmp((Id_MCDC_116->Id_MCDC_0->Id_MCDC_71 + Id_MCDC_83)->Id_MCDC_74, "wo"))
;
else
{
return 0x06010001;
}
}
else
{
return 0x06090011;
}
return 0x00;
}
UINT32 Id_MCDC_97()
{
UINT32 Id_MCDC_123;
UINT32 Id_MCDC_124;
UINT32 Id_MCDC_125;
UINT32 Id_MCDC_126;
UINT32 Id_MCDC_127;
UINT32 Id_MCDC_128=0x00;
UINT16 Id_MCDC_129;
UINT32 Id_MCDC_130=0x00;
UINT32 Id_MCDC_131=0x00;
UINT8 Id_MCDC_35=0;
UINT8 Id_MCDC_132=0;
UINT8 Id_MCDC_36=0;
UINT8 Id_MCDC_133=0;
UINT8 Id_MCDC_134=0x00;
INT32 Id_MCDC_135;
void * Id_MCDC_136;
UINT32* Id_MCDC_137;
UINT16 Id_MCDC_138;
Id_MCDC_123 = 0x1800;
for(Id_MCDC_132 = 0 ; Id_MCDC_132 < 4 ; Id_MCDC_132++ )
{
Id_MCDC_112[Id_MCDC_132] = 0x1A00 + Id_MCDC_132;
}
if(Id_MCDC_123 <= 0x1803)
{
for(Id_MCDC_138 = 0 ; Id_MCDC_138 < 4 ; Id_MCDC_138++ )
{
Id_MCDC_126 = Id_MCDC_123;
if(Id_MCDC_89(Id_MCDC_123, 1,  & Id_MCDC_131,  & Id_MCDC_130,  & Id_MCDC_135, ( void * )  & Id_MCDC_128, 0) != 0x00)
{
return 0xFFFF;
}
if(Id_MCDC_89(Id_MCDC_123, 2,  & Id_MCDC_131,  & Id_MCDC_130,  & Id_MCDC_135, ( void * )  & Id_MCDC_128, 0) != 0x00)
{
return 0xFFFF;
}
Id_MCDC_113.Id_MCDC_66->Id_MCDC_42[Id_MCDC_138].Id_MCDC_34 = ( UINT8 ) Id_MCDC_128;
Id_MCDC_123 = Id_MCDC_112[Id_MCDC_138];
if(Id_MCDC_89(Id_MCDC_123, 0,  & Id_MCDC_131,  & Id_MCDC_130,  & Id_MCDC_135, ( void * )  & Id_MCDC_128, 0) != 0x00)
{
return 0xFFFF;
}
Id_MCDC_35 = ( UINT8 ) Id_MCDC_128;
for(Id_MCDC_132 = 0 ; Id_MCDC_132 < Id_MCDC_35 ; Id_MCDC_132++ )
{
if(Id_MCDC_89(Id_MCDC_123, (Id_MCDC_132 + 1),  & Id_MCDC_131,  & Id_MCDC_130,  & Id_MCDC_135, ( void * )  & Id_MCDC_128, 0) != 0x00)
{
return 0xFFFF;
}
Id_MCDC_124 = Id_MCDC_128;
Id_MCDC_124 = (Id_MCDC_124 >> 16);
Id_MCDC_125 = Id_MCDC_128;
Id_MCDC_129 = ( UINT16 ) Id_MCDC_125;
Id_MCDC_129 = (Id_MCDC_129 >> 8);
Id_MCDC_134 = ( UINT8 ) Id_MCDC_129;
Id_MCDC_127 = Id_MCDC_96(Id_MCDC_124, Id_MCDC_134,  & Id_MCDC_131,  & Id_MCDC_130,  & Id_MCDC_135,  & Id_MCDC_136);
if(Id_MCDC_127 != 0x00)
{
return Id_MCDC_127;
}
}
Id_MCDC_126 = Id_MCDC_126 + 1;
Id_MCDC_123 = Id_MCDC_126;
}
}
return 0x00;
}
void  Id_MCDC_99()
{
CHAR Id_MCDC_139 [4]={0, 0, 0, 0};
UINT32 Id_MCDC_140=0;
Id_MCDC_113.Id_MCDC_66 = ( PDO_Buffer* ) malloc(372);
Id_MCDC_97();
for(Id_MCDC_140 = 0 ; Id_MCDC_140 < 4 ; Id_MCDC_140++ )
{
if((Id_MCDC_113.Id_MCDC_66->Id_MCDC_42[Id_MCDC_140].Id_MCDC_34 <= 240) && (Id_MCDC_113.Id_MCDC_66->Id_MCDC_42[Id_MCDC_140].Id_MCDC_34 >= 1))
{
Id_MCDC_139[Id_MCDC_140] = Id_MCDC_113.Id_MCDC_66->Id_MCDC_42[Id_MCDC_140].Id_MCDC_34;
}
else
{
Id_MCDC_139[Id_MCDC_140] = 1;
}
}
Id_MCDC_114 = Id_MCDC_95(4, Id_MCDC_139);
return ;
}
void  Id_MCDC_100()
{
int  Id_MCDC_141;
void * Id_MCDC_142;
Id_MCDC_99();
return ;
}
int  main()
{
void * Id_MCDC_143;
void * Id_MCDC_144;
void * Id_MCDC_145;
void * Id_MCDC_146;
void * Id_MCDC_147;
void * Id_MCDC_148;
int  Id_MCDC_149;
int  Id_MCDC_150;
int  Id_MCDC_151;
int  Id_MCDC_152;
int  Id_MCDC_153;
int  Id_MCDC_154;
int  Id_MCDC_155;
short  Id_MCDC_156;
short  Id_MCDC_157;
FILE* Id_MCDC_158;
unsigned short  Id_MCDC_159;
unsigned short  Id_MCDC_160;
unsigned short  Id_MCDC_161=0;
unsigned short  Id_MCDC_162=0;
char  Id_MCDC_163;
char  Id_MCDC_164;
Id_MCDC_158 = fopen("in.eds", "r");
if(Id_MCDC_158 == (( void * ) 0))
{
return 1;
}
;
Id_MCDC_100();
}
void  __VERIFIER_assert(int  Id_MCDC_103)
{
if( ! (Id_MCDC_103))
{
ERROR : __VERIFIER_error();
}
return ;
}

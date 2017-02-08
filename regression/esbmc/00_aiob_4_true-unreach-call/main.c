extern void __VERIFIER_error() __attribute__ ((__noreturn__));
struct Velocity_Mode;
struct PCMode;
struct DeviceControl;
struct _IO_FILE;
struct _IO_marker;
typedef struct DeviceControl DEV_CNTRL;
typedef struct _IO_FILE FILE;
typedef struct _IO_FILE __FILE;
typedef struct _IO_FILE _IO_FILE;
typedef long  __off_t;
typedef long long  __quad_t;
typedef __quad_t __off64_t;
typedef void  _IO_lock_t;
typedef struct PCMode POS_CTRL_MODE;
typedef struct Velocity_Mode VLMODE;
typedef unsigned int  size_t;
struct _IO_FILE{
int  Id_MCDC_0;
char * Id_MCDC_1;
char * Id_MCDC_2;
char * Id_MCDC_3;
char * Id_MCDC_4;
char * Id_MCDC_5;
char * Id_MCDC_6;
char * Id_MCDC_7;
char * Id_MCDC_8;
char * Id_MCDC_9;
char * Id_MCDC_10;
char * Id_MCDC_11;
struct _IO_marker* Id_MCDC_12;
struct _IO_FILE* Id_MCDC_13;
int  Id_MCDC_14;
int  Id_MCDC_15;
__off_t Id_MCDC_16;
unsigned short  Id_MCDC_17;
signed char  Id_MCDC_18;
char  Id_MCDC_19 [1];
_IO_lock_t* Id_MCDC_20;
__off64_t Id_MCDC_21;
void * Id_MCDC_22;
void * Id_MCDC_23;
void * Id_MCDC_24;
void * Id_MCDC_25;
size_t Id_MCDC_26;
int  Id_MCDC_27;
char  Id_MCDC_28 [40];
} ;
typedef unsigned char  UINT8;
typedef unsigned short  UINT16;
typedef unsigned int  UINT32;
typedef unsigned long long  UINT64;
typedef char  INT8;
typedef char  CHAR;
typedef char * STRING;
typedef short  INT16;
struct DeviceControl{
INT8* Id_MCDC_29;
UINT16* Id_MCDC_30;
UINT16* Id_MCDC_31;
INT16* Id_MCDC_32;
INT16* Id_MCDC_33;
INT16* Id_MCDC_34;
INT16* Id_MCDC_35;
INT16* Id_MCDC_36;
INT8* Id_MCDC_37;
} ;
typedef int  INT32;
struct PCMode{
INT16* Id_MCDC_38;
INT32* Id_MCDC_39;
INT32* Id_MCDC_40;
INT32* Id_MCDC_41;
UINT16* Id_MCDC_42;
UINT32* Id_MCDC_43;
UINT16* Id_MCDC_44;
INT32* Id_MCDC_45;
INT32* Id_MCDC_46;
} ;
struct Velocity_Mode{
INT16* Id_MCDC_47;
INT16* Id_MCDC_48;
INT16* Id_MCDC_49;
INT16* Id_MCDC_50;
INT16* Id_MCDC_51;
UINT32* Id_MCDC_52;
INT32* Id_MCDC_53;
INT32* Id_MCDC_54;
INT32* Id_MCDC_55;
INT32* Id_MCDC_56;
UINT8* Id_MCDC_57;
UINT32* Id_MCDC_58;
UINT32* Id_MCDC_59;
UINT32* Id_MCDC_60;
UINT32* Id_MCDC_61;
UINT32* Id_MCDC_62;
UINT32* Id_MCDC_63;
UINT32* Id_MCDC_64;
UINT32* Id_MCDC_65;
UINT32* Id_MCDC_66;
UINT32* Id_MCDC_67;
UINT32* Id_MCDC_68;
UINT32* Id_MCDC_69;
UINT32* Id_MCDC_70;
UINT32* Id_MCDC_71;
UINT32* Id_MCDC_72;
UINT32* Id_MCDC_73;
UINT32* Id_MCDC_74;
UINT32* Id_MCDC_75;
UINT32* Id_MCDC_76;
UINT16* Id_MCDC_77;
UINT32* Id_MCDC_78;
UINT16* Id_MCDC_79;
UINT32* Id_MCDC_80;
UINT16* Id_MCDC_81;
UINT32* Id_MCDC_82;
UINT32* Id_MCDC_83;
UINT32* Id_MCDC_84;
INT16* Id_MCDC_85;
INT16* Id_MCDC_86;
INT16* Id_MCDC_87;
} ;
typedef long  INT64;
typedef float  REAL32;
void  Id_MCDC_89(VLMODE* Id_MCDC_88);
extern FILE* fopen(char  const* Id_MCDC_90, char  const* Id_MCDC_91);
int  main();
void  __VERIFIER_assert(int  Id_MCDC_92);
unsigned int  Id_MCDC_93;

void  Id_MCDC_89(VLMODE* Id_MCDC_88)
{
void * Id_MCDC_94 [41]={Id_MCDC_88->Id_MCDC_47, Id_MCDC_88->Id_MCDC_48, Id_MCDC_88->Id_MCDC_49, Id_MCDC_88->Id_MCDC_50, Id_MCDC_88->Id_MCDC_51, Id_MCDC_88->Id_MCDC_52, Id_MCDC_88->Id_MCDC_53, Id_MCDC_88->Id_MCDC_54, Id_MCDC_88->Id_MCDC_55, Id_MCDC_88->Id_MCDC_56, Id_MCDC_88->Id_MCDC_57, Id_MCDC_88->Id_MCDC_58, Id_MCDC_88->Id_MCDC_59, Id_MCDC_88->Id_MCDC_60, Id_MCDC_88->Id_MCDC_61, Id_MCDC_88->Id_MCDC_62, Id_MCDC_88->Id_MCDC_63, Id_MCDC_88->Id_MCDC_64, Id_MCDC_88->Id_MCDC_65, Id_MCDC_88->Id_MCDC_66, Id_MCDC_88->Id_MCDC_67, Id_MCDC_88->Id_MCDC_68, Id_MCDC_88->Id_MCDC_69, Id_MCDC_88->Id_MCDC_70, Id_MCDC_88->Id_MCDC_71, Id_MCDC_88->Id_MCDC_72, Id_MCDC_88->Id_MCDC_73, Id_MCDC_88->Id_MCDC_74, Id_MCDC_88->Id_MCDC_75, Id_MCDC_88->Id_MCDC_76, Id_MCDC_88->Id_MCDC_77, Id_MCDC_88->Id_MCDC_78, Id_MCDC_88->Id_MCDC_79, Id_MCDC_88->Id_MCDC_80, Id_MCDC_88->Id_MCDC_81, Id_MCDC_88->Id_MCDC_82, Id_MCDC_88->Id_MCDC_83, Id_MCDC_88->Id_MCDC_84, Id_MCDC_88->Id_MCDC_85, Id_MCDC_88->Id_MCDC_86, Id_MCDC_88->Id_MCDC_87};
UINT32 const Id_MCDC_95 [42] [3]={{( UINT16 ) 0x6042, ( UINT8 ) 0x00, ( UINT32 ) 0x0003}, {( UINT16 ) 0x6043, ( UINT8 ) 0x00, ( UINT32 ) 0x0003}, {( UINT16 ) 0x6053, ( UINT8 ) 0x00, ( UINT32 ) 0x0003}, {( UINT16 ) 0x6054, ( UINT8 ) 0x00, ( UINT32 ) 0x0003}, {( UINT16 ) 0x6055, ( UINT8 ) 0x00, ( UINT32 ) 0x0003}, {( UINT16 ) 0x604E, ( UINT8 ) 0x00, ( UINT32 ) 0x0007}, {( UINT16 ) 0x604C, ( UINT8 ) 0x01, ( UINT32 ) 0x0004}, {( UINT16 ) 0x604C, ( UINT8 ) 0x02, ( UINT32 ) 0x0004}, {( UINT16 ) 0x604B, ( UINT8 ) 0x01, ( UINT32 ) 0x0003}, {( UINT16 ) 0x604B, ( UINT8 ) 0x02, ( UINT32 ) 0x0003}, {( UINT16 ) 0x604D, ( UINT8 ) 0x00, ( UINT32 ) 0x02}, {( UINT16 ) 0x6046, ( UINT8 ) 0x01, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6046, ( UINT8 ) 0x02, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6047, ( UINT8 ) 0x01, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6047, ( UINT8 ) 0x02, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6047, ( UINT8 ) 0x0003, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6047, ( UINT8 ) 0x0004, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6058, ( UINT8 ) 0x01, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6058, ( UINT8 ) 0x02, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6059, ( UINT8 ) 0x01, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6059, ( UINT8 ) 0x02, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6059, ( UINT8 ) 0x0003, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6059, ( UINT8 ) 0x0004, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6056, ( UINT8 ) 0x01, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6056, ( UINT8 ) 0x02, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6057, ( UINT8 ) 0x01, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6057, ( UINT8 ) 0x02, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6057, ( UINT8 ) 0x0003, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6057, ( UINT8 ) 0x0004, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6048, ( UINT8 ) 0x01, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6048, ( UINT8 ) 0x02, ( UINT32 ) 0x0006}, {( UINT16 ) 0x6049, ( UINT8 ) 0x01, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6049, ( UINT8 ) 0x02, ( UINT32 ) 0x0006}, {( UINT16 ) 0x604A, ( UINT8 ) 0x01, ( UINT32 ) 0x0007}, {( UINT16 ) 0x604A, ( UINT8 ) 0x02, ( UINT32 ) 0x0006}, {( UINT16 ) 0x604F, ( UINT8 ) 0x00, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6050, ( UINT8 ) 0x00, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6051, ( UINT8 ) 0x00, ( UINT32 ) 0x0007}, {( UINT16 ) 0x6044, ( UINT8 ) 0x00, ( UINT32 ) 0x0003}, {( UINT16 ) 0x6045, ( UINT8 ) 0x00, ( UINT32 ) 0x0003}, {( UINT16 ) 0x6052, ( UINT8 ) 0x00, ( UINT32 ) 0x0003}, {0x00, 0x00, 0x00}};
UINT32 Id_MCDC_96=0;
while((Id_MCDC_93 = Id_MCDC_96 , ("18_6630_4294986945" , __VERIFIER_assert((Id_MCDC_93 >= 0 && Id_MCDC_93 < 42))) , Id_MCDC_95[Id_MCDC_93])[0] != 0x00)
{
Id_MCDC_96++ ;
}
}
int  main()
{
void * Id_MCDC_97;
void * Id_MCDC_98;
void * Id_MCDC_99;
void * Id_MCDC_100;
void * Id_MCDC_101;
void * Id_MCDC_102;
int  Id_MCDC_103;
int  Id_MCDC_104;
int  Id_MCDC_105;
int  Id_MCDC_106;
int  Id_MCDC_107;
int  Id_MCDC_108;
int  Id_MCDC_109;
short  Id_MCDC_110;
short  Id_MCDC_111;
FILE* Id_MCDC_112;
unsigned short  Id_MCDC_113;
unsigned short  Id_MCDC_114;
unsigned short  Id_MCDC_115=0;
unsigned short  Id_MCDC_116=0;
char  Id_MCDC_117;
char  Id_MCDC_118;
VLMODE Id_MCDC_119;
POS_CTRL_MODE Id_MCDC_120;
DEV_CNTRL Id_MCDC_121;
Id_MCDC_112 = fopen("in.eds", "r");
if(Id_MCDC_112 == (( void * ) 0))
{
return 1;
}
;
Id_MCDC_89( & Id_MCDC_119);
}
void  __VERIFIER_assert(int  Id_MCDC_92)
{
if( ! (Id_MCDC_92))
{
ERROR : __VERIFIER_error();
}
return ;
}

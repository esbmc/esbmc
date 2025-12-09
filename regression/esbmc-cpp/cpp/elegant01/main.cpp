#include <cstdint>
#define NULL 0
#include <cstdlib>
extern "C"
{
  auto execute(
    /*PipelineExecutionContext* pipelineExecutionContext*/ uint8_t *var_0_111,
    /*WorkerContext* workerContext*/ uint8_t *var_0_112,
    /*NES::Runtime::TupleBuffer* inputTupleBuffer*/ uint8_t *var_0_113)
  {
    unsigned int N = nondet_uint();
    __ESBMC_assume(N > 1 && N < 5);
    var_0_111 = new uint8_t[N]; //(uint8_t*) malloc(sizeof(uint8_t)*N);
    var_0_112 = new uint8_t[N]; //(uint8_t*) malloc(sizeof(uint8_t)*N);
    var_0_113 = new uint8_t[N]; //(uint8_t*) malloc(sizeof(uint8_t)*N);
    // Verify input pointers
    __ESBMC_assume(var_0_111 != NULL);
    __ESBMC_assume(var_0_112 != NULL);
    __ESBMC_assume(var_0_113 != NULL);

    //variable declarations
    uint64_t var_0_0;
    uint64_t var_0_1;
    uint64_t var_0_2;
    uint64_t var_0_4;
    uint8_t *var_0_6 = new uint8_t;
    uint64_t var_0_7;
    uint8_t *var_0_8 = new uint8_t;
    uint64_t var_0_9;
    uint8_t *var_0_10 = new uint8_t;
    uint64_t var_0_11;
    uint8_t *var_5_102 = new uint8_t;
    uint8_t *var_5_103 = new uint8_t;
    uint8_t *var_5_104 = new uint8_t;
    uint64_t var_5_105;
    uint64_t var_5_106;
    uint64_t var_5_107;
    uint64_t var_5_108;
    uint64_t var_5_109;
    uint8_t *var_5_110 = new uint8_t;
    uint8_t *var_5_111 = new uint8_t;
    bool var_0_12;
    uint8_t *var_1_131 = new uint8_t;
    uint8_t *var_1_132 = new uint8_t;
    uint64_t var_1_133;
    uint64_t var_1_134;
    uint64_t var_1_135;
    uint64_t var_1_136;
    uint8_t *var_1_137;
    uint64_t var_1_138;
    uint8_t *var_1_139 = new uint8_t;
    uint8_t *var_1_140 = new uint8_t;
    uint64_t var_1_0;
    uint64_t var_1_1;
    uint8_t *var_1_2 = new uint8_t;
    uint64_t var_1_3;
    uint8_t *var_1_4 = new uint8_t;
    int64_t var_1_5;
    int32_t var_1_6;
    int64_t var_1_7;
    int64_t var_1_8;
    int32_t var_1_9;
    int64_t var_1_10;
    int64_t var_1_11;
    uint64_t var_1_12;
    uint64_t var_1_13;
    uint8_t *var_1_14 = new uint8_t;
    uint64_t var_1_15;
    uint8_t *var_1_16 = new uint8_t;
    uint64_t var_1_18;
    uint8_t *var_1_19 = new uint8_t;
    uint64_t var_1_21;
    uint8_t *var_1_22 = new uint8_t;
    uint64_t var_1_24;
    uint64_t var_1_25;
    uint64_t var_1_27;
    bool var_1_28;
    bool var_1_29;
    bool var_1_30;
    uint8_t *var_3_108;
    uint8_t *var_3_109;
    uint64_t var_3_110;
    uint64_t var_3_111;
    uint64_t var_3_112;
    uint64_t var_3_113;
    uint8_t *var_3_114;
    uint64_t var_3_115;
    uint8_t *var_3_116;
    uint8_t *var_3_4;
    uint8_t *var_3_6;
    uint64_t var_3_8;
    uint8_t *var_6_103;
    uint8_t *var_6_104;
    uint8_t *var_6_105;
    uint64_t var_6_106;
    uint64_t var_6_107;
    uint64_t var_6_108;
    uint64_t var_6_109;
    uint64_t var_6_110;
    uint8_t *var_6_111;
    uint8_t *var_6_112;
    uint64_t var_3_10;
    uint64_t var_3_11;
    uint8_t *var_4_101;
    uint8_t *var_4_102;
    uint8_t *var_4_103;
    uint64_t var_4_104;
    uint64_t var_4_105;
    uint64_t var_4_106;
    uint64_t var_4_107;
    uint64_t var_4_108;
    uint8_t *var_4_109;
    uint8_t *var_4_110;
    uint8_t *var_2_105;
    uint8_t *var_2_106;
    uint8_t *var_2_107;
    uint64_t var_2_108;
    uint64_t var_2_109;
    uint64_t var_2_110;

    //function definitions
    auto NES__Runtime__TupleBuffer__getWatermark =
      (uint64_t(*)(uint8_t *))0x7f7159b2d750;
    auto NES__Runtime__TupleBuffer__getOriginId =
      (uint64_t(*)(uint8_t *))0x7f7159b2d720;
    auto allocateBufferProxy = (uint8_t * (*)(uint8_t *))0x7f7159ab9a50;
    auto NES__Runtime__TupleBuffer__getBuffer =
      (uint8_t * (*)(uint8_t *))0x7f7159b2d6d0;
    auto NES__Runtime__TupleBuffer__getNumberOfTuples =
      (uint64_t(*)(uint8_t *))0x7f7159b2d6f0;
    auto NES__Runtime__TupleBuffer__setNumberOfTuples =
      (void (*)(uint8_t *, uint64_t))0x7f7159b2d700;
    auto NES__Runtime__TupleBuffer__setWatermark =
      (void (*)(uint8_t *, uint64_t))0x7f7159b2d760;
    auto NES__Runtime__TupleBuffer__setOriginId =
      (void (*)(uint8_t *, uint64_t))0x7f7159b2d730;
    auto emitBufferProxy =
      (void (*)(uint8_t *, uint8_t *, uint8_t *))0x7f7159ab9df0;
  //basic blocks
  Block_0:
    var_0_0 = 0;
    var_0_1 = 0;
    var_0_2 = NES__Runtime__TupleBuffer__getWatermark(var_0_113);
    var_0_4 = NES__Runtime__TupleBuffer__getOriginId(var_0_113);
    var_0_6 = allocateBufferProxy(var_0_112);
    var_0_7 = 0;
    var_0_8 = NES__Runtime__TupleBuffer__getBuffer(var_0_6);
    __ESBMC_assume(var_0_8 != NULL);
    var_0_9 = NES__Runtime__TupleBuffer__getNumberOfTuples(var_0_113);
    var_0_10 = NES__Runtime__TupleBuffer__getBuffer(var_0_113);
    __ESBMC_assume(var_0_10 != NULL);
    var_0_11 = 0;
    // prepare block arguments
    var_5_102 = var_0_112;
    var_5_103 = var_0_111;
    var_5_104 = var_0_6;
    var_5_105 = var_0_4;
    var_5_106 = var_0_2;
    var_5_107 = var_0_7;
    var_5_108 = var_0_11;
    var_5_109 = var_0_9;
    var_5_110 = var_0_8;
    var_5_111 = var_0_10;
    goto Block_5;

  // loop head
  Block_5:
    var_0_12 = var_5_108 < var_5_109;
    if (var_0_12)
    {
      // prepare block arguments
      var_1_131 = var_5_102;
      var_1_132 = var_5_103;
      var_1_133 = var_5_105;
      var_1_134 = var_5_106;
      var_1_135 = var_5_109;
      var_1_136 = var_5_108;
      var_1_137 = var_5_104;
      var_1_138 = var_5_107;
      var_1_139 = var_5_110;
      var_1_140 = var_5_111;
      goto Block_1;
    }
    else
    {
      // prepare block arguments
      var_2_105 = var_5_102;
      var_2_106 = var_5_103;
      var_2_107 = var_5_104;
      var_2_108 = var_5_105;
      var_2_109 = var_5_106;
      var_2_110 = var_5_107;
      goto Block_2;
    }

  Block_1:
    // load data from memory with address calculation
    var_1_0 = 8;
    var_1_1 = var_1_0 * var_1_136;
    var_1_2 = var_1_140 + var_1_1;
    var_1_3 = 0;
    var_1_4 = var_1_2 + var_1_3;
    __ESBMC_assume(var_1_4 != 0);
    var_1_5 = *reinterpret_cast<int64_t *>(var_1_4);

    // first map operator
    var_1_6 = 2;
    var_1_7 = (int64_t)var_1_6;
    var_1_8 = var_1_5 * var_1_7;

    // second map operator
    var_1_9 = 2;
    var_1_10 = (int64_t)var_1_9;
    var_1_11 = var_1_5 + var_1_10;

    // store fields in result buffer
    var_1_12 = 24;
    var_1_13 = var_1_12 * var_1_138;
    var_1_14 = var_1_139 + var_1_13;
    __ESBMC_assume(var_1_14 != 0);
    var_1_15 = 0;
    var_1_16 = var_1_14 + var_1_15;
    __ESBMC_assume(var_1_5 != 0);
    __ESBMC_assume(var_1_16 != 0);
    *reinterpret_cast<int64_t *>(var_1_16) = var_1_5;
    var_1_18 = 8;
    var_1_19 = var_1_14 + var_1_18;
    *reinterpret_cast<int64_t *>(var_1_19) = var_1_8;
    var_1_21 = 16;
    var_1_22 = var_1_14 + var_1_21;
    __ESBMC_assume(var_1_11 != NULL);
    __ESBMC_assume(var_1_22 != NULL);
    *reinterpret_cast<int64_t *>(var_1_22) = var_1_11;
    var_1_24 = 1;
    var_1_25 = var_1_138 + var_1_24;

    // check if result buffer is full
    var_1_27 = 170;
    var_1_28 = var_1_25 > var_1_27;
    var_1_29 = var_1_25 == var_1_27;
    var_1_30 = var_1_28 || var_1_29;
    if (var_1_30)
    {
      // prepare block arguments
      var_3_108 = var_1_131;
      var_3_109 = var_1_132;
      var_3_110 = var_1_133;
      var_3_111 = var_1_134;
      var_3_112 = var_1_135;
      var_3_113 = var_1_136;
      var_3_114 = var_1_137;
      var_3_115 = var_1_25;
      var_3_116 = var_1_140;
      goto Block_3;
    }
    else
    {
      // prepare block arguments
      var_4_101 = var_1_131;
      var_4_102 = var_1_132;
      var_4_103 = var_1_137;
      var_4_104 = var_1_133;
      var_4_105 = var_1_134;
      var_4_106 = var_1_25;
      var_4_107 = var_1_135;
      var_4_108 = var_1_136;
      var_4_109 = var_1_139;
      var_4_110 = var_1_140;
      goto Block_4;
    }

  Block_3:
    NES__Runtime__TupleBuffer__setNumberOfTuples(var_3_114, var_3_115);
    NES__Runtime__TupleBuffer__setWatermark(var_3_114, var_3_111);
    NES__Runtime__TupleBuffer__setOriginId(var_3_114, var_3_110);
    emitBufferProxy(var_3_108, var_3_109, var_3_114);
    var_3_4 = allocateBufferProxy(var_3_108);
    var_3_6 = NES__Runtime__TupleBuffer__getBuffer(var_3_4);
    var_3_8 = 0;
    // prepare block arguments
    var_6_103 = var_3_108;
    var_6_104 = var_3_109;
    var_6_105 = var_3_4;
    var_6_106 = var_3_110;
    var_6_107 = var_3_111;
    var_6_108 = var_3_8;
    var_6_109 = var_3_112;
    var_6_110 = var_3_113;
    var_6_111 = var_3_6;
    var_6_112 = var_3_116;
    goto Block_6;

  Block_6:
    var_3_10 = 1;
    var_3_11 = var_6_110 + var_3_10;
    // prepare block arguments
    var_5_102 = var_6_103;
    var_5_103 = var_6_104;
    var_5_104 = var_6_105;
    var_5_105 = var_6_106;
    var_5_106 = var_6_107;
    var_5_107 = var_6_108;
    var_5_108 = var_3_11;
    var_5_109 = var_6_109;
    var_5_110 = var_6_111;
    var_5_111 = var_6_112;
    goto Block_5;

  Block_4:
    // prepare block arguments
    var_6_103 = var_4_101;
    var_6_104 = var_4_102;
    var_6_105 = var_4_103;
    var_6_106 = var_4_104;
    var_6_107 = var_4_105;
    var_6_108 = var_4_106;
    var_6_109 = var_4_107;
    var_6_110 = var_4_108;
    var_6_111 = var_4_109;
    var_6_112 = var_4_110;
    goto Block_6;

  Block_2:
    NES__Runtime__TupleBuffer__setNumberOfTuples(var_2_107, var_2_110);
    NES__Runtime__TupleBuffer__setWatermark(var_2_107, var_2_109);
    NES__Runtime__TupleBuffer__setOriginId(var_2_107, var_2_108);
    emitBufferProxy(var_2_105, var_2_106, var_2_107);
    return;
  }
}

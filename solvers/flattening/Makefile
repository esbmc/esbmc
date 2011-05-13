include ../../config.inc
include ../../common
include ../../local.inc

CXXFLAGS+= -I../../util -I../../

OBJS= equality.o arrays.o functions.o sat_minimizer.o boolbv_width.o \
	boolbv.o bv_pointers.o bv_utils.o boolbv_abs.o boolbv_with.o \
	boolbv_typecast.o boolbv_index.o boolbv_member.o boolbv_if.o \
	boolbv_byte_extract.o boolbv_add_sub.o boolbv_mult.o \
	boolbv_constant.o boolbv_extractbit.o boolbv_bv_rel.o \
	boolbv_shift.o boolbv_case.o boolbv_cond.o \
	boolbv_concatenation.o boolbv_div.o boolbv_mod.o \
	boolbv_extractbits.o boolbv_replication.o boolbv_reduction.o \
	boolbv_overflow.o boolbv_get.o boolbv_bitwise.o \
	boolbv_equality.o boolbv_unary_minus.o boolbv_ieee_float_rel.o \
	pointer_logic.o boolbv_quantifier.o boolbv_byte_update.o \
	boolbv_array_of.o boolbv_map.o boolbv_type.o boolbv_union.o

ifdef MODULE_FLOATBV
CXXFLAGS += -DHAVE_FLOATBV
FLOATBVOBJ = ../../floatbv/floatbv.o
endif

all: flattening.o

flattening.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS) $(FLOATBVOBJ)

clean:
	-rm *.o

.PHONY: clean

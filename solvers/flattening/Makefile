ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/common

CXXFLAGS+= -I../../util -I../../

SRCS= equality.cpp arrays.cpp functions.cpp boolbv_width.cpp \
	boolbv.cpp bv_pointers.cpp bv_utils.cpp boolbv_abs.cpp \
        boolbv_with.cpp boolbv_typecast.cpp boolbv_index.cpp \
	boolbv_member.cpp boolbv_if.cpp boolbv_byte_extract.cpp \
	boolbv_add_sub.cpp boolbv_mult.cpp \
	boolbv_constant.cpp boolbv_extractbit.cpp boolbv_bv_rel.cpp \
	boolbv_shift.cpp boolbv_case.cpp boolbv_cond.cpp \
	boolbv_concatenation.cpp boolbv_div.cpp boolbv_mod.cpp \
	boolbv_extractbits.cpp boolbv_replication.cpp boolbv_reduction.cpp \
	boolbv_overflow.cpp boolbv_get.cpp boolbv_bitwise.cpp \
	boolbv_equality.cpp boolbv_unary_minus.cpp boolbv_ieee_float_rel.cpp \
	pointer_logic.cpp boolbv_quantifier.cpp boolbv_byte_update.cpp \
	boolbv_array_of.cpp boolbv_map.cpp boolbv_type.cpp boolbv_union.cpp

ifdef MINISAT
SRCS+= sat_minimizer.cpp
endif

ifdef MODULE_FLOATBV
CXXFLAGS += -DHAVE_FLOATBV
FLOATBVOBJ = ../../floatbv/floatbv.o
endif

all: flattening.o

flattening.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS) $(FLOATBVOBJ)

clean:
	-rm *.o
	-rm *.d

.PHONY: clean

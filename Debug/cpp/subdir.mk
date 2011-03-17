################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../cpp/cpp_constructor.cpp \
../cpp/cpp_convert_type.cpp \
../cpp/cpp_declaration.cpp \
../cpp/cpp_declarator.cpp \
../cpp/cpp_declarator_converter.cpp \
../cpp/cpp_final.cpp \
../cpp/cpp_id.cpp \
../cpp/cpp_is_pod.cpp \
../cpp/cpp_language.cpp \
../cpp/cpp_name.cpp \
../cpp/cpp_namespace_spec.cpp \
../cpp/cpp_parse_tree.cpp \
../cpp/cpp_parser.cpp \
../cpp/cpp_scope.cpp \
../cpp/cpp_scopes.cpp \
../cpp/cpp_token_buffer.cpp \
../cpp/cpp_typecast.cpp \
../cpp/cpp_typecheck.cpp \
../cpp/cpp_typecheck_bases.cpp \
../cpp/cpp_typecheck_code.cpp \
../cpp/cpp_typecheck_compound_type.cpp \
../cpp/cpp_typecheck_constructor.cpp \
../cpp/cpp_typecheck_conversions.cpp \
../cpp/cpp_typecheck_declaration.cpp \
../cpp/cpp_typecheck_enum_type.cpp \
../cpp/cpp_typecheck_expr.cpp \
../cpp/cpp_typecheck_fargs.cpp \
../cpp/cpp_typecheck_find_constructor.cpp \
../cpp/cpp_typecheck_function.cpp \
../cpp/cpp_typecheck_function_bodies.cpp \
../cpp/cpp_typecheck_initializer.cpp \
../cpp/cpp_typecheck_namespace.cpp \
../cpp/cpp_typecheck_resolve.cpp \
../cpp/cpp_typecheck_template.cpp \
../cpp/cpp_typecheck_type.cpp \
../cpp/cpp_typecheck_virtual_table.cpp \
../cpp/expr2cpp.cpp \
../cpp/irep2name.cpp \
../cpp/lex.yy.cpp \
../cpp/parse.cpp \
../cpp/template_map.cpp 

OBJS += \
./cpp/cpp_constructor.o \
./cpp/cpp_convert_type.o \
./cpp/cpp_declaration.o \
./cpp/cpp_declarator.o \
./cpp/cpp_declarator_converter.o \
./cpp/cpp_final.o \
./cpp/cpp_id.o \
./cpp/cpp_is_pod.o \
./cpp/cpp_language.o \
./cpp/cpp_name.o \
./cpp/cpp_namespace_spec.o \
./cpp/cpp_parse_tree.o \
./cpp/cpp_parser.o \
./cpp/cpp_scope.o \
./cpp/cpp_scopes.o \
./cpp/cpp_token_buffer.o \
./cpp/cpp_typecast.o \
./cpp/cpp_typecheck.o \
./cpp/cpp_typecheck_bases.o \
./cpp/cpp_typecheck_code.o \
./cpp/cpp_typecheck_compound_type.o \
./cpp/cpp_typecheck_constructor.o \
./cpp/cpp_typecheck_conversions.o \
./cpp/cpp_typecheck_declaration.o \
./cpp/cpp_typecheck_enum_type.o \
./cpp/cpp_typecheck_expr.o \
./cpp/cpp_typecheck_fargs.o \
./cpp/cpp_typecheck_find_constructor.o \
./cpp/cpp_typecheck_function.o \
./cpp/cpp_typecheck_function_bodies.o \
./cpp/cpp_typecheck_initializer.o \
./cpp/cpp_typecheck_namespace.o \
./cpp/cpp_typecheck_resolve.o \
./cpp/cpp_typecheck_template.o \
./cpp/cpp_typecheck_type.o \
./cpp/cpp_typecheck_virtual_table.o \
./cpp/expr2cpp.o \
./cpp/irep2name.o \
./cpp/lex.yy.o \
./cpp/parse.o \
./cpp/template_map.o 

CPP_DEPS += \
./cpp/cpp_constructor.d \
./cpp/cpp_convert_type.d \
./cpp/cpp_declaration.d \
./cpp/cpp_declarator.d \
./cpp/cpp_declarator_converter.d \
./cpp/cpp_final.d \
./cpp/cpp_id.d \
./cpp/cpp_is_pod.d \
./cpp/cpp_language.d \
./cpp/cpp_name.d \
./cpp/cpp_namespace_spec.d \
./cpp/cpp_parse_tree.d \
./cpp/cpp_parser.d \
./cpp/cpp_scope.d \
./cpp/cpp_scopes.d \
./cpp/cpp_token_buffer.d \
./cpp/cpp_typecast.d \
./cpp/cpp_typecheck.d \
./cpp/cpp_typecheck_bases.d \
./cpp/cpp_typecheck_code.d \
./cpp/cpp_typecheck_compound_type.d \
./cpp/cpp_typecheck_constructor.d \
./cpp/cpp_typecheck_conversions.d \
./cpp/cpp_typecheck_declaration.d \
./cpp/cpp_typecheck_enum_type.d \
./cpp/cpp_typecheck_expr.d \
./cpp/cpp_typecheck_fargs.d \
./cpp/cpp_typecheck_find_constructor.d \
./cpp/cpp_typecheck_function.d \
./cpp/cpp_typecheck_function_bodies.d \
./cpp/cpp_typecheck_initializer.d \
./cpp/cpp_typecheck_namespace.d \
./cpp/cpp_typecheck_resolve.d \
./cpp/cpp_typecheck_template.d \
./cpp/cpp_typecheck_type.d \
./cpp/cpp_typecheck_virtual_table.d \
./cpp/expr2cpp.d \
./cpp/irep2name.d \
./cpp/lex.yy.d \
./cpp/parse.d \
./cpp/template_map.d 


# Each subdirectory must supply rules for building sources it contributes
cpp/%.o: ../cpp/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



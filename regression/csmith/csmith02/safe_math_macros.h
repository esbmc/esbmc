
#ifndef SAFE_MATH_H
#define SAFE_MATH_H







#define safe_unary_minus_func_int8_t_s(_si) \
  ({ int8_t si = (_si) ; \
   (((int8_t)(si))==(INT8_MIN)) && (INT8_MAX>=INT_MAX)? \
    ((int8_t)(si)): \
    (int8_t) (-((int8_t)(si))) \
  ;})

#define safe_add_func_int8_t_s_s(_si1,_si2) \
		({ int8_t si1 = (_si1); int8_t si2 = (_si2) ; \
                 (((((int8_t)(si1))>((int8_t)0)) && (((int8_t)(si2))>((int8_t)0)) && (((int8_t)(si1)) > ((INT8_MAX)-((int8_t)(si2))))) \
		  || ((((int8_t)(si1))<((int8_t)0)) && (((int8_t)(si2))<((int8_t)0)) && (((int8_t)(si1)) < ((INT8_MIN)-((int8_t)(si2)))))) \
		 && (INT8_MAX>=INT_MAX)? \
		 ((int8_t)(si1)) :						\
		 (int8_t) (((int8_t)(si1)) + ((int8_t)(si2)))		\
		;}) 

#define safe_sub_func_int8_t_s_s(_si1,_si2) \
		({ int8_t si1 = (_si1); int8_t si2 = (_si2) ; \
                (((((int8_t)(si1))^((int8_t)(si2))) \
		& (((((int8_t)(si1)) ^ ((((int8_t)(si1))^((int8_t)(si2))) \
		& (((int8_t)1) << (sizeof(int8_t)*CHAR_BIT-1))))-((int8_t)(si2)))^((int8_t)(si2)))) < ((int8_t)0)) \
		&& (INT8_MAX>=INT_MAX)? \
		((int8_t)(si1)) \
		: (int8_t) (((int8_t)(si1)) - ((int8_t)(si2))) \
		;})

#define safe_mul_func_int8_t_s_s(_si1,_si2) \
  ({ int8_t si1 = (_si1); int8_t si2 = (_si2) ; \
  (((((int8_t)(si1)) > ((int8_t)0)) && (((int8_t)(si2)) > ((int8_t)0)) && (((int8_t)(si1)) > ((INT8_MAX) / ((int8_t)(si2))))) || \
  ((((int8_t)(si1)) > ((int8_t)0)) && (((int8_t)(si2)) <= ((int8_t)0)) && (((int8_t)(si2)) < ((INT8_MIN) / ((int8_t)(si1))))) || \
  ((((int8_t)(si1)) <= ((int8_t)0)) && (((int8_t)(si2)) > ((int8_t)0)) && (((int8_t)(si1)) < ((INT8_MIN) / ((int8_t)(si2))))) || \
  ((((int8_t)(si1)) <= ((int8_t)0)) && (((int8_t)(si2)) <= ((int8_t)0)) && (((int8_t)(si1)) != ((int8_t)0)) && (((int8_t)(si2)) < ((INT8_MAX) / ((int8_t)(si1)))))) && (INT8_MAX>=INT_MAX) \
  ? ((int8_t)(si1)) \
  : ((int8_t) (((int8_t)(si1)) * ((int8_t)(si2))));})

#define safe_mod_func_int8_t_s_s(_si1,_si2) \
  ({ int8_t si1 = (_si1); int8_t si2 = (_si2) ; \
  ((((int8_t)(si2)) == ((int8_t)0)) || ((((int8_t)(si1)) == (INT8_MIN)) && (((int8_t)(si2)) == ((int8_t)-1)))) \
  ? ((int8_t)(si1)) \
  : (int8_t) (((int8_t)(si1)) % ((int8_t)(si2)));})

#define safe_div_func_int8_t_s_s(_si1,_si2) \
  ({ int8_t si1 = (_si1); int8_t si2 = (_si2) ; \
  ((((int8_t)(si2)) == ((int8_t)0)) || ((((int8_t)(si1)) == (INT8_MIN)) && (((int8_t)(si2)) == ((int8_t)-1)))) \
  ? ((int8_t)(si1)) \
  : (int8_t) (((int8_t)(si1)) / ((int8_t)(si2)));})

#define safe_lshift_func_int8_t_s_s(_left,_right) \
  ({ int8_t left = (_left); int right = (_right) ; \
   ((((int8_t)(left)) < ((int8_t)0)) \
  || (((int)(right)) < 0) \
  || (((int)(right)) >= 32) \
  || (((int8_t)(left)) > ((INT8_MAX) >> ((int)(right))))) \
  ? ((int8_t)(left)) \
  : (int8_t) (((int8_t)(left)) << ((int)(right)));})

#define safe_lshift_func_int8_t_s_u(_left,_right) \
  ({ int8_t left = (_left); unsigned int right = (_right) ; \
   ((((int8_t)(left)) < ((int8_t)0)) \
  || (((unsigned int)(right)) >= 32) \
  || (((int8_t)(left)) > ((INT8_MAX) >> ((unsigned int)(right))))) \
  ? ((int8_t)(left)) \
  : (int8_t) (((int8_t)(left)) << ((unsigned int)(right)));})

#define safe_rshift_func_int8_t_s_s(_left,_right) \
	({ int8_t left = (_left); int right = (_right) ; \
        ((((int8_t)(left)) < ((int8_t)0)) \
			 || (((int)(right)) < 0) \
			 || (((int)(right)) >= 32)) \
			? ((int8_t)(left)) \
			: (int8_t) (((int8_t)(left)) >> ((int)(right)));})

#define safe_rshift_func_int8_t_s_u(_left,_right) \
  ({ int8_t left = (_left); unsigned int right = (_right) ; \
   ((((int8_t)(left)) < ((int8_t)0)) \
			 || (((unsigned int)(right)) >= 32)) \
			? ((int8_t)(left)) \
			: (int8_t) (((int8_t)(left)) >> ((unsigned int)(right)));})



#define safe_unary_minus_func_int16_t_s(_si) \
  ({ int16_t si = (_si) ; \
   (((int16_t)(si))==(INT16_MIN)) && (INT16_MAX>=INT_MAX)? \
    ((int16_t)(si)): \
    (int16_t) (-((int16_t)(si))) \
  ;})

#define safe_add_func_int16_t_s_s(_si1,_si2) \
		({ int16_t si1 = (_si1); int16_t si2 = (_si2) ; \
                 (((((int16_t)(si1))>((int16_t)0)) && (((int16_t)(si2))>((int16_t)0)) && (((int16_t)(si1)) > ((INT16_MAX)-((int16_t)(si2))))) \
		  || ((((int16_t)(si1))<((int16_t)0)) && (((int16_t)(si2))<((int16_t)0)) && (((int16_t)(si1)) < ((INT16_MIN)-((int16_t)(si2)))))) \
		 && (INT16_MAX>=INT_MAX)? \
		 ((int16_t)(si1)) :						\
		 (int16_t) (((int16_t)(si1)) + ((int16_t)(si2)))		\
		;}) 

#define safe_sub_func_int16_t_s_s(_si1,_si2) \
		({ int16_t si1 = (_si1); int16_t si2 = (_si2) ; \
                (((((int16_t)(si1))^((int16_t)(si2))) \
		& (((((int16_t)(si1)) ^ ((((int16_t)(si1))^((int16_t)(si2))) \
		& (((int16_t)1) << (sizeof(int16_t)*CHAR_BIT-1))))-((int16_t)(si2)))^((int16_t)(si2)))) < ((int16_t)0)) \
		&& (INT16_MAX>=INT_MAX)? \
		((int16_t)(si1)) \
		: (int16_t) (((int16_t)(si1)) - ((int16_t)(si2))) \
		;})

#define safe_mul_func_int16_t_s_s(_si1,_si2) \
  ({ int16_t si1 = (_si1); int16_t si2 = (_si2) ; \
  (((((int16_t)(si1)) > ((int16_t)0)) && (((int16_t)(si2)) > ((int16_t)0)) && (((int16_t)(si1)) > ((INT16_MAX) / ((int16_t)(si2))))) || \
  ((((int16_t)(si1)) > ((int16_t)0)) && (((int16_t)(si2)) <= ((int16_t)0)) && (((int16_t)(si2)) < ((INT16_MIN) / ((int16_t)(si1))))) || \
  ((((int16_t)(si1)) <= ((int16_t)0)) && (((int16_t)(si2)) > ((int16_t)0)) && (((int16_t)(si1)) < ((INT16_MIN) / ((int16_t)(si2))))) || \
  ((((int16_t)(si1)) <= ((int16_t)0)) && (((int16_t)(si2)) <= ((int16_t)0)) && (((int16_t)(si1)) != ((int16_t)0)) && (((int16_t)(si2)) < ((INT16_MAX) / ((int16_t)(si1)))))) && (INT16_MAX>=INT_MAX) \
  ? ((int16_t)(si1)) \
  : ((int16_t) (((int16_t)(si1)) * ((int16_t)(si2))));})

#define safe_mod_func_int16_t_s_s(_si1,_si2) \
  ({ int16_t si1 = (_si1); int16_t si2 = (_si2) ; \
  ((((int16_t)(si2)) == ((int16_t)0)) || ((((int16_t)(si1)) == (INT16_MIN)) && (((int16_t)(si2)) == ((int16_t)-1)))) \
  ? ((int16_t)(si1)) \
  : (int16_t) (((int16_t)(si1)) % ((int16_t)(si2)));})

#define safe_div_func_int16_t_s_s(_si1,_si2) \
  ({ int16_t si1 = (_si1); int16_t si2 = (_si2) ; \
  ((((int16_t)(si2)) == ((int16_t)0)) || ((((int16_t)(si1)) == (INT16_MIN)) && (((int16_t)(si2)) == ((int16_t)-1)))) \
  ? ((int16_t)(si1)) \
  : (int16_t) (((int16_t)(si1)) / ((int16_t)(si2)));})

#define safe_lshift_func_int16_t_s_s(_left,_right) \
  ({ int16_t left = (_left); int right = (_right) ; \
   ((((int16_t)(left)) < ((int16_t)0)) \
  || (((int)(right)) < 0) \
  || (((int)(right)) >= 32) \
  || (((int16_t)(left)) > ((INT16_MAX) >> ((int)(right))))) \
  ? ((int16_t)(left)) \
  : (int16_t) (((int16_t)(left)) << ((int)(right)));})

#define safe_lshift_func_int16_t_s_u(_left,_right) \
  ({ int16_t left = (_left); unsigned int right = (_right) ; \
   ((((int16_t)(left)) < ((int16_t)0)) \
  || (((unsigned int)(right)) >= 32) \
  || (((int16_t)(left)) > ((INT16_MAX) >> ((unsigned int)(right))))) \
  ? ((int16_t)(left)) \
  : (int16_t) (((int16_t)(left)) << ((unsigned int)(right)));})

#define safe_rshift_func_int16_t_s_s(_left,_right) \
	({ int16_t left = (_left); int right = (_right) ; \
        ((((int16_t)(left)) < ((int16_t)0)) \
			 || (((int)(right)) < 0) \
			 || (((int)(right)) >= 32)) \
			? ((int16_t)(left)) \
			: (int16_t) (((int16_t)(left)) >> ((int)(right)));})

#define safe_rshift_func_int16_t_s_u(_left,_right) \
  ({ int16_t left = (_left); unsigned int right = (_right) ; \
   ((((int16_t)(left)) < ((int16_t)0)) \
			 || (((unsigned int)(right)) >= 32)) \
			? ((int16_t)(left)) \
			: (int16_t) (((int16_t)(left)) >> ((unsigned int)(right)));})



#define safe_unary_minus_func_int32_t_s(_si) \
  ({ int32_t si = (_si) ; \
   (((int32_t)(si))==(INT32_MIN)) && (INT32_MAX>=INT_MAX)? \
    ((int32_t)(si)): \
    (int32_t) (-((int32_t)(si))) \
  ;})

#define safe_add_func_int32_t_s_s(_si1,_si2) \
		({ int32_t si1 = (_si1); int32_t si2 = (_si2) ; \
                 (((((int32_t)(si1))>((int32_t)0)) && (((int32_t)(si2))>((int32_t)0)) && (((int32_t)(si1)) > ((INT32_MAX)-((int32_t)(si2))))) \
		  || ((((int32_t)(si1))<((int32_t)0)) && (((int32_t)(si2))<((int32_t)0)) && (((int32_t)(si1)) < ((INT32_MIN)-((int32_t)(si2)))))) \
		 && (INT32_MAX>=INT_MAX)? \
		 ((int32_t)(si1)) :						\
		 (int32_t) (((int32_t)(si1)) + ((int32_t)(si2)))		\
		;}) 

#define safe_sub_func_int32_t_s_s(_si1,_si2) \
		({ int32_t si1 = (_si1); int32_t si2 = (_si2) ; \
                (((((int32_t)(si1))^((int32_t)(si2))) \
		& (((((int32_t)(si1)) ^ ((((int32_t)(si1))^((int32_t)(si2))) \
		& (((int32_t)1) << (sizeof(int32_t)*CHAR_BIT-1))))-((int32_t)(si2)))^((int32_t)(si2)))) < ((int32_t)0)) \
		&& (INT32_MAX>=INT_MAX)? \
		((int32_t)(si1)) \
		: (int32_t) (((int32_t)(si1)) - ((int32_t)(si2))) \
		;})

#define safe_mul_func_int32_t_s_s(_si1,_si2) \
  ({ int32_t si1 = (_si1); int32_t si2 = (_si2) ; \
  (((((int32_t)(si1)) > ((int32_t)0)) && (((int32_t)(si2)) > ((int32_t)0)) && (((int32_t)(si1)) > ((INT32_MAX) / ((int32_t)(si2))))) || \
  ((((int32_t)(si1)) > ((int32_t)0)) && (((int32_t)(si2)) <= ((int32_t)0)) && (((int32_t)(si2)) < ((INT32_MIN) / ((int32_t)(si1))))) || \
  ((((int32_t)(si1)) <= ((int32_t)0)) && (((int32_t)(si2)) > ((int32_t)0)) && (((int32_t)(si1)) < ((INT32_MIN) / ((int32_t)(si2))))) || \
  ((((int32_t)(si1)) <= ((int32_t)0)) && (((int32_t)(si2)) <= ((int32_t)0)) && (((int32_t)(si1)) != ((int32_t)0)) && (((int32_t)(si2)) < ((INT32_MAX) / ((int32_t)(si1)))))) && (INT32_MAX>=INT_MAX) \
  ? ((int32_t)(si1)) \
  : ((int32_t) (((int32_t)(si1)) * ((int32_t)(si2))));})

#define safe_mod_func_int32_t_s_s(_si1,_si2) \
  ({ int32_t si1 = (_si1); int32_t si2 = (_si2) ; \
  ((((int32_t)(si2)) == ((int32_t)0)) || ((((int32_t)(si1)) == (INT32_MIN)) && (((int32_t)(si2)) == ((int32_t)-1)))) \
  ? ((int32_t)(si1)) \
  : (int32_t) (((int32_t)(si1)) % ((int32_t)(si2)));})

#define safe_div_func_int32_t_s_s(_si1,_si2) \
  ({ int32_t si1 = (_si1); int32_t si2 = (_si2) ; \
  ((((int32_t)(si2)) == ((int32_t)0)) || ((((int32_t)(si1)) == (INT32_MIN)) && (((int32_t)(si2)) == ((int32_t)-1)))) \
  ? ((int32_t)(si1)) \
  : (int32_t) (((int32_t)(si1)) / ((int32_t)(si2)));})

#define safe_lshift_func_int32_t_s_s(_left,_right) \
  ({ int32_t left = (_left); int right = (_right) ; \
   ((((int32_t)(left)) < ((int32_t)0)) \
  || (((int)(right)) < 0) \
  || (((int)(right)) >= 32) \
  || (((int32_t)(left)) > ((INT32_MAX) >> ((int)(right))))) \
  ? ((int32_t)(left)) \
  : (int32_t) (((int32_t)(left)) << ((int)(right)));})

#define safe_lshift_func_int32_t_s_u(_left,_right) \
  ({ int32_t left = (_left); unsigned int right = (_right) ; \
   ((((int32_t)(left)) < ((int32_t)0)) \
  || (((unsigned int)(right)) >= 32) \
  || (((int32_t)(left)) > ((INT32_MAX) >> ((unsigned int)(right))))) \
  ? ((int32_t)(left)) \
  : (int32_t) (((int32_t)(left)) << ((unsigned int)(right)));})

#define safe_rshift_func_int32_t_s_s(_left,_right) \
	({ int32_t left = (_left); int right = (_right) ; \
        ((((int32_t)(left)) < ((int32_t)0)) \
			 || (((int)(right)) < 0) \
			 || (((int)(right)) >= 32)) \
			? ((int32_t)(left)) \
			: (int32_t) (((int32_t)(left)) >> ((int)(right)));})

#define safe_rshift_func_int32_t_s_u(_left,_right) \
  ({ int32_t left = (_left); unsigned int right = (_right) ; \
   ((((int32_t)(left)) < ((int32_t)0)) \
			 || (((unsigned int)(right)) >= 32)) \
			? ((int32_t)(left)) \
			: (int32_t) (((int32_t)(left)) >> ((unsigned int)(right)));})



#define safe_unary_minus_func_int64_t_s(_si) \
  ({ int64_t si = (_si) ; \
   (((int64_t)(si))==(INT64_MIN)) && (INT64_MAX>=INT_MAX)? \
    ((int64_t)(si)): \
    (int64_t) (-((int64_t)(si))) \
  ;})

#define safe_add_func_int64_t_s_s(_si1,_si2) \
		({ int64_t si1 = (_si1); int64_t si2 = (_si2) ; \
                 (((((int64_t)(si1))>((int64_t)0)) && (((int64_t)(si2))>((int64_t)0)) && (((int64_t)(si1)) > ((INT64_MAX)-((int64_t)(si2))))) \
		  || ((((int64_t)(si1))<((int64_t)0)) && (((int64_t)(si2))<((int64_t)0)) && (((int64_t)(si1)) < ((INT64_MIN)-((int64_t)(si2)))))) \
		 && (INT64_MAX>=INT_MAX)? \
		 ((int64_t)(si1)) :						\
		 (int64_t) (((int64_t)(si1)) + ((int64_t)(si2)))		\
		;}) 

#define safe_sub_func_int64_t_s_s(_si1,_si2) \
		({ int64_t si1 = (_si1); int64_t si2 = (_si2) ; \
                (((((int64_t)(si1))^((int64_t)(si2))) \
		& (((((int64_t)(si1)) ^ ((((int64_t)(si1))^((int64_t)(si2))) \
		& (((int64_t)1) << (sizeof(int64_t)*CHAR_BIT-1))))-((int64_t)(si2)))^((int64_t)(si2)))) < ((int64_t)0)) \
		&& (INT64_MAX>=INT_MAX)? \
		((int64_t)(si1)) \
		: (int64_t) (((int64_t)(si1)) - ((int64_t)(si2))) \
		;})

#define safe_mul_func_int64_t_s_s(_si1,_si2) \
  ({ int64_t si1 = (_si1); int64_t si2 = (_si2) ; \
  (((((int64_t)(si1)) > ((int64_t)0)) && (((int64_t)(si2)) > ((int64_t)0)) && (((int64_t)(si1)) > ((INT64_MAX) / ((int64_t)(si2))))) || \
  ((((int64_t)(si1)) > ((int64_t)0)) && (((int64_t)(si2)) <= ((int64_t)0)) && (((int64_t)(si2)) < ((INT64_MIN) / ((int64_t)(si1))))) || \
  ((((int64_t)(si1)) <= ((int64_t)0)) && (((int64_t)(si2)) > ((int64_t)0)) && (((int64_t)(si1)) < ((INT64_MIN) / ((int64_t)(si2))))) || \
  ((((int64_t)(si1)) <= ((int64_t)0)) && (((int64_t)(si2)) <= ((int64_t)0)) && (((int64_t)(si1)) != ((int64_t)0)) && (((int64_t)(si2)) < ((INT64_MAX) / ((int64_t)(si1)))))) && (INT64_MAX>=INT_MAX) \
  ? ((int64_t)(si1)) \
  : ((int64_t) (((int64_t)(si1)) * ((int64_t)(si2))));})

#define safe_mod_func_int64_t_s_s(_si1,_si2) \
  ({ int64_t si1 = (_si1); int64_t si2 = (_si2) ; \
  ((((int64_t)(si2)) == ((int64_t)0)) || ((((int64_t)(si1)) == (INT64_MIN)) && (((int64_t)(si2)) == ((int64_t)-1)))) \
  ? ((int64_t)(si1)) \
  : (int64_t) (((int64_t)(si1)) % ((int64_t)(si2)));})

#define safe_div_func_int64_t_s_s(_si1,_si2) \
  ({ int64_t si1 = (_si1); int64_t si2 = (_si2) ; \
  ((((int64_t)(si2)) == ((int64_t)0)) || ((((int64_t)(si1)) == (INT64_MIN)) && (((int64_t)(si2)) == ((int64_t)-1)))) \
  ? ((int64_t)(si1)) \
  : (int64_t) (((int64_t)(si1)) / ((int64_t)(si2)));})

#define safe_lshift_func_int64_t_s_s(_left,_right) \
  ({ int64_t left = (_left); int right = (_right) ; \
   ((((int64_t)(left)) < ((int64_t)0)) \
  || (((int)(right)) < 0) \
  || (((int)(right)) >= 32) \
  || (((int64_t)(left)) > ((INT64_MAX) >> ((int)(right))))) \
  ? ((int64_t)(left)) \
  : (int64_t) (((int64_t)(left)) << ((int)(right)));})

#define safe_lshift_func_int64_t_s_u(_left,_right) \
  ({ int64_t left = (_left); unsigned int right = (_right) ; \
   ((((int64_t)(left)) < ((int64_t)0)) \
  || (((unsigned int)(right)) >= 32) \
  || (((int64_t)(left)) > ((INT64_MAX) >> ((unsigned int)(right))))) \
  ? ((int64_t)(left)) \
  : (int64_t) (((int64_t)(left)) << ((unsigned int)(right)));})

#define safe_rshift_func_int64_t_s_s(_left,_right) \
	({ int64_t left = (_left); int right = (_right) ; \
        ((((int64_t)(left)) < ((int64_t)0)) \
			 || (((int)(right)) < 0) \
			 || (((int)(right)) >= 32)) \
			? ((int64_t)(left)) \
			: (int64_t) (((int64_t)(left)) >> ((int)(right)));})

#define safe_rshift_func_int64_t_s_u(_left,_right) \
  ({ int64_t left = (_left); unsigned int right = (_right) ; \
   ((((int64_t)(left)) < ((int64_t)0)) \
			 || (((unsigned int)(right)) >= 32)) \
			? ((int64_t)(left)) \
			: (int64_t) (((int64_t)(left)) >> ((unsigned int)(right)));})








#define safe_unary_minus_func_uint8_t_u(_ui) \
  ({ uint8_t ui = (_ui); (uint8_t) (-((uint8_t)(ui)));})

#define safe_add_func_uint8_t_u_u(_ui1,_ui2) \
  ({ uint8_t ui1 = (_ui1); uint8_t ui2 = (_ui2) ; \
  (uint8_t) (((uint8_t)(ui1)) + ((uint8_t)(ui2)));})

#define safe_sub_func_uint8_t_u_u(_ui1,_ui2) \
  ({ uint8_t ui1 = (_ui1); uint8_t ui2 = (_ui2) ; (uint8_t) (((uint8_t)(ui1)) - ((uint8_t)(ui2)));})

#define safe_mul_func_uint8_t_u_u(_ui1,_ui2) \
  ({ uint8_t ui1 = (_ui1); uint8_t ui2 = (_ui2) ; ((uint8_t) (((unsigned int)(ui1)) * ((unsigned int)(ui2))));})

#define safe_mod_func_uint8_t_u_u(_ui1,_ui2) \
	({ uint8_t ui1 = (_ui1); uint8_t ui2 = (_ui2) ; \
         (((uint8_t)(ui2)) == ((uint8_t)0)) \
			? ((uint8_t)(ui1)) \
			: (uint8_t) (((uint8_t)(ui1)) % ((uint8_t)(ui2)));})

#define safe_div_func_uint8_t_u_u(_ui1,_ui2) \
	        ({ uint8_t ui1 = (_ui1); uint8_t ui2 = (_ui2) ; \
                 (((uint8_t)(ui2)) == ((uint8_t)0)) \
			? ((uint8_t)(ui1)) \
			: (uint8_t) (((uint8_t)(ui1)) / ((uint8_t)(ui2)));})

#define safe_lshift_func_uint8_t_u_s(_left,_right) \
	({ uint8_t left = (_left); int right = (_right) ; \
          ((((int)(right)) < 0) \
			 || (((int)(right)) >= 32) \
			 || (((uint8_t)(left)) > ((UINT8_MAX) >> ((int)(right))))) \
			? ((uint8_t)(left)) \
			: (uint8_t) (((uint8_t)(left)) << ((int)(right)));})

#define safe_lshift_func_uint8_t_u_u(_left,_right) \
	 ({ uint8_t left = (_left); unsigned int right = (_right) ; \
           ((((unsigned int)(right)) >= 32) \
			 || (((uint8_t)(left)) > ((UINT8_MAX) >> ((unsigned int)(right))))) \
			? ((uint8_t)(left)) \
			: (uint8_t) (((uint8_t)(left)) << ((unsigned int)(right)));})

#define safe_rshift_func_uint8_t_u_s(_left,_right) \
	({ uint8_t left = (_left); int right = (_right) ; \
          ((((int)(right)) < 0) \
			 || (((int)(right)) >= 32)) \
			? ((uint8_t)(left)) \
			: (uint8_t) (((uint8_t)(left)) >> ((int)(right)));})

#define safe_rshift_func_uint8_t_u_u(_left,_right) \
	({ uint8_t left = (_left); unsigned int right = (_right) ; \
                 (((unsigned int)(right)) >= 32) \
			 ? ((uint8_t)(left)) \
			 : (uint8_t) (((uint8_t)(left)) >> ((unsigned int)(right)));})




#define safe_unary_minus_func_uint16_t_u(_ui) \
  ({ uint16_t ui = (_ui); (uint16_t) (-((uint16_t)(ui)));})

#define safe_add_func_uint16_t_u_u(_ui1,_ui2) \
  ({ uint16_t ui1 = (_ui1); uint16_t ui2 = (_ui2) ; \
  (uint16_t) (((uint16_t)(ui1)) + ((uint16_t)(ui2)));})

#define safe_sub_func_uint16_t_u_u(_ui1,_ui2) \
  ({ uint16_t ui1 = (_ui1); uint16_t ui2 = (_ui2) ; (uint16_t) (((uint16_t)(ui1)) - ((uint16_t)(ui2)));})

#define safe_mul_func_uint16_t_u_u(_ui1,_ui2) \
  ({ uint16_t ui1 = (_ui1); uint16_t ui2 = (_ui2) ; ((uint16_t) (((unsigned int)(ui1)) * ((unsigned int)(ui2))));})

#define safe_mod_func_uint16_t_u_u(_ui1,_ui2) \
	({ uint16_t ui1 = (_ui1); uint16_t ui2 = (_ui2) ; \
         (((uint16_t)(ui2)) == ((uint16_t)0)) \
			? ((uint16_t)(ui1)) \
			: (uint16_t) (((uint16_t)(ui1)) % ((uint16_t)(ui2)));})

#define safe_div_func_uint16_t_u_u(_ui1,_ui2) \
	        ({ uint16_t ui1 = (_ui1); uint16_t ui2 = (_ui2) ; \
                 (((uint16_t)(ui2)) == ((uint16_t)0)) \
			? ((uint16_t)(ui1)) \
			: (uint16_t) (((uint16_t)(ui1)) / ((uint16_t)(ui2)));})

#define safe_lshift_func_uint16_t_u_s(_left,_right) \
	({ uint16_t left = (_left); int right = (_right) ; \
          ((((int)(right)) < 0) \
			 || (((int)(right)) >= 32) \
			 || (((uint16_t)(left)) > ((UINT16_MAX) >> ((int)(right))))) \
			? ((uint16_t)(left)) \
			: (uint16_t) (((uint16_t)(left)) << ((int)(right)));})

#define safe_lshift_func_uint16_t_u_u(_left,_right) \
	 ({ uint16_t left = (_left); unsigned int right = (_right) ; \
           ((((unsigned int)(right)) >= 32) \
			 || (((uint16_t)(left)) > ((UINT16_MAX) >> ((unsigned int)(right))))) \
			? ((uint16_t)(left)) \
			: (uint16_t) (((uint16_t)(left)) << ((unsigned int)(right)));})

#define safe_rshift_func_uint16_t_u_s(_left,_right) \
	({ uint16_t left = (_left); int right = (_right) ; \
          ((((int)(right)) < 0) \
			 || (((int)(right)) >= 32)) \
			? ((uint16_t)(left)) \
			: (uint16_t) (((uint16_t)(left)) >> ((int)(right)));})

#define safe_rshift_func_uint16_t_u_u(_left,_right) \
	({ uint16_t left = (_left); unsigned int right = (_right) ; \
                 (((unsigned int)(right)) >= 32) \
			 ? ((uint16_t)(left)) \
			 : (uint16_t) (((uint16_t)(left)) >> ((unsigned int)(right)));})




#define safe_unary_minus_func_uint32_t_u(_ui) \
  ({ uint32_t ui = (_ui); (uint32_t) (-((uint32_t)(ui)));})

#define safe_add_func_uint32_t_u_u(_ui1,_ui2) \
  ({ uint32_t ui1 = (_ui1); uint32_t ui2 = (_ui2) ; \
  (uint32_t) (((uint32_t)(ui1)) + ((uint32_t)(ui2)));})

#define safe_sub_func_uint32_t_u_u(_ui1,_ui2) \
  ({ uint32_t ui1 = (_ui1); uint32_t ui2 = (_ui2) ; (uint32_t) (((uint32_t)(ui1)) - ((uint32_t)(ui2)));})

#define safe_mul_func_uint32_t_u_u(_ui1,_ui2) \
  ({ uint32_t ui1 = (_ui1); uint32_t ui2 = (_ui2) ; ((uint32_t) (((unsigned int)(ui1)) * ((unsigned int)(ui2))));})

#define safe_mod_func_uint32_t_u_u(_ui1,_ui2) \
	({ uint32_t ui1 = (_ui1); uint32_t ui2 = (_ui2) ; \
         (((uint32_t)(ui2)) == ((uint32_t)0)) \
			? ((uint32_t)(ui1)) \
			: (uint32_t) (((uint32_t)(ui1)) % ((uint32_t)(ui2)));})

#define safe_div_func_uint32_t_u_u(_ui1,_ui2) \
	        ({ uint32_t ui1 = (_ui1); uint32_t ui2 = (_ui2) ; \
                 (((uint32_t)(ui2)) == ((uint32_t)0)) \
			? ((uint32_t)(ui1)) \
			: (uint32_t) (((uint32_t)(ui1)) / ((uint32_t)(ui2)));})

#define safe_lshift_func_uint32_t_u_s(_left,_right) \
	({ uint32_t left = (_left); int right = (_right) ; \
          ((((int)(right)) < 0) \
			 || (((int)(right)) >= 32) \
			 || (((uint32_t)(left)) > ((UINT32_MAX) >> ((int)(right))))) \
			? ((uint32_t)(left)) \
			: (uint32_t) (((uint32_t)(left)) << ((int)(right)));})

#define safe_lshift_func_uint32_t_u_u(_left,_right) \
	 ({ uint32_t left = (_left); unsigned int right = (_right) ; \
           ((((unsigned int)(right)) >= 32) \
			 || (((uint32_t)(left)) > ((UINT32_MAX) >> ((unsigned int)(right))))) \
			? ((uint32_t)(left)) \
			: (uint32_t) (((uint32_t)(left)) << ((unsigned int)(right)));})

#define safe_rshift_func_uint32_t_u_s(_left,_right) \
	({ uint32_t left = (_left); int right = (_right) ; \
          ((((int)(right)) < 0) \
			 || (((int)(right)) >= 32)) \
			? ((uint32_t)(left)) \
			: (uint32_t) (((uint32_t)(left)) >> ((int)(right)));})

#define safe_rshift_func_uint32_t_u_u(_left,_right) \
	({ uint32_t left = (_left); unsigned int right = (_right) ; \
                 (((unsigned int)(right)) >= 32) \
			 ? ((uint32_t)(left)) \
			 : (uint32_t) (((uint32_t)(left)) >> ((unsigned int)(right)));})




#define safe_unary_minus_func_uint64_t_u(_ui) \
  ({ uint64_t ui = (_ui); (uint64_t) (-((uint64_t)(ui)));})

#define safe_add_func_uint64_t_u_u(_ui1,_ui2) \
  ({ uint64_t ui1 = (_ui1); uint64_t ui2 = (_ui2) ; \
  (uint64_t) (((uint64_t)(ui1)) + ((uint64_t)(ui2)));})

#define safe_sub_func_uint64_t_u_u(_ui1,_ui2) \
  ({ uint64_t ui1 = (_ui1); uint64_t ui2 = (_ui2) ; (uint64_t) (((uint64_t)(ui1)) - ((uint64_t)(ui2)));})

#define safe_mul_func_uint64_t_u_u(_ui1,_ui2) \
  ({ uint64_t ui1 = (_ui1); uint64_t ui2 = (_ui2) ; ((uint64_t) (((unsigned long long)(ui1)) * ((unsigned long long)(ui2))));})

#define safe_mod_func_uint64_t_u_u(_ui1,_ui2) \
	({ uint64_t ui1 = (_ui1); uint64_t ui2 = (_ui2) ; \
         (((uint64_t)(ui2)) == ((uint64_t)0)) \
			? ((uint64_t)(ui1)) \
			: (uint64_t) (((uint64_t)(ui1)) % ((uint64_t)(ui2)));})

#define safe_div_func_uint64_t_u_u(_ui1,_ui2) \
	        ({ uint64_t ui1 = (_ui1); uint64_t ui2 = (_ui2) ; \
                 (((uint64_t)(ui2)) == ((uint64_t)0)) \
			? ((uint64_t)(ui1)) \
			: (uint64_t) (((uint64_t)(ui1)) / ((uint64_t)(ui2)));})

#define safe_lshift_func_uint64_t_u_s(_left,_right) \
	({ uint64_t left = (_left); int right = (_right) ; \
          ((((int)(right)) < 0) \
			 || (((int)(right)) >= 32) \
			 || (((uint64_t)(left)) > ((UINT64_MAX) >> ((int)(right))))) \
			? ((uint64_t)(left)) \
			: (uint64_t) (((uint64_t)(left)) << ((int)(right)));})

#define safe_lshift_func_uint64_t_u_u(_left,_right) \
	 ({ uint64_t left = (_left); unsigned int right = (_right) ; \
           ((((unsigned int)(right)) >= 32) \
			 || (((uint64_t)(left)) > ((UINT64_MAX) >> ((unsigned int)(right))))) \
			? ((uint64_t)(left)) \
			: (uint64_t) (((uint64_t)(left)) << ((unsigned int)(right)));})

#define safe_rshift_func_uint64_t_u_s(_left,_right) \
	({ uint64_t left = (_left); int right = (_right) ; \
          ((((int)(right)) < 0) \
			 || (((int)(right)) >= 32)) \
			? ((uint64_t)(left)) \
			: (uint64_t) (((uint64_t)(left)) >> ((int)(right)));})

#define safe_rshift_func_uint64_t_u_u(_left,_right) \
	({ uint64_t left = (_left); unsigned int right = (_right) ; \
                 (((unsigned int)(right)) >= 32) \
			 ? ((uint64_t)(left)) \
			 : (uint64_t) (((uint64_t)(left)) >> ((unsigned int)(right)));})



#endif

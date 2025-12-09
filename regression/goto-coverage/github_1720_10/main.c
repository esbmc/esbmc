//This program is zodiac.c
#include<stdio.h>
#include <assert.h>
#include <stdbool.h>
int kappa;
int main()
{
kappa=0;
int  date,ch;

printf("ENTER DATE OF BIRTH AND THEN MONTH NO. codewithc");
//scanf("%d %d",&date,&ch);
date = nondet_int();
_ESBMC_input("date",date);   
ch = nondet_int();
_ESBMC_input("ch",ch);   
/*klee_make_symbolic(&date, sizeof(int), "date");*/
/*klee_make_symbolic(&ch, sizeof(int), "ch");*/

if (((date>=21)  && (ch==3)) || ((date<=20 ) && (ch==4 )))
printf("YOU ARE A ARIES AND YOUR BIRTHSTONE IS BLOODSTONE");
if (((date>=21)  && (ch==4)) || ((date<=21 ) && (ch==5 )))
printf("YOU ARE A TAURUS AND YOUR BIRTHSTONE IS SAPPHIRE");
if (((date>=22)  && (ch==5)) || ((date<=21 ) && (ch==6 )))
printf("YOU ARE A GEMINI AND YOUR BIRTHSTONE IS AGATE");
if (((date>=22)  && (ch==6)) || ((date<=22 ) && (ch==7 )))
printf("YOU ARE A CANCER AND YOUR BIRTHSTONE IS EMERALD");
if (((date>=23)  && (ch==7)) || ((date<=22 ) && (ch==8 )))
printf("YOU ARE A LEO AND YOUR BIRTHSTONE IS ONYX");
if (((date>=23)  && (ch==8)) || ((date<=22 ) && (ch==9 )))
printf("YOU ARE A VIRGO AND YOUR BIRTHSTONE IS CARNELIAN");
if (((date>=23)  && (ch==9)) || ((date<=23 ) && (ch==10 )))
printf("YOU ARE A LIBRA AND YOUR BIRTHSTONE IS CHRYSOLITE");
if (((date>=24)  && (ch==10)) || ((date<=21 ) && (ch==11 )))
printf("YOU ARE A SCORPIO AND YOUR BIRTHSTONE IS BERYL");
if (((date>=22)  && (ch==11)) || ((date<=21 ) && (ch==12 )))
printf("YOU ARE A SAGITTARIUS AND YOUR BIRTHSTONE IS TOPAZ");
if (((date>=22)  && (ch==12)) || ((date<=21 ) && (ch==1 )))
printf("YOU ARE A CAPRICORN AND YOUR BIRTHSTONE IS RUBY");
if (((date>=22)  && (ch==1)) || ((date<=18 ) && (ch==2 )))
printf("YOU ARE A AQUARIUS AND YOUR BIRTHSTONE IS GARNET");
if (((date>=19)  && (ch==2)) || ((date<=20 ) && (ch==3 )))
printf("YOU ARE A PISCES AND YOUR BIRTHSTONE IS AMETHYST");


return 0;
}

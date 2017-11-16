/*************************/
/*************************/
// INPUT.H
/*************************/
/*************************/


#include<stdio.h>
//#include<conio.h>
struct list{
char alph;
int freq;
} ;
struct list a[256];
int n;
void input()
{
FILE *fin,*fout;
char filein[20], fileout[20], ch;
int i,k,f;

printf("enter the filename of from which the data is to be read::\n");
scanf("%s", filein);
fin=fopen(filein,"r");
for(i=0;i<256;i++)
	a[i].freq=0;

	  while((ch=fgetc(fin))!=EOF)
	  {
	     a[ch].alph=ch;
	     a[n].freq++;
	 }
fclose(fin);
}

/*************************/
/*************************/
// message.txt
/*************************/
/*************************/

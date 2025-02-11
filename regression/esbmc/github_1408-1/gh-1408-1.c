struct InnerStruct
{
   char innerValue[10];
};

int main()
{
   struct InnerStruct array[2];
   strcpy(array[0].innerValue, "20");
   char *ptr;
   assert(strtol(array[0].innerValue, ptr, 10) == 40);
}

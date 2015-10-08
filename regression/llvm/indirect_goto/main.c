
int main(){
    int value  = 2;
    const void *labels[] = {&&val_0, &&val_1, &&val_2};
    goto *labels[value];
    val_0:
        printf("The value is 0\n");
        goto end;
    val_1:
        printf("The value is 1\n");
        goto end;
    val_2:
        printf("The value is 2\n");
        assert(0);
        goto end;
    end:

    return(0);
}

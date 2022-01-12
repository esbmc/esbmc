fun sort(arr: IntArray, N: Int)
{
    for(i in 0..(N-1))
    {
        var min = i;
        for(j in i..(N-1))
        {
            if(arr[j] < arr[min])
                min = j;
        }
        val tmp = arr[i]
        arr[i] = arr[min]
        arr[min] = tmp
    }        
}

fun is_sorted(arr: IntArray, N: Int): Boolean
{
    if(N < 2) return true;
    for(i in 1..(N-1))
    {
        if(arr[i] < arr[i-1]) return false;
    }
    return true;
}

fun initialize_array(arr: IntArray, N: Int)
{
    for(i in 0..(N-1))
      arr[i] = N - i;
}

fun main() {
    val arr = IntArray(20);
    initialize_array(arr, 20);

    sort(arr, 20);

    assert(is_sorted(arr,20));
}
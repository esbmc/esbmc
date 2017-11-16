#define ELEMEN 3

typedef unsigned int uData8;
typedef int Data8;

uData8 showAverage_v1(Data8 *sensorData)
{  

	Data8 i=0, sensorValue=0, numElements=0, aux=0;

	for(i=0; i<ELEMEN; i++)
    {
		if ( sensorData[i]!=0 )
        {
			sensorValue = sensorValue + sensorData[i];
			++numElements;
		}
	}
	if (numElements!=0)
    {
		aux = sensorValue/numElements;
	}

	return aux;
}

int main()
{
  int i, sensorData[ELEMEN];

  for(i=0; i<ELEMEN; i++)  
    sensorData[i]=2;

  assert(0==((sensorData[0]-sensorData[1]-sensorData[2])/3));
  assert(2==((sensorData[0]+sensorData[1]+sensorData[2])/3));
  assert(2==((sensorData[0]*sensorData[1]*sensorData[2])/3));
}

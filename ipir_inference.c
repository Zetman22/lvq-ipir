#include <string.h>
#include <stdio.h>
#include "sdkconfig.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_adc/adc_continuous.h"
#include "esp_adc_cal.h"
#include "driver/adc.h"
#include<math.h>
#include<stdio.h>
#include<string.h>
#include <stdlib.h>

static const char *TAG = "ADC EXAMPLE";

// define the pin for the led and the adc 
#define ADC_pin ADC_CHANNEL_6
#define LED_pin 2
static esp_adc_cal_characteristics_t adc1_chars;

// function to find the min of the vector with 200 values
float mina(float arr[200]){
	float minn = arr[0];
	for(int i=0;i<200;i++){
		if(minn>arr[i]){
			minn = arr[i];
		}
	}
	return minn;
}

// function to find the max of the vector with 200 values
float maxa(float arr[200]){
	float maxx = arr[0];
	for(int i=0;i<200;i++){
		if(maxx<arr[i]){
			maxx = arr[i];
		}
	}
	return maxx;
}

// function to find the index of the min value
float min_index(float arr[200]){
	float minn = arr[0];
	float index = 0;
	for(int i=0;i<200;i++){
		if(minn>arr[i]){
			minn = arr[i];
			index = i;
		}
	}
	return index;
}

// function to find the index of the max value
float max_index(float arr[200]){
	float maxx = arr[0];
	float index = 0;
	for(int i=0;i<200;i++){
		if(maxx<arr[i]){
			maxx = arr[i];
			index = i;
		}
	}
	return index;
}

// function to calculate the average value of the vector with 200 values
float mean(float arr[200]){
	float sum = 0;
	for(int i=0;i<200;i++){
		sum = sum+arr[i];
	}
	return sum/200;
}

// function to calculate the standard deviation of the vector
float calculateSD(float data[],int n) {
    float sum = 0.0, mean, SD = 0.0;
    int i;
    for (i = 0; i < n; ++i) {
        sum += data[i];
    }
    mean = sum / n;
    for (i = 0; i < n; ++i) {
        SD += pow(data[i] - mean, 2);
    }
    return sqrt(SD / n);
}

// function to find the most suitable codebook
float find_best_codebook(float *codebooks[6],float *matrix[5],float array[]){
	float dis[10];
	float minus[5];
	for(int i=0;i<10;i++){
		dis[i]=0;
		for(int j=0;j<5;j++){
			minus[j]=array[j]-codebooks[i][j];
		}
		for(int j=0;j<5;j++){
			float s = 0;
			for(int x=0;x<5;x++){
				s = s+ matrix[j][x]*minus[x];
			}
			dis[i]+= s*minus[j];
		}
		ESP_LOGI(TAG, "check: %d %f",i, dis[i]);
	}
	float minn = dis[0];
	float index = 0;
	for(int i=0;i<10;i++){
		if(minn>dis[i]){
			minn = dis[i];
			index = i;
		}
	}
	ESP_LOGI(TAG, "check: %f", index);
	return codebooks[(int)index][5];
}
// function to extract feature and make decision on the presence of human
int function(float **codebooks, float **matrix){
	int k=0;
	float array[5];
	float indexmin;
	float indexmax;
	float arr[200];
	float norm;
	float std;
    int voltage;

    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 0, &adc1_chars);
    ESP_ERROR_CHECK(adc1_config_width(ADC_WIDTH_BIT_12));
    ESP_ERROR_CHECK(adc1_config_channel_atten(ADC_pin,ADC_ATTEN_DB_11));
    while(1)
    {
        voltage = adc1_get_raw(ADC_pin);
        ESP_LOGI(TAG, "ADC1: %d", voltage);
        arr[k]=voltage;
        k+=1;
        if(k==200){
			array[0] = mina(arr);
			indexmin = min_index(arr);
			array[1] = maxa(arr);
			indexmax = max_index(arr);
			array[2] = mean(arr);
			array[3] = calculateSD(arr,200);
			array[4] = 100*(array[1]-array[0])/(indexmax-indexmin);
			norm = (array[0]+array[1]+array[2]+array[3]+array[4])/5;
			std = calculateSD(array,5);
			for(int j=0;j<5;j++){
				array[j]=(array[j]-norm)/std;
			}
			ESP_LOGI(TAG, "array0: %f", array[0]);
			ESP_LOGI(TAG, "array1: %f", array[1]);
			ESP_LOGI(TAG, "array2: %f", array[2]);
			ESP_LOGI(TAG, "array3: %f", array[3]);
			ESP_LOGI(TAG, "array4: %f", array[4]);
			float check = find_best_codebook(codebooks , matrix,array);
            if(check==1.0){
                gpio_set_level(LED_pin,0);
            }
            else{
                gpio_set_level(LED_pin,1);
            }
			for(int i=0;i<150;i++){
				arr[i]=arr[i+50];
			}
			k=150;
		}
        vTaskDelay(10/portTICK_PERIOD_MS);
    }
	return 0;
}

void app_main(void){
    gpio_set_direction(LED_pin, GPIO_MODE_OUTPUT);

	//Data Setter, these parameter were trained by the GMLVQ
	float codebook1[6] = {-0.158,0.070,1.622,-2.271,-1.259,0.0};
	float codebook2[6] = {-0.322,0.285,2.110,-0.909,-1.585,0.0};
	float codebook3[6] = {0.751,0.941,2.029,-0.386,-0.758,0.0};
	float codebook4[6] = {0.400,0.233,1.668,-1.302,-1.744,0.0};
	float codebook5[6] = {0.783,0.848,0.805,-1.269,-1.219,0.0};
	float codebook6[6] = {0.433,0.386,-0.239,1.342,-0.270,1.0};
	float codebook7[6] = {0.449,1.096,0.840,-0.444,-1.640,1.0};
	float codebook8[6] = {-0.552,0.345,1.388,-0.520,-0.362,1.0};
	float codebook9[6] = {0.378,1.255,0.934,-1.641,-0.413,1.0};
	float codebook10[6] = {-1.271,1.014,-0.132,-0.959,0.969,1.0};

	float matrix1[5]={0.607,-0.106,0.465,-0.093,0.038};
	float matrix2[5]={-0.106,0.018,-0.081,0.016,-0.007};
	float matrix3[5]={0.465,-0.081,0.357,-0.071,0.029};
	float matrix4[5]={-0.093,0.016,-0.071,0.014,-0.006};
	float matrix5[5]={0.038,-0.007,0.029,-0.006,0.002};

	//Codebook and matrix init
	float *codebooks[10];
	for(int i=0;i<10;i++){
		codebooks[i] = (float*)malloc(sizeof(float)*6);
	}

	float *matrix[5];
	for(int i=0;i<5;i++){
		matrix[i] = (float*)malloc(sizeof(float)*5);
	}

	codebooks[0] = codebook1;
	codebooks[1] = codebook2;
	codebooks[2] = codebook3;
	codebooks[3] = codebook4;
	codebooks[4] = codebook5;
	codebooks[5] = codebook6;
	codebooks[6] = codebook7;
	codebooks[7] = codebook8;
	codebooks[8] = codebook9;
	codebooks[9] = codebook10;

	matrix[0] = matrix1;
	matrix[1] = matrix2;
	matrix[2] = matrix3;
	matrix[3] = matrix4;
	matrix[4] = matrix5;

	// run the function
	function(codebooks, matrix);
}


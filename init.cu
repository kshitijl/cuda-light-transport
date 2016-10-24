#include <stdio.h>
#include <cuda.h>

__global__ void draw_circle(char *image, int width, int height, int c_x, int c_y, int r) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height) {
    int dx = x-c_x, dy = y-c_y;
    int idx = 3*(y*width + x);
    int answer = 0;
    
    if(dx*dx + dy*dy <= r*r) {
      answer = 128;
    }
    image[idx] = answer;
    image[idx+1] = 0;
    image[idx+2] = 0;
  }
}

void write_ppm(const char * filename, int width, int height, char * data) {
  FILE *fp = fopen(filename, "wb");
  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  fwrite(data, width*height*3, 1, fp);
  fclose(fp);
}

int main(void) {
  int w=1024,h=768;

  char *a_h, *a_d;

  int size = w*h*3*sizeof(char);
  cudaMalloc((void **)&a_d, size);
  
  dim3 dimBlock(32,32);
  dim3 dimGrid(w/32 + (w%32 > 0), h/32 + (h%32 > 0));
  draw_circle<<<dimGrid, dimBlock>>>(a_d, w, h, w/2, h/2, h/3);

  a_h = (char *)malloc(size);

  cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  write_ppm("circle.ppm", w, h, a_h);
  
  cudaFree(a_d);
  free(a_h);

  return 0;
}

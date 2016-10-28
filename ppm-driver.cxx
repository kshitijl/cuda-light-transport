#include <string>

int main(void) {
  int w=2000,h=2000;

  char *a_h, *a_d;

  int size = w*h*3*sizeof(char);
  cudaMalloc((void **)&a_d, size);
  a_h = (char *)malloc(size);  

  for(int ii = 0; ii < 100; ++ii) {
    dim3 dimBlock(32,32);
    dim3 dimGrid(w/32 + (w%32 > 0), h/32 + (h%32 > 0));
    draw_circle<<<dimGrid, dimBlock>>>(a_d, w, h, float(ii)/50);
    cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    write_ppm(("circle." + std::to_string(ii)+ ".ppm").c_str(), w, h, a_h);
  }
  
  cudaFree(a_d);
  free(a_h);

  return 0;
}

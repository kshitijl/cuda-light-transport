#include <stdio.h>
#include <string>

#include <cuda.h>
#include "helper_math.h"

const float tiny = 1e-5;

struct ray_t {
  float3 origin;
  float3 direction;
};  

struct intersection_result_t {
  float distance;
  float3 intersection_point;
  float3 surface_normal;
};

struct sphere_t {
  float3 center;
  float radius;
  
  __device__ intersection_result_t intersect(const ray_t ray) const {
    float3 l = ray.origin - center;
    float b = 2*dot(ray.direction, l);
    float c = dot(l,l) - radius*radius;

    float discr = b*b - 4*c;
    if(discr < tiny)
      return intersection_result_t{-1};
    
    float q = (b > 0) ? 
      -0.5 * (b + sqrt(discr)) : 
      -0.5 * (b - sqrt(discr));
    float x0 = fmaxf(q, -tiny); 
    float x1 = fmaxf(c / q, -tiny);

    float closest = fminf(x0, x1);

    float3 point = ray.origin + closest*ray.direction;
    float3 normal = point - center;
    
    return intersection_result_t{closest,
        point,
        normal/sqrt(dot(normal,normal))};
  }  
};

const __device__ float3 eye{0,0,0};
const __device__ float3 light{-5, -5, 0.2};

const __device__ int nspheres = 2;

__global__ void draw_circle(char *image, int width, int height, float t) {
  const sphere_t spheres[] = {sphere_t{float3{0.1, 0.1, 3}, 1.1},
                              sphere_t{float3{-0.8+t, -0.5, 1.8}, 0.25}};
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height) {
    float3 direction = float3{float(x)/width-float(0.5), float(y)/float(height)-float(0.5), 1} - eye;
    
    ray_t ray{eye, direction/sqrt(dot(direction,direction))};

    int answer = 10;

    intersection_result_t best{1e10};
    
    for(int ii = 0; ii < nspheres; ++ii) {
      auto result = spheres[ii].intersect(ray);
      if(result.distance > 0) {
        if(result.distance < best.distance)
          best = result;
      }
    }
    
    if(best.distance < 1e9) {
      float3 new_dir = light - best.intersection_point;
      float dist_to_light = sqrt(dot(new_dir,new_dir));            
      ray_t shadow_ray{best.intersection_point, new_dir/dist_to_light};
      float shading = fmaxf(0, dot(shadow_ray.direction, best.surface_normal));
      
      for(int ii = 0; ii < nspheres; ++ii) {
        auto result = spheres[ii].intersect(shadow_ray);
        if(result.distance > 0) {
          if(result.distance < dist_to_light)
            shading = 0;
        }
      }
      
      answer = 255*shading;
    }
    int idx = 3*(y*width + x);
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
  int w=4000,h=4000;

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

#include <stdio.h>
#include <string>
#include <assert.h>

#include <cuda.h>
#include "helper_math.h"

#include <curand_kernel.h>
#include <curand_normal.h>

namespace gpu_random {
__device__ float2 uniforms(uint4 counter, uint2 key) {
  float2 answer;
  
  uint4 result = curand_Philox4x32_10(counter, key);

  answer.x = _curand_uniform(result.x);
  answer.y = _curand_uniform(result.y);

  return answer;
}
}

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
  float3 emittance;
  
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

__device__ float3 sample_cosine_weighted_direction(float3 normal, float r1, float r2) {
  float theta = acosf(sqrt(1.0-r1));
  float phi = 2.0 * 3.141592653 * r2;

  float xs = sinf(theta) * cosf(phi);
  float ys = cosf(theta);
  float zs = sinf(theta)*sinf(phi);

  float3 y = normal, h = normal;
  if(fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
    h.x = 1.0;
  else if (fabs(h.y)<=fabs(h.x) && fabs(h.y) <= fabs(h.z))
    h.y = 1.0;
  else
    h.z = 1.0;

  float3 x = cross(h,y);
  x /= sqrt(dot(x,x));
  float3 z = cross(x, y);
  z /= sqrt(dot(z,z));

  float3 dir = xs*x + ys*y + zs*z;
  return dir/sqrt(dot(dir,dir));
}

const __device__ float3 eye{0,0,0};

__global__ void draw_circle(float3 *image, int width, int height, int nsamples) {
  const int nspheres = 3;
  const sphere_t spheres[] = { sphere_t{float3{-5, -5, 0.2f}, 2.2, float3{1.0,0.0,0.0}},
                               sphere_t{float3{-0.5, -0.5, 1.8}, 0.25, float3{0.0,0.0,0.0}},                               
                               sphere_t{float3{0.1, 0.1, 3}, 1.1, float3{0.0,0.0,0.0}}
                               
                             };
  
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint sample = blockIdx.z * blockDim.z + threadIdx.z;

  if(x < width && y < height) {
    float3 direction = float3{float(x)/width-float(0.5), float(y)/float(height)-float(0.5), 1} - eye;
    
    ray_t ray{eye, direction/sqrt(dot(direction,direction))};

    float3 accumulator{0,0,0}, mask{1,1,1};

    for(uint bounces = 0; bounces < 10; ++bounces) {
      intersection_result_t best{1e10};
      int best_sphere_i = 0;
    
      for(int ii = 0; ii < nspheres; ++ii) {
        auto result = spheres[ii].intersect(ray);
        if(result.distance > 0) {
          if(result.distance < best.distance) {
            best = result;
            best_sphere_i = ii;
          }
        }
      }
    
      if(best.distance < 1e9) {
        float2 random_uniforms = gpu_random::uniforms(uint4{x,y,sample,0},
                                                      uint2{bounces,0});
        float3 new_dir = sample_cosine_weighted_direction(best.surface_normal,
                                                          random_uniforms.x,
                                                          random_uniforms.y);
        ray = ray_t{best.intersection_point, new_dir};

        accumulator += mask*spheres[best_sphere_i].emittance/(best.distance*best.distance);
        mask *= 0.9;
      }
      else {
        break;
      }
    }

    int idx = (y*width + x)*nsamples + sample;
    image[idx] = accumulator;
  }
}

__global__ void combine_samples_device(float3* images, char* final,
                                       int w, int h, int niters) {
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < w && y < h) {
    float3 answer{0,0,0};
    
    for(int sample = 0; sample < niters; ++sample) {
      int idx = (y*w + h)*niters + sample;
      answer += images[idx];
    }

    answer /= 10;
    
    int idx = 3*(y*w + h);
    final[idx] = answer.x*255;
    final[idx+1] = answer.y*255;
    final[idx+2] = answer.z*255;
  }
}

void write_ppm(const char * filename, int width, int height, char * data) {
  FILE *fp = fopen(filename, "wb");
  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  fwrite(data, width*height*3, 1, fp);
  fclose(fp);
}

void combine_samples(float3* images, char* final, int w, int h, int niters) {
  dim3 dimBlock(32,32);
  dim3 dimGrid(w/32 + (w%32 > 0), h/32 + (h%32 > 0));
  combine_samples_device<<<dimGrid, dimBlock>>>(images, final, w, h, niters);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  int niters = std::atoi(argv[1]);
  
  int w=1000,h=1000;

  char *a_h, *a_d;
  float3 *images;

  int size = w*h*3*sizeof(char);
  cudaMalloc((void **)&a_d, size);
  cudaMalloc((void **)&images, w*h*niters*sizeof(float3));
  
  a_h = (char *)malloc(size);

  dim3 dimBlock(16,16,4);
  dim3 dimGrid(w/16 + (w%16 > 0), h/16 + (h%16 > 0),niters/4 + (niters%4>0));
  cudaMemset(images, 0, size*niters);
  draw_circle<<<dimGrid, dimBlock>>>(images, w, h, niters);

  combine_samples(images, a_d, w, h, niters);
  cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  write_ppm("circle.ppm", w, h, a_h);

  
  cudaFree(a_d);
  free(a_h);

  return 0;
}

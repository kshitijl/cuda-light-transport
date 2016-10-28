#include <stdio.h>
#include <string>
#include <assert.h>

#include <cuda.h>
#include "helper_math.h"
#include "simple-interop.hxx"

#include <memory>

#include <moderngpu/transform.hxx>
#include <moderngpu/memory.hxx>


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
using uchar = unsigned char;

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

__global__ void ray_trace(uchar3 *image, uint width, uint height,
                          unsigned int iteration,
                          const sphere_t* spheres, int nspheres) {
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;

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
        float2 random_uniforms = gpu_random::uniforms(uint4{x,y,iteration,0},
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

    int idx = (y*width + x);

    uchar3 prev = image[idx];

    image[idx] = uchar3{uchar(100*accumulator.x) + prev.x,
                        uchar(100*accumulator.y) + prev.y,
                        uchar(100*accumulator.z) + prev.z};
  }
}


struct raytracer_t {
  mgpu::standard_context_t context;
  mgpu::mem_t<sphere_t> geometry;
  
  raytracer_t() {
  }
  
  void draw_scene(uchar3 *image_out, uint width, uint height) {
    float tt = float(clock())/CLOCKS_PER_SEC;

    geometry = mgpu::to_mem(std::vector<sphere_t>{
        sphere_t{float3{-5, -5, 0.2f}, 5.0, float3{0.2,0.0,0.0}},
          sphere_t{float3{sin(tt), -0.5, 1.8}, 0.25, float3{0.0,0.0,0.0}},                               
            sphere_t{float3{0.1, 0.1, 3}, 1.1, float3{0.0,0.0,0.0}}
            
      },
      context);
  
    dim3 grid_dim{width/32 + (width % 32 > 0), height/32 + (height % 32 > 0)};
    dim3 block_dim{32,32};
    cudaMemset(image_out, 0, width*height*sizeof(uchar3));

    for(int ii = 0; ii < 100; ++ii) {
      ray_trace<<<grid_dim, block_dim>>>(image_out, width, height,
                                         ii,
                                         geometry.data(),
                                         geometry.size());
    }
  }
};

struct global_state_t {
  simple_interop_t interop;
  raytracer_t raytracer;
};

global_state_t* global_state;

void render() {
  unsigned int width = global_state->interop.width, height = global_state->interop.height;

  global_state->interop.cuda_render([=](uchar3 *output) {
      global_state->raytracer.draw_scene(output, width, height);
    });

  
  glFlush();
  glutPostRedisplay();  
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowSize(2000, 2000);
  glutCreateWindow("Render with CUDA");

  global_state_t main_global_state{simple_interop_t(1000,1000)};
  global_state = &main_global_state;
  
  glutDisplayFunc(render);
  glutMainLoop();

  return 0;
}

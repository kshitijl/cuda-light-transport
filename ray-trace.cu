#include <stdio.h>
#include <string>
#include <assert.h>

#include <cuda.h>
#include "helper_math.h"
#include "simple-interop.hxx"

#include <memory>

#include <moderngpu/kernel_segreduce.hxx>
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
  double radius;  
  float3 center;
  float3 emittance;
  float3 color;
  
  __device__ intersection_result_t intersect(const ray_t ray) const {
    float3 op = center-ray.origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
    double t, eps=1e-4, b=dot(op, ray.direction), det=b*b - dot(op,op) + radius*radius;
    
    if (det<0) return intersection_result_t{-1}; else det=sqrt(det); 
    double closest = (t=b-det)>eps ? t : ((t=b+det)>eps ? t : -1);

    if(closest < 0)
      return intersection_result_t{-1};

    float3 point = ray.origin + closest*ray.direction;
    float3 normal = (point - center)/radius;

    if(radius < 100)
      normal = -normal;
    
    return intersection_result_t{closest,
        point,
        -normal};
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

  float3 x = normalize(cross(h,y));
  float3 z = normalize(cross(x, y));

  float3 dir = xs*x + ys*y + zs*z;
  return normalize(dir);
}

const __device__ float3 eye{50,52,295.6};

__global__ void ray_trace(float3 *image, uint width, uint height,
                          uint nsamples,
                          float time,
                          const sphere_t* spheres, int nspheres) {
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint sample = blockIdx.z * blockDim.z + threadIdx.z;

  if(x < width && y < height) {
    float3 camdir = normalize(float3{sin(time)/9,-0.042612,-1});
    float3 cx{width*0.5135f/height}, cy = normalize(cross(cx, camdir))*0.5135;
    float3 direction = cx*( (x+sin(time))/width - 0.5) +
      cy * ( (y+cos(time))/height - 0.5) + camdir;
      
    ray_t ray{eye + direction*(140), normalize(direction)};

    float3 accumulator{0,0,0}, mask{1,1,1};

    for(uint bounces = 0; bounces < 2; ++bounces) {
      intersection_result_t best{1e150};
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
    
      if(best.distance < 1e150) {
        float2 random_uniforms = gpu_random::uniforms(uint4{x,y,sample,0},
                                                      uint2{bounces,0});
        float3 new_dir = sample_cosine_weighted_direction(best.surface_normal,
                                                          random_uniforms.x,
                                                          random_uniforms.y);
        ray = ray_t{best.intersection_point, new_dir};

        accumulator += mask*spheres[best_sphere_i].emittance;
        mask *= spheres[best_sphere_i].color;
      }
      else {
        break;
      }
    }

    int idx = (y*width + x)*nsamples + sample;

    image[idx] = accumulator/nsamples;
  }
}

__device__ uchar to_uchar(float x) {
  return 255*pow(clamp(x, 0.0f, 1.0f), 1/2.2) + 0.5;
}

__global__ void float3_to_uchar3(float3* img_in, uchar3* img_out,
                                 uint width, uint height) {
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height) {
    int idx = y*width + x;    
    float3 p = img_in[idx];

    img_out[idx] = uchar3{to_uchar(p.x),
                          to_uchar(p.y),
                          to_uchar(p.z)};
  }  
}

using uint = unsigned int;

struct raytracer_t {
  mgpu::standard_context_t context;
  mgpu::mem_t<sphere_t> geometry;
  mgpu::mem_t<float3> img_buffer, accum_buffer;

  uint nsamples;
  
  raytracer_t(uint width, uint height, uint nsamples) :
    nsamples(nsamples),
    img_buffer(width*height*nsamples, context),
    accum_buffer(width*height, context) {
  }
  
  void draw_scene(uchar3 *image_out, uint width, uint height) {
    float tt = float(clock())/CLOCKS_PER_SEC;
    
    geometry = mgpu::to_mem(std::vector<sphere_t>{//Scene: radius, position, emission, color, material 
        sphere_t{1e5, float3{ 1e5+1,40.8,81.6}, float3{0,0,0},float3{.75,.25,.25}},//Left 
          sphere_t{1e5, float3{-1e5+99,40.8,81.6},float3{0,0,0},float3{.25,.25,.75}},//Rght 
            sphere_t{1e5, float3{50,40.8, 1e5},     float3{0,0,0},float3{.75,.75,.75}},//Back 
              sphere_t{1e5, float3{50,40.8,-1e5+170}, float3{0,0,0},float3{0,0,0}          },//Frnt 
                sphere_t{1e5, float3{50, 1e5, 81.6},    float3{0,0,0},float3{.75,.75,.75}},//Botm 
                  sphere_t{1e5, float3{50,-1e5+81.6,81.6},float3{0,0,0},float3{.75,.75,.75}},//Top 
                    sphere_t{16.5,float3{27,16.5,47},       float3{0,0,0},float3{1,1,1}*.999},//Mirr 
                      sphere_t{16.5,float3{73,16.5,78},       float3{0,0,0},float3{1,1,1}*.999},//Glas 
                        sphere_t{600, float3{50,681.6-.27,81.6},float3{12,12,12},  float3{0,0,0}} //Lite 
      },
      context);
    
    dim3 grid_dim{width/16 + (width % 16 > 0),
        height/16 + (height % 16 > 0),
        nsamples/4 + (nsamples % 4 > 0)};
    dim3 block_dim{16,16,4};

    ray_trace<<<grid_dim, block_dim>>>(img_buffer.data(), width, height,
                                       nsamples,
                                       tt,
                                       geometry.data(),
                                       geometry.size());

    int nsamples_local = nsamples;    
    auto img_buffer_data = img_buffer.data();
    mgpu::segreduce(img_buffer.data(), width*height*nsamples,
                    mgpu::make_load_iterator<int>([=]MGPU_DEVICE(int index) {
                        return nsamples_local*index;
                      }),
                    width*height,
                    accum_buffer.data(),
                    mgpu::plus_t<float3>(), float3{0,0,0},
                    context);

    float3_to_uchar3<<<grid_dim, block_dim>>>(accum_buffer.data(), image_out,
                                              width, height);
    
  }
};

struct global_state_t {
  simple_interop_t interop;
  raytracer_t raytracer;

  global_state_t(uint width, uint height) :
    interop(width,height),
    raytracer(width,height, 40) {}
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
  glutInitWindowSize(1920, 1080);
  glutCreateWindow("Render with CUDA");

  global_state_t main_global_state(1024,768);
  global_state = &main_global_state;
  
  glutDisplayFunc(render);
  glutMainLoop();

  return 0;
}
/*
   Ray cam(Vec(50,52,295.6), Vec(0,-0.042612,-1).norm()); // cam pos, dir 
   Vec cx=Vec(w*.5135/h), cy=(cx%cam.d).norm()*.5135, r, *c=new Vec[w*h]; 
 for (int s=0; s<samps; s++){ 
             double r1=2*erand48(Xi), dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1); 
             double r2=2*erand48(Xi), dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2); 
             Vec d = cx*( ( (sx+.5 + dx)/2 + x)/w - .5) + 
                     cy*( ( (sy+.5 + dy)/2 + y)/h - .5) + cam.d; 
             r = r + radiance(Ray(cam.o+d*140,d.norm()),0,Xi)*(1./samps); 
           } // Camera rays are pushed ^^^^^ forward to start in interior 
           c[i] = c[i] + Vec(clamp(r.x),clamp(r.y),clamp(r.z))*.25; 

*/

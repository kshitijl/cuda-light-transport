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
    double t, eps=1e-2, b=dot(op, ray.direction), det=b*b - dot(op,op) + radius*radius;
    
    if (det<0) return intersection_result_t{-1}; else det=sqrt(det); 
    double closest = (t=b-det)>eps ? t : ((t=b+det)>eps ? t : -1);

    if(closest < 0)
      return intersection_result_t{-1};

    float3 point = ray.origin + closest*ray.direction;
    float3 normal = (point - center)/radius;

    // Cornell box has us inside the spheres, so make sure to return a
    // normal in the appropriate direction: away from the incoming ray
    
    if(dot(ray.direction, normal) > 0)
      normal = -normal; 
    
    return intersection_result_t{closest,
        point+normal*eps,
        normal};
  }  
};

__device__ float3 sample_cosine_weighted_direction(float3 normal, float r1, float r2) {
  float p1 = 2*M_PI*r1, p2 = r2, p2s = sqrt(r2);
  float3 w = normal, u = normalize(cross(fabs(w.x)>.1?float3{0,1} : float3{1}, w)),
    v = cross(w,u);
  float3 d = u*cos(p1)*p2s + v * sin(p1)*p2s + w*sqrt(1-p2);
  return normalize(d);
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

    for(uint bounces = 0; bounces < 10; ++bounces) {
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

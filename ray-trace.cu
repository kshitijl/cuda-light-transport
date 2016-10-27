#include "simple-interop.hxx"
#include "helper_math.h"

#include <memory>

#include <moderngpu/transform.hxx>
#include <moderngpu/memory.hxx>

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

__global__ void ray_trace(uchar3 *image, uint width, uint height,
                          const sphere_t* spheres, int nspheres) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height) {
    float3 direction = float3{float(x)/width-float(0.5), float(y)/float(height)-float(0.5), 1} - eye;
    
    ray_t ray{eye, direction/sqrt(dot(direction,direction))};

    unsigned char answer = 10;

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
    int idx = y*width + x;
    image[idx] = uchar3{answer,answer,answer};
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
        sphere_t{float3{0.1, 0.1, 3}, 1.1},
          sphere_t{float3{sin(tt-0.8f)-0.4f, -0.5, 1.8}, 0.25}},
      context);    
  
    dim3 grid_dim{width/32 + (width % 32 > 0), height/32 + (height % 32 > 0)};
    dim3 block_dim{32,32};

    ray_trace<<<grid_dim, block_dim>>>(image_out, width, height,
                                       geometry.data(),
                                       geometry.size());
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

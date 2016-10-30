#pragma once

#include "types.hxx"

#include <moderngpu/kernel_segreduce.hxx>
#include <moderngpu/transform.hxx>
#include <moderngpu/memory.hxx>

#include "camera.hxx"

#include "sampler.hxx"

struct raytracer_t {
  mgpu::standard_context_t context;
  mgpu::mem_t<sphere_t> geometry;
  mgpu::mem_t<float3> sample_buffer, image_buffer, frame_buffer;

  uint width, height, nsamples;

  uint frame_number = 0;

  camera_t camera;
  
  raytracer_t(uint width, uint height, uint nsamples) :
    width(width), height(height), nsamples(nsamples),
    sample_buffer(width*height*nsamples, context),
    image_buffer(width*height, context),
    frame_buffer(width*height, context) {
    camera_moved();
  }

  void camera_moved() {
    auto frame_buffer_data = frame_buffer.data();
    mgpu::transform([=]MGPU_DEVICE(int index) {
        frame_buffer_data[index] = float3{0,0,0};
      },
      width*height,
      context);

    frame_number = 0;
  }
  
  void draw_scene(uchar3 *image_out, uint width, uint height) {
    frame_number = 0;
    frame_number++;
    float tt = float(clock())/CLOCKS_PER_SEC;
    
    geometry = mgpu::to_mem(std::vector<sphere_t>{//Scene: radius, position, emission, color, material
        sphere_t{8.5,float3{40+40*sin(tt), 8.5,99}, float3{3,3,3},float3{1,1,1}*.999},//bulb     
        sphere_t{1e5, float3{-1e5+99,40.8,81.6},float3{0,0,0},float3{.25,.25,.75}},//Rght 

        
          sphere_t{1e5, float3{ 1e5+1,40.8,81.6}, float3{0.0,0,0},float3{.75,.25,.25}},//Left 

            sphere_t{1e5, float3{50,40.8, 1e5},     float3{0,0,0},float3{.75,.75,.75}},//Back 
              sphere_t{1e5, float3{50,40.8,-1e5+300}, float3{0,0,0},float3{1,1,1}          },//Frnt 
                sphere_t{1e5, float3{50, 1e5, 81.6},    float3{0,0,0},float3{.75,.75,.75}},//Botm 
                  sphere_t{1e5, float3{50,-1e5+81.6,81.6},float3{0,0,0.0},float3{.75,.75,.75}},//Top 
                    sphere_t{16.5,float3{27,16.5,47},       float3{0,0,0},float3{1,1,1}*.999},//Mirr 
                      sphere_t{16.5,float3{73,16.5,78},       float3{0,0,0},float3{1,1,1}*.999}//Glas
                                            
                          //      sphere_t{600, float3{50,681.6-.27,81.6},0*float3{12,12,12},  float3{1,1,1}} //Lite 
                          },
      context);
    
    dim3 grid_dim{width/16 + (width % 16 > 0),
        height/16 + (height % 16 > 0),
        nsamples/4 + (nsamples % 4 > 0)};
    dim3 block_dim{16,16,4};

    sample_paths<<<grid_dim, block_dim>>>(sample_buffer.data(), width, height,
                                       nsamples,
                                       frame_number,
                                       camera,
                                       geometry.data(),
                                          geometry.size(),
      0);

    int nsamples_local = nsamples;    

    auto sample_buffer_data = sample_buffer.data();
    auto frame_buffer_data = frame_buffer.data(),
      image_buffer_data = image_buffer.data();
    
    mgpu::transform([=]MGPU_DEVICE(uint index) {
        float3 answer{0,0,0};
        
        for(int ii = 0; ii < nsamples_local; ++ii) {
          answer += sample_buffer_data[nsamples_local*index+ii];
        }

        image_buffer_data[index] = answer;
      },
      width*height,
      context);
        
#ifdef PROGRESSIVE
    auto frame_buffer_data = frame_buffer.data(), image_buffer_data = image_buffer.data();
    int fn = frame_number;
    mgpu::transform([=]MGPU_DEVICE(int index) {
        float3 prev = frame_buffer_data[index];
        
        float3 answer = prev + (image_buffer_data[index] - prev)/fn;

        frame_buffer_data[index] = answer;
      },
      width*height,
      context);
#endif

    simple_interop::float3_to_uchar3(image_buffer.data(), image_out, width, height);
    
  }
};


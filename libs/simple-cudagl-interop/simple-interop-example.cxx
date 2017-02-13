#include "simple-interop.hxx"

using uchar = unsigned char;

__global__ void ray_trace(uchar3 *output, int width, int height) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if(ix < width and iy < height) {
    const int out_idx = width*iy + ix;

    output[out_idx] = uchar3{uchar(255-ix/4),uchar(iy/6),10};
  }
}

simple_interop_t *interop;

void render(){
  unsigned int width = interop->width, height = interop->height;
  
  interop->cuda_render([=](uchar3 *d_textureBufferData) {
      dim3 grid_dim{width/32 + (width % 32 > 0), height/32 + (height % 32 > 0)};
      dim3 block_dim{32,32};
      ray_trace<<<grid_dim, block_dim>>>(d_textureBufferData, width, height);
    });

  
  glFlush();
  glutPostRedisplay();  
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowSize(1024, 768);
  glutCreateWindow("Render with CUDA");

  simple_interop_t main_interop(1000,1000);
  interop = &main_interop;

  glutDisplayFunc(render);
  glutMainLoop();
  return 0;
}

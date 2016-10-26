#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <functional>

struct simple_interop_t {
  const unsigned int width, height;
  GLuint mtexture;

  uchar3* h_textureBufferData = nullptr;
  uchar3* d_textureBufferData = nullptr;

  GLuint gl_pixelBufferObject = 0;
  cudaGraphicsResource * cudaPboResource = nullptr;

  simple_interop_t(unsigned int w, unsigned int h) : width(w), height(h) {
    glEnable(GL_TEXTURE_2D);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glGenTextures(1, &mtexture);
    glBindTexture(GL_TEXTURE_2D, mtexture);

    glTexImage2D(GL_TEXTURE_2D,
                 0,                    // level 0
                 3,                    // use only R, G, and B components
                 width, height,        // texture has width x height texels
                 0,                    // no border
                 GL_RGB,               // texels are in RGB format
                 GL_UNSIGNED_BYTE,     // color components are unsigned bytes
                 h_textureBufferData);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenBuffers(1, &gl_pixelBufferObject);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar3),
                 h_textureBufferData, GL_STREAM_COPY);
 
    cudaError result = cudaGraphicsGLRegisterBuffer(&cudaPboResource, gl_pixelBufferObject,
                                                    cudaGraphicsMapFlagsWriteDiscard);

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

  }

  void cuda_render(std::function<void(uchar3*)> ff) {
    cudaGraphicsMapResources(1, &cudaPboResource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData,
                                         &num_bytes, cudaPboResource);
    ff(d_textureBufferData);
    cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

    draw_quad();
  }

  void draw_quad() {
    glClear(GL_COLOR_BUFFER_BIT);  
    
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, mtexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);
 
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    width,height,
                    GL_RGB, GL_UNSIGNED_BYTE, 0);
  
    float x = 1.0;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(-x, -x, 0.5);
    glTexCoord2f(1.0, 0.0); glVertex3f(x, -x, 0.5);
    glTexCoord2f(1.0, 1.0); glVertex3f(x, x, 0.5);
    glTexCoord2f(0.0, 1.0); glVertex3f(-x, x, 0.5);
    glEnd();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  
};

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

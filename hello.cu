#define GL_GLEXT_PROTOTYPES

#include <GL/gl.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <time.h>

int width = 1024, height = 768;
GLuint mtexture;

uchar3* h_textureBufferData = nullptr;
uchar3* d_textureBufferData = nullptr;

GLuint gl_pixelBufferObject = 0;
cudaGraphicsResource * cudaPboResource = nullptr;

__global__ void ray_trace(uchar3 *output) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  const int out_idx = 2*iy + ix;
  output[out_idx] = uchar3{255*ix,127*iy,0};
}

void render(){
   cudaGraphicsMapResources(1, &cudaPboResource, 0);
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData,
    &num_bytes, cudaPboResource);
 
  dim3 gridDim{2,2};
  dim3 blockDim{1,1};
  ray_trace<<<gridDim, blockDim>>>(d_textureBufferData);
 
  cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

  
  glClear(GL_COLOR_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  int tt = clock() % 1000000;
  float angle = 360.0*tt/1000000;
  glRotatef(angle, 0, 0, 1);

  glEnable(GL_TEXTURE_2D); // you should use shader, but for an example fixed pipeline is ok ;)
  glBindTexture(GL_TEXTURE_2D, mtexture);
glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);
 
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                  2,2,
    GL_RGB, GL_UNSIGNED_BYTE, 0);
  
  float x = 0.8;
  glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(-x, -x, 0.5);
    glTexCoord2f(1.0, 0.0); glVertex3f(x, -x, 0.5);
    glTexCoord2f(1.0, 1.0); glVertex3f(x, x, 0.5);
    glTexCoord2f(0.0, 1.0); glVertex3f(-x, x, 0.5);
  glEnd();
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glFlush();
  glutPostRedisplay();  
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowSize(520, 390);
  glutCreateWindow("Render with CUDA");

#define red {0xff, 0x00, 0x00}
#define yellow {0xff, 0xff, 0x00}
#define magenta {0xff, 0, 0xff}
GLubyte textureData[][3] = {
    red, yellow,
    yellow, red,
};

  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glEnable(GL_TEXTURE_2D);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glGenTextures(1, &mtexture);
  glBindTexture(GL_TEXTURE_2D, mtexture);
  glTexImage2D(GL_TEXTURE_2D,
               0,                    // level 0
               3,                    // use only R, G, and B components
               2, 2,                 // texture has 2x2 texels
               0,                    // no border
               GL_RGB,               // texels are in RGB format
               GL_UNSIGNED_BYTE,     // color components are unsigned bytes
               textureData);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
 glGenBuffers(1, &gl_pixelBufferObject);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, 2 * 2 * sizeof(uchar4),
    h_textureBufferData, GL_STREAM_COPY);
 
  cudaError result = cudaGraphicsGLRegisterBuffer(&cudaPboResource, gl_pixelBufferObject,
    cudaGraphicsMapFlagsWriteDiscard);

    glutDisplayFunc(render);
    glutMainLoop();
  return 0;
}

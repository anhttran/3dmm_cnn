/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "FBRender.h"

const unsigned int FBRender::winWidth = 128;
const unsigned int FBRender::winHeight = 128;
using namespace std;

FBRender::FBRender( int width,
				    int height,
					bool showWindow ):
fbWidth( width ), fbHeight( height )
{	
	initGL();
	initGLBuffers();
}

FBRender::~FBRender()
{
}

void FBRender::init(int width, int height, bool showWindow)
{
    fbWidth = width;
    fbHeight = height;

    //Init OpenGL context
    initGL();

    //Init FBO
    initGLBuffers();
}

void
FBRender::initGL()
{
	char *myArgv [1];
	int myArgc = 1;
	myArgv [0]= strdup( "c" );

	// Create GL context
    if (glutGetWindow() == 0)
    glutInit( &myArgc, myArgv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize( fbWidth, fbHeight );
	glutInitWindowPosition(100,100);
	wdId = glutCreateWindow("Super Triangle Calculator");
    	int iGLUTWindowHandle = glutCreateWindow( "CUDA OpenGL post-processing" );
	wdId2 = iGLUTWindowHandle;

	//OpenGL extensions		
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		std::cerr << "***FBRender::init(): GLEW Error: " << glewGetErrorString(err) << '\n';
		exit(EXIT_FAILURE);
	}
	if(!glewIsSupported("GL_EXT_framebuffer_object")) {
		std::cerr << "***FBRender::init(): GL_EXT_framebuffer_object extension was not found\n";
		exit(EXIT_FAILURE);
	}

	//Black background color
	glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );

    //Viewport
	glDisable( GL_DEPTH_TEST );
    glViewport( 0, 0, fbWidth, fbHeight );

    // projection
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective( 60.0, ( GLfloat )fbWidth/( GLfloat )fbHeight, 0.1, 10.0 );
		
}

void 
FBRender::initGLBuffers()
{
	//Create a texture
	glGenTextures( 1, &g_dynamicTextureID );
	glBindTexture( GL_TEXTURE_2D, g_dynamicTextureID );
	//Set basic parameters
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	//Allocate texture storage
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, fbWidth, fbHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL );

	//Create a renderbuffer
	glGenRenderbuffersEXT( 1, &g_depthRenderBuffer );
	glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, g_depthRenderBuffer );
	//Allocate storage
	glRenderbufferStorageEXT( GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, fbWidth, fbHeight );
	//Clean up
	glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, 0 );

	//Create a framebuffer
	glGenFramebuffersEXT( 1, &g_frameBuffer );
	glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, g_frameBuffer );
	//Attach images
	glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, g_dynamicTextureID, 0 );
	glFramebufferRenderbufferEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, g_depthRenderBuffer );
	//Clean up
	glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );	
}

void
FBRender::mapRendering( SimpleMesh& m )
{
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, g_depthRenderBuffer);	//USELESS
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, g_frameBuffer);			//MUST HAVE

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	if ( m.tex_.isLoaded() ) 
	{
		glEnable( GL_TEXTURE_2D );
		glBindTexture( GL_TEXTURE_2D, m.tex_.getTextureID() );
	}
	else 
		glDisable( GL_TEXTURE_2D );
	glEnable( GL_DEPTH_TEST );
	glDepthFunc( GL_LEQUAL );
	glEnable( GL_CULL_FACE );
	glCullFace( GL_FRONT );


	glShadeModel( GL_SMOOTH );
	glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );

	glEnableClientState( GL_VERTEX_ARRAY );
	if ( m.normals && m.normid > 0 )	glEnableClientState( GL_NORMAL_ARRAY );
	if ( m.texcoords_ && m.txid > 0 )	glEnableClientState( GL_TEXTURE_COORD_ARRAY );
	if ( m.colors_ && m.colorid > 0 )	glEnableClientState( GL_COLOR_ARRAY);
	
	glBindBuffer( GL_ARRAY_BUFFER, m.ptid );
	glVertexPointer( 3, GL_FLOAT, 0, NULL );

	if ( m.normals && m.normid > 0 ) 
	{
		glBindBuffer( GL_ARRAY_BUFFER, m.normid );
		glNormalPointer( GL_FLOAT, 0, NULL );
	}
	if ( m.colors_ && m.colorid > 0 ) 
	{
		glBindBuffer( GL_ARRAY_BUFFER, m.colorid );
		glColorPointer( 4,GL_FLOAT, 0, NULL );
	}

	if ( m.texcoords_ && m.txid > 0 ) 
	{
		glBindBuffer( GL_ARRAY_BUFFER, m.txid );
		glTexCoordPointer( 2, GL_FLOAT, 0, NULL );
	}

	//WARNING: we work only with triangle-based models
	//If an error occurs at this point, check if the model was correctly loaded ;)
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m.idxid);
	glDrawElements( GL_TRIANGLES, m.nFaces_ * 3, GL_UNSIGNED_INT, NULL );

	glDisableClientState( GL_VERTEX_ARRAY );
	glDisableClientState( GL_NORMAL_ARRAY );
	glDisableClientState( GL_TEXTURE_COORD_ARRAY );
	glDisableClientState( GL_COLOR_ARRAY );
}

void
FBRender::unmapRendering()
{
	//Unbind the framebuffer
	glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
}

//K is a 3x3 camera matrix
//imgWidth and imgHeight is the size of the image associated with K
void 
FBRender::loadIntrinGL( CvMat *K, 
					    double znear, 
						double zfar, 
						int imgWidth, 
                        int imgHeight)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	
	//flip y, because image y increases downward, whereas camera y increases upward
	glScaled(1, -1, 1);
	float fovy = ( float )( 2 * atan( ( float )imgHeight / ( 2*cvmGet( K, 1, 1 ))) * 180.f / 3.141592f );
	float aspect = ( float )(( float )imgWidth / ( float )imgHeight * cvmGet( K, 0, 0 ) / cvmGet( K, 1, 1 ));

	gluPerspective(fovy, aspect, znear, zfar) ;
	
	//bind FBO
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, g_depthRenderBuffer);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, g_frameBuffer);
}

//H is a 4x4 3D transformation matrix (R and t)
void 
FBRender::loadExtrinGL(CvMat *H)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//flip z, because opengl camera is looking down -z, as opposed to +z, 
	//which is standard in computer vision
	//glScaled(1, 1, -1); //FIXYANN: Our model has the same coord. sys. as OpenGL
	GLdouble utm2cam_gl[16];
	for(int i = 0; i < 4; ++i) {
		for(int j = 0; j < 4; ++j) {
			utm2cam_gl[j*4 + i] = cvmGet(H, i, j);
		}
	}
	glMultMatrixd(utm2cam_gl);
}

void 
FBRender::readFB( cv::Mat &img )
{
	//bind FBO
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, g_depthRenderBuffer);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, g_frameBuffer);
	
	//copy from dynamic texture (FBO) to opencv structure
	glReadPixels( 0, 0, fbWidth, fbHeight, GL_BGR, GL_UNSIGNED_BYTE, ( GLvoid* )(img.data));
	
	// Unbind the frame-buffer and render-buffer objects.
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
}

void 
FBRender::readDB( cv::Mat &depth )
{
	//bind FBO
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, g_depthRenderBuffer);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, g_frameBuffer);
	
	//copy from depth buffer to opencv structure
	glReadPixels( 0, 0, fbWidth, fbHeight, GL_DEPTH_COMPONENT, GL_FLOAT, depth.data );
	
	// Unbind the frame-buffer and render-buffer objects.
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
}




/****************************************************************
*						DEBUGGING FUNCTIONS						*
****************************************************************/


void FBRender::checkProjection(double x, double y, double z)
{
	GLdouble check_mv[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, check_mv);
	GLdouble check_proj[16];
	glGetDoublev(GL_PROJECTION_MATRIX, check_proj);
	GLint check_view[4];
	glGetIntegerv(GL_VIEWPORT, check_view);
	GLdouble winx, winy, winz;
	gluProject(x, y, z, check_mv, check_proj, check_view, &winx, &winy, &winz);
	std::cout << "(" << x << "," << y << "," << z << ") projects to "
	          << "(" << winx << "," << winy << "," << winz << ")\n";
}

void FBRender::checkModelView(double x, double y, double z)
{
	GLdouble check_mv[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, check_mv);
	CvMat *H = cvCreateMat(4, 4, CV_64FC1);
	for(int i = 0; i < 4; ++i) {
		for(int j = 0; j < 4; ++j) {
			cvmSet(H, i, j, check_mv[j*4 + i]);
		}
	}
	CvMat *pt4x1 = cvCreateMat(4, 1, CV_64FC1);
	cvmSet(pt4x1, 0, 0, x);
	cvmSet(pt4x1, 1, 0, y);
	cvmSet(pt4x1, 2, 0, z);
	cvmSet(pt4x1, 3, 0, 1);
	CvMat *Hpt4x1 = cvCreateMat(4, 1, CV_64FC1);
	cvMatMul(H, pt4x1, Hpt4x1);
	
	std::cout << "(" << x << "," << y << "," << z << ") in camera (eye) is "
	          << "(" << cvmGet(Hpt4x1, 0, 0) << "," 
	                 << cvmGet(Hpt4x1, 1, 0) << "," 
	                 << cvmGet(Hpt4x1, 2, 0) << ","
	                 << cvmGet(Hpt4x1, 3, 0) << ")\n";
	cvReleaseMat(&H);
	cvReleaseMat(&pt4x1);
	cvReleaseMat(&Hpt4x1);
}

void FBRender::checkClip(double x, double y, double z)
{
	GLdouble check_mv[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, check_mv);
	CvMat *H = cvCreateMat(4, 4, CV_64FC1);
	for(int i = 0; i < 4; ++i) {
		for(int j = 0; j < 4; ++j) {
			cvmSet(H, i, j, check_mv[j*4 + i]);
		}
	}
	GLdouble check_proj[16];
	glGetDoublev(GL_PROJECTION_MATRIX, check_proj);
	CvMat *P = cvCreateMat(4, 4, CV_64FC1);
	for(int i = 0; i < 4; ++i) {
		for(int j = 0; j < 4; ++j) {
			cvmSet(P, i, j, check_proj[j*4 + i]);
		}
	}
	CvMat *pt4x1 = cvCreateMat(4, 1, CV_64FC1);
	cvmSet(pt4x1, 0, 0, x);
	cvmSet(pt4x1, 1, 0, y);
	cvmSet(pt4x1, 2, 0, z);
	cvmSet(pt4x1, 3, 0, 1);
	CvMat *Hpt4x1 = cvCreateMat(4, 1, CV_64FC1);
	cvMatMul(H, pt4x1, Hpt4x1);
	cvCopy(Hpt4x1, pt4x1);
	cvMatMul(P, pt4x1, Hpt4x1);
	
	std::cout << "(" << x << "," << y << "," << z << ") in clip is "
	          << "(" << cvmGet(Hpt4x1, 0, 0) << "," 
	                 << cvmGet(Hpt4x1, 1, 0) << "," 
	                 << cvmGet(Hpt4x1, 2, 0) << ","
	                 << cvmGet(Hpt4x1, 3, 0) << ")\n";
	cvReleaseMat(&H);
	cvReleaseMat(&pt4x1);
	cvReleaseMat(&Hpt4x1);
	cvReleaseMat(&P);
}

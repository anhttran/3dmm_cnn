/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
//----------------------------------------------------------------------
// File: FTModel.cpp
//      Authors:  Yann Dumortier (yann.dumortier@gmail.com),
//                Jongmoo Choi (jongmooc@usc.edu),
//                Sang-il Choi (csichoisi@gmail.net)
// Description: 
// This file is part of the "Real-time 3D Face Tracking and Modeling Using a Webcam" 
//      developed at the University of Southern California by:
//
// Yann Dumortier (yann.dumortier@gmail.com),
// Jongmoo Choi (jongmooc@usc.edu),
// Gerard Medioni (medioni@usc.edu).
//----------------------------------------------------------------------
//      Copyright (c) 2011 University of Southern California.  All Rights Reserved.

#include "FTModel.h"
//#include "MeshModel.h"

//CHECKED and APPROUVED without bug ;)
//Give fx=fy and Tx Ty Tz in the face coordinate system (to be consistent with the tracking, also in this system)
int
StatiCam::calibIntWoRot( float cx,
						 float cy,
						 unsigned n,
						 float *pts2D, 
						 float *pts3D,
						 float *t )
{
	float height = cy*2;
	cx_ = cx;
	cy_ = cy;

	float *tab_A = new float[8*n];

	for ( unsigned i_2D=0, i_3D=0, i_A=0; i_2D<2*n; i_2D+=2, i_3D+=3 )
	{
		//"1st row"
		tab_A[i_A++] = -pts3D[i_3D];						//-Xi
		tab_A[i_A++] = pts3D[i_3D+2]*(cx_-pts2D[i_2D]);		//Zi(cx-ui)
		tab_A[i_A++] = 1;									//1
		tab_A[i_A++] = 0;									//0

		//"2nd row"
		tab_A[i_A++] = pts3D[i_3D+1];						//Yi
		tab_A[i_A++] = pts3D[i_3D+2]*(cy_-pts2D[i_2D+1]);	//Zi(cy-vi)
		tab_A[i_A++] = 0;									//0
		tab_A[i_A++] = 1;									//1
	}

	cv::Mat A( 2*n, 4, CV_32F, tab_A );
	cv::Mat b( 2*n, 1, CV_32F, pts2D );

	cv::Mat x = (A.t()*A).inv()*A.t()*b; 

	//Intrinsic parameters
	f_ = (( float* )( x.data ))[0] / (( float* )( x.data ))[1];

	//Extrinsic parameters
	t[0] = -((( float* )( x.data ))[2]-cx_ ) / (( float* )( x.data ))[0];
	t[1] = ((( float* )( x.data ))[3]-cy_ ) / (( float* )( x.data ))[0];
	t[2] = 1.f / (( float* )( x.data ))[1];

	/*std::cerr << "focal: " << f_  << ", (cx,cy): (" << cx_ << "," << cy_ << ")\n";
	std::cerr << "t=[" << t[0] << "," << t[1] << "," << t[2] << "]\n";*/

	if(t[2]>0) 
	{
		t[0] = ((( float* )( x.data ))[2]-cx_ ) / (( float* )( x.data ))[0];
		t[1] = -((( float* )( x.data ))[3]-cy_ ) / (( float* )( x.data ))[0];
		t[2] = -1.f / (( float* )( x.data ))[1];
		f_ = -1*f_;
	}


	delete[] tab_A;
	return 1;
}

int
StatiCam::calibIntWoRot2( float cx,
						 float cy,
						 unsigned n,
						 float *pts2D, 
						 float *pts3D,
						 float *t )
{
	float height = cy*2;
	cx_ = cx;
	cy_ = cy;
	int n2 = 6;

	float *tab_A = new float[8*n2];
	float *tab_B = new float[2*n2];
	int* ind = new int[n];
	float *t2 = new float[3];
	float f_2;
	cv::RNG rng;

	int bestCons = 0;
	for (int i=0; i< 100; i++)
	{
		for (int j=0; j<n; j++) ind[j] = j;
		for (int j=0; j<n2;j++)
		{
			int jj = rng.next() % n;
			if (j != jj)
			{
				int k= ind[jj];
				ind[jj] = ind[j];
				ind[j] = k;
			}
		}
		for ( unsigned i_D=0; i_D<n2; i_D++ )
		{
			int j = ind[i_D];
			//"1st row"
			tab_A[i_D * 8] = -pts3D[j*3];						//-Xi
			tab_A[i_D * 8 + 1] = pts3D[j*3+2]*(cx_-pts2D[j*2]);		//Zi(cx-ui)
			tab_A[i_D * 8 + 2] = 1;									//1
			tab_A[i_D * 8 + 3] = 0;									//0

			//"2nd row"
			tab_A[i_D * 8 + 4] = pts3D[j*3+1];						//Yi
			tab_A[i_D * 8 + 5] = pts3D[j*3+2]*(cy_-pts2D[j*2+1]);	//Zi(cy-vi)
			tab_A[i_D * 8 + 6] = 0;									//0
			tab_A[i_D * 8 + 7] = 1;									//1
		
			tab_B[i_D * 2] = pts2D[j*2];
			tab_B[i_D * 2 + 1] = pts2D[j*2 + 1];
		}

		cv::Mat A( 2*n2, 4, CV_32F, tab_A );
		cv::Mat b( 2*n2, 1, CV_32F, tab_B );

		cv::Mat x = (A.t()*A).inv()*A.t()*b; 
			//Intrinsic parameters
			f_2 = (( float* )( x.data ))[0] / (( float* )( x.data ))[1];

			//Extrinsic parameters
			t2[0] = -((( float* )( x.data ))[2]-cx_ ) / (( float* )( x.data ))[0];
			t2[1] = ((( float* )( x.data ))[3]-cy_ ) / (( float* )( x.data ))[0];
			t2[2] = 1.f / (( float* )( x.data ))[1];
			// Fix focal len
			float scale = 1000/f_2;
			f_2 = 1000;
			t2[0] = t2[0] * scale;
			t2[1] = t2[1] * scale;
			t2[2] = t2[2] * scale;
			//printf("f_2 = %f       t2[2] = %f \n",f_2,t2[2]);

			/*std::cerr << "focal: " << f_  << ", (cx,cy): (" << cx_ << "," << cy_ << ")\n";
			std::cerr << "t=[" << t[0] << "," << t[1] << "," << t[2] << "]\n";*/

			//if(t2[2]>0) 
			//{
				//t2[0] = ((( float* )( x.data ))[2]-cx_ ) / (( float* )( x.data ))[0];
				//t2[1] = -((( float* )( x.data ))[3]-cy_ ) / (( float* )( x.data ))[0];
				//t2[2] = -1.f / (( float* )( x.data ))[1];
				//f_2 = -1*f_2;

			//}

		int cons = 0;
		float u, v;
		for (unsigned i_D=0; i_D<n; i_D++)
		{
			u = -f_2*(pts3D[i_D *3] + t2[0])/(pts3D[i_D *3 + 2] + t2[2]) + cx_;
			v = f_2*(pts3D[i_D *3+1] + t2[1])/(pts3D[i_D *3 + 2] + t2[2]) + cy_;
			float dist = std::pow(u - pts2D[i_D *2],2);
			dist += std::pow(v - pts2D[i_D *2 + 1],2);
			if (dist < 100) cons++;
		}
		if (cons > bestCons)
		{
			bestCons = cons;
			t[0] = t2[0];
			t[1] = t2[1];
			t[2] = t2[2];
			f_ = f_2;
			if (bestCons > 27) break;
		}
	}
	//printf("Bc = %d\n",bestCons);

	delete[] tab_A;
	delete[] tab_B;
	delete[] ind;
	delete[] t2;
	return 1;
}

//Load the correct 3-D shape according to the 
//"id" returned by the recognition module (0 <=> generic model).
Face::Face( unsigned id )
{
	id_ = id;
	nLdmks_ = 0;
	R_ = new float[6];
	t_ = &R_[3];
	memset(( void* )R_, 0, 6*sizeof( float ));	
}

Face::Face( Face &f,
		    unsigned id )
{
			if ( id ) id_ = id;
			else id_ = f.id_;
			nLdmks_ = f.nLdmks_;
			memcpy( landmarks_, f.landmarks_, MAX_LDMKS*sizeof( int ));

			//TODO: use a copy constructor of f.mesh instead;
			mesh_.nFaces_ = f.mesh_.nFaces_ ;
			mesh_.faces_ = new unsigned[3*mesh_.nFaces_];
			memcpy( mesh_.faces_, f.mesh_.faces_, 3*mesh_.nFaces_*sizeof( unsigned ));

			mesh_.nVertices_ = f.mesh_.nVertices_;
			mesh_.vertices_ = new float[3*mesh_.nVertices_];
			memcpy( mesh_.vertices_, f.mesh_.vertices_, 3*mesh_.nVertices_*sizeof( float ));

			mesh_.texcoords_ = new float[3*mesh_.nVertices_];

			R_ = new float[6];
			t_ = &R_[3];
			memcpy(( void* )R_, ( void* )f.R_, 6*sizeof( float )); 
}


Face::~Face()
{
	delete[] R_; //delete R_ AND t_;
}


//WARNING: triangle vertices order is important to ensure a correct display.
//See OpenGL documentation about glEnable( GL_CULL_FACE ) and glCullFace( GL_FRONT )
//for more details.
int
Face::loadPLYModel( const char *fileName )
{
	std::ifstream file( fileName );
    if ( !file ) return invalidPLYFile();

	std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

	std::string buf = buffer.str();
	if ( buf.substr( 0, 3 ) != "ply" ) return invalidPLYFile();

	size_t pos;
	pos = buf.find( "element vertex" );
	if ( pos == std::string::npos) return invalidPLYFile();
	buffer.seekg( pos + 14 );
	buffer >> mesh_.nVertices_;
	mesh_.vertices_ = new float[3*mesh_.nVertices_]; 

	pos = buf.find( "element face" );
	if ( pos == std::string::npos) return invalidPLYFile();
	buffer.seekg( pos + 12 );
	buffer >> mesh_.nFaces_;
	mesh_.faces_ = new unsigned[3*mesh_.nFaces_];

	pos = buf.find( "end_header" );
	buffer.seekg( pos );
	buffer.ignore( 1024, '\n' );

	//Vertices
	for ( unsigned i=0, idx=0; i<mesh_.nVertices_; ++i )
	{	
		buffer >> mesh_.vertices_[idx++];	//x
		mesh_.vertices_[idx-1] += MODEL_TX;
		mesh_.vertices_[idx-1] *= MODEL_SCALE;
		buffer >> mesh_.vertices_[idx++];	//y
		mesh_.vertices_[idx-1] += MODEL_TY;
		mesh_.vertices_[idx-1] *= MODEL_SCALE;
		buffer >> mesh_.vertices_[idx++];	//z
		mesh_.vertices_[idx-1] += MODEL_TZ;
		mesh_.vertices_[idx-1] *= MODEL_SCALE;
		buffer.ignore( 1024, '\n' );		//potential comments
	}

	//Faces
	unsigned nEdges;
	for ( unsigned i=0, idx=0; i<mesh_.nFaces_; ++i )
	{
		buffer >> nEdges;
		if ( nEdges != 3 ) return invalidPLYFile();
		buffer >> mesh_.faces_[idx++];		//v1
		buffer >> mesh_.faces_[idx++];		//v2
		buffer >> mesh_.faces_[idx++];		//v3
		buffer.ignore( 1024, '\n' );		//potential comments
	}

	return 1;
}

//Load landmark points corresponding to the generic 3D face model
//Landmarks do not have to be point of the model. In such case, 
//closest points are used.

int Face::loadPLYModel2(const char* ply_file){
	int state = 0;

	char str[250];
	char* pos[10];
	unsigned char prop_count = 0;
	unsigned char props[9];
	int count;
	int vcount = 0;
	int fcount = 0;
	int i;

	FILE* file = fopen(ply_file,"r");
	if (!file) return -1;
	while (!feof(file) && state <= 4){
		fgets(str,250,file);
		if (strlen(str) < 3) continue;
		count = splittext(str, pos);
		if (count < 1 || (strcmp(pos[0],"comment") == 0)) continue;
		
		switch (state){
			case 0:								// at beginning
				if (count != 3 || (strcmp(pos[0],"element") != 0) || (strcmp(pos[1],"vertex") != 0)) continue;
				mesh_.nVertices_ = atoi(pos[2]);
				if (mesh_.nVertices_ < 1) {
					mesh_.nVertices_ = 0; return -1;
				}
				mesh_.vertices_ = new float[3*mesh_.nVertices_];
				state = 1;
				break;
			case 1:								// get properties
				if (strcmp(pos[0],"end_header") == 0) state = 3;
				else if (strcmp(pos[0],"element") == 0){
					if (strcmp(pos[1],"face") == 0){
						state = 2;
						mesh_.nFaces_ = atoi(pos[2]);
						mesh_.faces_ = new unsigned int[3*mesh_.nFaces_];
					}
				}
				else if (count == 3 && (strcmp(pos[0],"property") == 0)){
					if (strcmp(pos[2],"x") == 0) {
						props[prop_count] = PROP_X;
						prop_count++;
					}
					if (strcmp(pos[2],"y") == 0){
						props[prop_count] = PROP_Y;
						prop_count++;
					}
					if (strcmp(pos[2],"z") == 0){
						props[prop_count] = PROP_Z;
						prop_count++;
					}
					if (strcmp(pos[2],"red") == 0){
						mesh_.colors_ = new float[4*mesh_.nVertices_];
						props[prop_count] = PROP_R;
						prop_count++;
					}
					if (strcmp(pos[2],"green") == 0){
						props[prop_count] = PROP_G;
						prop_count++;
					}
					if (strcmp(pos[2],"blue") == 0){
						props[prop_count] = PROP_B;
						prop_count++;
					}
				}
				break;
			case 2:
				if (strcmp(pos[0],"end_header") == 0) state = 3;
				break;
			case 3:
				for (i = 0; i < prop_count; i++){
					switch (props[i]){
						case PROP_X:
							mesh_.vertices_[vcount*3] = atof(pos[i]); break;
						case PROP_Y:
							mesh_.vertices_[vcount*3+1] = atof(pos[i]); break;
						case PROP_Z:
							mesh_.vertices_[vcount*3+2] = atof(pos[i]); break;
						case PROP_R:
							mesh_.colors_[vcount*4] = atoi(pos[i])/255.0f;
							mesh_.colors_[vcount*4+3] = 1.0f; break;
						case PROP_G:
							mesh_.colors_[vcount*4+1] = atoi(pos[i])/255.0f; break;
						case PROP_B:
							mesh_.colors_[vcount*4+2] = atoi(pos[i])/255.0f; break;
					}
				}
				vcount++;
				if (vcount == mesh_.nVertices_) {
					if (mesh_.nFaces_ > 0)
						state = 4;
					else 
						state = 5;
				}
				break;
			case 4:
				mesh_.faces_[3*fcount] = atoi(pos[1]);
				mesh_.faces_[3*fcount+1] = atoi(pos[2]);
				mesh_.faces_[3*fcount+2] = atoi(pos[3]);
				fcount++;
				break;
		}
	}
	fclose(file);
	return 0;
}

void Face::loadMesh( cv::Mat shape, cv::Mat tex, cv::Mat faces){
	mesh_.nVertices_ = shape.rows;
	mesh_.vertices_ = new float[3*mesh_.nVertices_];
	mesh_.colors_ = new float[4*mesh_.nVertices_];
	for (int i=0;i<mesh_.nVertices_;i++){
		for (int j=0;j<3;j++){
			mesh_.vertices_[3*i+j] = shape.at<float>(i,j);
			mesh_.colors_[4*i+j] = tex.at<float>(i,j)/255.0f;
		}
		mesh_.colors_[4*i+3] = 1.0f;
	}
	mesh_.nFaces_ = faces.rows;
	mesh_.faces_ = new unsigned int[3*mesh_.nFaces_];
	for (int i=0;i<mesh_.nFaces_;i++){
		for (int j=0;j<3;j++){
			mesh_.faces_[3*i+j] = faces.at<int>(i,j);
		}
	}
}


int 
Face::loadPLYLandmarks( const char* fileName )
{
	unsigned *idx;
	
	std::ifstream file( fileName );
    if ( !file ) return invalidPLYFile();

	std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

	std::string buf = buffer.str();
	if ( buf.substr( 0, 3 ) != "ply" ) return invalidPLYFile();

	size_t pos;
	pos = buf.find( "element vertex" );
	if ( pos == std::string::npos) return invalidPLYFile();
	buffer.seekg( pos + 14 );
	buffer >> nLdmks_;
	idx = new unsigned[nLdmks_]; 

	pos = buf.find( "comment Landmark_seq:" );
	if ( pos == std::string::npos) return invalidPLYFile();
	buffer.seekg( pos + 21 );
	for ( unsigned i=0; i<nLdmks_; ++i )
		buffer >> idx[i];

	pos = buf.find( "end_header" );
	buffer.seekg( pos );
	buffer.ignore( 1024, '\n' );

	memset( landmarks_, -1, sizeof( int )*MAX_LDMKS );

	//Landmarks
	for ( unsigned i=0; i<nLdmks_; ++i )
	{	
		float x, y, z;
		buffer >> x >> y >> z;
		x += MODEL_TX;
		y += MODEL_TY;
		z += MODEL_TZ;
		x *= MODEL_SCALE;
		y *= MODEL_SCALE;
		z *= MODEL_SCALE;
		buffer.ignore( 1024, '\n' );			//avoid potential comments

		//Faster, but speed is useless at the initialization.
		//for ( unsigned j=0; j<3*mesh_.nVertices_; j+=3 )
		//{
		//	if ( x == mesh_.vertices_[j] && y == mesh_.vertices_[j+1] && z == mesh_.vertices_[j+2] )
		//		landmarks_[idx[i]] = j/3;
		//}

		//More generic since allows landmarks point to not be exact
		//3D model points.
		float minDist = FLT_MAX;
		for ( unsigned j=0; j<3*mesh_.nVertices_; j+=3 )
		{
			float dist = pow( x-mesh_.vertices_[j], 2 ) + pow( y-mesh_.vertices_[j+1], 2 ) + pow( z-mesh_.vertices_[j+2], 2 );
			if ( dist < minDist )
			{
				minDist = dist;
				landmarks_[idx[i]] = j/3;
			}		
		}

	}
	delete []idx;
	return 1;
}

void
Face::savePLYModel( const char *fileName )
{
	std::ofstream file( fileName );

	if ( !file )
    {
		std::cerr << "Creation Error\n";
        return;
    }
	printf("Infor: %d %d %p %p\n",mesh_.nVertices_,mesh_.nFaces_,mesh_.colors_,mesh_.texcoords_);
	file << "ply\n";
	file << "format ascii 1.0\n";
	file << "element vertex " << mesh_.nVertices_ << std::endl;
	file << "property float x\nproperty float y\nproperty float z\n";
	file << "property uchar blue\nproperty uchar green\nproperty uchar red\n";
	//file << "property float texture_u\nproperty float texture_v\n";
	file << "element face " << mesh_.nFaces_ << std::endl;
	file << "property list uchar int vertex_indices\n";
	file << "end_header\n";

	for ( unsigned i_3D=0, i_2D=0; i_3D<3*mesh_.nVertices_; i_3D+=3, i_2D+=2 )
	{
		file << mesh_.vertices_[i_3D] << " ";
		file << mesh_.vertices_[i_3D+1] << " ";
		file << mesh_.vertices_[i_3D+2] << " "; /*std::endl;*/
		if (mesh_.colors_ != NULL) {
			file << (int)static_cast<unsigned char>(mesh_.colors_[2*i_2D+2]*255) << " ";
			file << (int)static_cast<unsigned char>(mesh_.colors_[2*i_2D+1]*255) << " ";
			file << (int)static_cast<unsigned char>(mesh_.colors_[2*i_2D]*255) << std::endl;
		}
		else
		{
			unsigned x = mesh_.texcoords_[i_2D]   * mesh_.tex_.img_.cols;
			unsigned y = mesh_.texcoords_[i_2D+1] * mesh_.tex_.img_.rows;
			unsigned idx = 3*(x+y*mesh_.tex_.img_.cols);
			file << (int)static_cast<unsigned char>(mesh_.tex_.img_.data[idx]) << " ";
			file << (int)static_cast<unsigned char>(mesh_.tex_.img_.data[idx+1]) << " ";
			file << (int)static_cast<unsigned char>(mesh_.tex_.img_.data[idx+2]) << std::endl;
		}
	}

	for ( unsigned i_3D=0; i_3D<3*mesh_.nFaces_; )
	{
		file << "3 ";
		file << mesh_.faces_[i_3D++] << " ";
		file << mesh_.faces_[i_3D++] << " ";
		file << mesh_.faces_[i_3D++] << std::endl;
	}

	file.close();
	//cv::imwrite( "texture.bmp", mesh_.tex_.img_ );
}

bool Face::estimateNormals( ){
	if (mesh_.normals != 0){
		delete mesh_.normals;
	}

	mesh_.normals = new float[3*mesh_.nVertices_];
	for (int i=0;i<3*mesh_.nVertices_;i++) mesh_.normals[i] = 0;

	float nx, ny, nz;
	for (int i=0;i<mesh_.nFaces_;i++){
		for (int j=0;j<3;j++) {
			triangleNormalFromVertex(i, j, nx, ny, nz);
			mesh_.normals[3*mesh_.faces_[3*i+j]] += nx;
			mesh_.normals[3*mesh_.faces_[3*i+j]+1] += ny;
			mesh_.normals[3*mesh_.faces_[3*i+j]+2] += nz;
		}
	}
	
	for (int i=0;i<mesh_.nVertices_;i++){
		float no = sqrt(mesh_.normals[3*i]*mesh_.normals[3*i]+mesh_.normals[3*i+1]*mesh_.normals[3*i+1]+mesh_.normals[3*i+2]*mesh_.normals[3*i+2]);
		for (int j=0;j<3;j++) mesh_.normals[3*i+j] = mesh_.normals[3*i+j]/no;
	}
	return true;
}

void Face::triangleNormalFromVertex(int face_id, int vertex_id, float &nx, float &ny, float &nz) {
	int ind0 = mesh_.faces_[3*face_id + vertex_id];
	int ind1 = mesh_.faces_[3*face_id + (vertex_id+1)%3];
	int ind2 = mesh_.faces_[3*face_id + (vertex_id+2)%3];

	float a[3],b[3],v[3];
	for (int j=0;j<3;j++){
		a[j] = mesh_.vertices_[3*ind1+j] - mesh_.vertices_[3*ind0+j];
		b[j] = mesh_.vertices_[3*ind2+j] - mesh_.vertices_[3*ind0+j];
	}
	v[0] = a[1]*b[2] - a[2]*b[1];
	v[1] = a[2]*b[0] - a[0]*b[2];
	v[2] = a[0]*b[1] - a[1]*b[0];
	float no = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
	float dp = a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
	float la = sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
	float lb = sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
	float alpha = acos(dp/(la*lb));

	nx = alpha * v[0]/no;
	ny = alpha * v[1]/no;
	nz = alpha * v[2]/no;
}

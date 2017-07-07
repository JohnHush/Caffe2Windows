#include <caffe/caffe.hpp>
#include <google/protobuf/text_format.h>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <direct.h>

using namespace caffe;
using namespace cv;
using std::string;


uint32_t swap_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

int main()
{
	string image_filename("C:/Users/JohnHush/Desktop/minist_data/t10k-images-idx3-ubyte");
	string label_filename("C:/Users/JohnHush/Desktop/minist_data/t10k-labels-idx1-ubyte");
	string jpg_path("C:/Users/JohnHush/Desktop/minist_data/");

	for ( int i = 0 ; i < 10 ; i ++ )
		mkdir((jpg_path + std::to_string(i)).c_str());

	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

	CHECK(image_file) << "Unable to open file " << image_filename;
	CHECK(label_file) << "Unable to open file " << label_filename;
	// Read the magic and the meta data
	uint32_t magic;
	uint32_t num_items;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;

	image_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
	label_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
	image_file.read(reinterpret_cast<char*>(&num_items), 4);
	num_items = swap_endian(num_items);
	label_file.read(reinterpret_cast<char*>(&num_labels), 4);
	num_labels = swap_endian(num_labels);
	CHECK_EQ(num_items, num_labels);
	image_file.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	image_file.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);

	// Storing to db
	char label;
	char* pixels = new char[rows * cols];
	int count = 0;
	string value;

	for (int item_id = 0; item_id < num_items; ++item_id) {
		Mat imgSrc(28,28,CV_8UC1);
		Mat imgRst(280,280,CV_8UC1);
		image_file.read(pixels, rows * cols);
		label_file.read(&label, 1);

		for (int irow = 0; irow < 28; irow++)
			for (int icol = 0; icol < 28; icol++)
				imgSrc.at<unsigned char>(irow, icol) = pixels[irow*28+icol];
		
		resize( imgSrc , imgRst , imgRst.size() );
		string key_str = caffe::format_int(item_id, 8);

		string pic_path = jpg_path + std::to_string(int(label)) +"/"+ key_str + ".jpg";

		imwrite( pic_path , imgRst );

	}
	// write the last batch

	LOG(INFO) << "Processed " << count << " files.";
	delete[] pixels;

}

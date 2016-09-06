#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>

using namespace std;

int main()
{
	ifstream in("test.txt");
	ofstream out("test.json");
	string line;
	out << '[';
	int dot_flag = 1;
	if (in) {
		while (getline(in, line))
		{
			if (dot_flag) {
				out << "{";
				dot_flag = 0;
			}
			else
				out << ",{";
			const string flag1(" in the image");
			const string flag2("?");
			size_t pos1 = line.find(flag1, 0);
			size_t pos2 = line.find(flag2, 0);
			if (pos1 == string::npos)
				pos1 = line.find(string(" in this image"));
			if (pos1 == string::npos)
				pos1 = line.find(string(" in image"));
			if (pos1 == string::npos)
				pos1 = line.find(string(" of this image"));
			if (pos1 == string::npos)
				pos1 = line.find(string(" of the image"));
			if (pos1 == string::npos)
				pos1 = line.find(string(" of image"));
			if (pos1 == string::npos)
				pos1 = line.find(string(" on this image"));
			if (pos1 == string::npos)
				pos1 = line.find(string(" on the image"));
			if (pos1 == string::npos)
				pos1 = line.find(string(" on image"));
			if (pos1 == string::npos)
				pos1 = line.find(string(" image"));
			if (pos1 == string::npos || pos2 == string::npos)
				break;
			size_t pos3 = pos1 + 8;
			size_t pos4 = pos2 - 1;
			string image, question, answer;
			question.assign(line, 0, pos1);
			image.assign(line, pos3, pos4 - pos3);
			getline(in, answer);
			image = "\"" + image + "\"";
			question = "\"" + question + " ?" + "\"";
			answer = "\"" + answer + "\"";
			out << "\"image\":" << image << ",";
			out << "\"question\":" << question << ",";
			out << "\"answer\":" << answer;
			out << "}";
		}
	}
	out << ']';
	in.close();
	out.close();
}
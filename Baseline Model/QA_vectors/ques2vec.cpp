#include <iostream>
#include <vector>
#include <map>
#include <string>
using namespace std;

map<string, vector<double>> word_index;

void read_wordvec(char *filename, int dim) {
	FILE *fp = fopen(filename, "r");
	string tmpc;
	vector<double> tmpv;
	char ch;
	tmpv.resize(dim);
	while (!feof(fp)) {
		if ((ch = fgetc(fp)) == '#') {
			for (int i = 0; (ch = fgetc(fp)) != '#'; ++i)
				tmpc.append(1, ch);
			for (int i = 0; i < dim; ++i)
				fscanf(fp, "%lf", &tmpv[i]);
			word_index.insert(map<string, vector<double>>::value_type(tmpc, tmpv));
			tmpc.clear();
		}
	}
}

vector<double> ques_vec_generator(string& ques, int dim) {
	vector<string> queswords;
	vector<double> ques_vector;
	vector<double> tmpv;
	string tmp;
	for (unsigned int i = 0; i < ques.length(); ++i) {
		while (isalpha(ques[i]))
			tmp.append(1, ques[i++]);
		queswords.push_back(tmp);
		tmp.clear();
	}
	ques_vector.resize(dim);
	for (unsigned int i = 0; i < queswords.size(); ++i){
		if (word_index.find(queswords[i]) == word_index.end())
			continue;
		tmpv = word_index[queswords[i]];
		for (int j = 0; j < dim; ++j) {
			ques_vector[j] += tmpv[j];
		}
	}
	return ques_vector;
}

int main() {
	char filename[50] = "word-vec.txt";
	read_wordvec(filename, 300);
	string ques = "what is he doing?";
	vector<double> ques_vector = ques_vec_generator(ques, 300);
	for (unsigned int i = 0; i < ques_vector.size(); ++i)
		cout << ques_vector[i] << endl;
	cout << "question: " << ques << endl;
	cout << "dimmension:" << ques_vector.size();
	return 0;
}
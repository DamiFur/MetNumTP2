#include <iostream>
#include <stdlib.h>
#include <list>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <math.h>
#include <set>
#include <cmath>
#include <time.h>

using namespace std;

void trainMatrix(string train, int** ans, int K);
vector<vector<double>> toX(int** ans, int K);
vector<vector<double>> trasponer(vector<vector<double>> matrix, int n, int m);

int main(int argc, char * argv[]){

//ESTO ESTA COMENTADO PORQUE NO ME ARMÉ NINGUN CASO COMPLETO DE INPUT, SIN EMBARGO LA FUNCION QUE ABRE UN ARCHIVO QUE TIENE VECTORES DE IMÁGENES Y LOS PARSEA ANDA
	string inputPath, outputPath; 
	 int metodo; 
	 cout << argc << endl;
	 if (argc < 4){ 
	 	cout << "Input: ";
	 	cin >> inputPath;
	 	cout << "Output: ";
	 	cin >> outputPath; 
	 	cout << "Metodo (0|1|2): ";
	 	cin >> metodo;
	 } else {
	 	inputPath = argv[1];
	 	outputPath = argv[2];
	 	metodo = atoi(argv[3]);
	 	cout << "Input: " << inputPath << endl;
	 	cout << "Output: " << outputPath << endl;
	 	cout << "Metodo: " << metodo << endl;
	 	if (!(metodo == 0 || metodo == 1 || metodo == 2))
	 		return 1;
	 }

	 ifstream input;
	 ofstream output;
	 input.open(inputPath);
	 output.open(outputPath);

	 string train;
	 string test;
	 int kappa;
	 int alpha;
	 int gamma;
	 int crossK;

	 input >> train;
	 test = train + "/test.csv"
	 train += "/train.csv"
	 
	 input >> kappa;
	 input >> alpha;
	 input >> gamma;
	 input >> crossK;

	 bool partitions[crossK][42000];

	 for(int i = 0; i < crossK; i++){
	 	for(int j = 0; j < 42000; j++)
	 		input >> partitions[i][j];
	 }

	/*cout << "enter funcion" << endl;
	string train;
	cin >> train;*/

	int K = 42000;
	//trasformamos train en una matriz donde cada fila tiene el label del digito en la primer columna y 784 columnas más con los pixels
	int* ans[K];
	for(int i = 0; i < K; i++){
		ans[i] = new int[785];
	}
	trainMatrix(train, ans, K);

	vector<vector<double>> x = toX(ans, K);

	vector<vector<double>> xt = trasponer(x, K, 784);

	// vector<vector<double>> xtx = multiply(x, xt, )

	// for(int y = 0; y < K; y++){
	// 	for(int z = 0; z < 784; z++){
	// 		cout << x[y][z] << " ";
	// 	}
	// 	cout << endl << endl;
	// }


}

void trainMatrix(string train, int** ans, int K){

	ifstream input;
	input.open(train);

	// //Eliminamos la primer fila que tiene los nombres de las columnas que no nos sirven
	string deleteFirstRow;
	getline(input, deleteFirstRow);

	for(int i = 0; i < K; i++){
		string row;
		getline(input, row);
		replace(row.begin(), row.end(), ',', ' ');
		stringstream ss;
		ss << row;
		for(int j = 0; j < 785; j++){
			ss >> ans[i][j];
		}
	}

}

vector<vector<double>> trasponer(vector<vector<double>> matrix, int n, int m){

	vector<vector<double>> ans (m, vector<double> (n, 0));

	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			ans[j][i] = matrix[i][j];
		}
	}

	return ans;
}

vector<vector<double>> toX(int** ans, int K){

	double average[784];

	for(int a = 0; a < 784; a++){
		average[a] = 0.0;
	}

	for(int i = 0; i < 784; i++){
		for(int j = 0; j < K; j++){
			average[i] += (double) ans[j][i+1];
		}
	}

	cout << "average: " << endl;
	for(int b = 0; b < 784; b++){
		average[b] /= (double) K;
		cout << average[b] << endl;
	}

	vector<vector<double>> x (K, vector<double> (784, 0));
	// for(int c = 0; c < K; c++){
	// 	x[c] = new vector<double>();
	// }

	double n = sqrt(K - 1);
	for(int d = 0; d < K; d++){
		for(int e = 0; e < 784; e++){
			x[d][e] = (ans[d][e+1] - average[e]) / n;
		}
	}

	return x;

}

#define cuad(x) ((x)*(x))
double distancia(const vector<double>& v1, const vector<double>& v2) {
	double ret = 0.0;

	for (int i = 0; i<v1.size(); ++i) {
		ret += cuad(v1[i]-v2[i]);
	}
	return sqrt(ret);
}

int knn(const vector<pair<int, vector<double>>>& train, const vector<double>& adivinar, int k) {

	set<pair<double, int>> dist_index;

	for (int i = 0; i<train.size(); ++i) {
		dist_index.insert(make_pair(distancia(train[i].second, adivinar), train[i].first));
		if (dist_index.size() > k) {
			auto ulti = dist_index.end();
			ulti--;
			dist_index.erase(ulti);
		}
	}

	int votos[10];
	for (const auto& v : dist_index) {
		votos[v.second]++;
	}

	int ganador = 0;
	for (int i = 1; i<10; ++i) {
		if (votos[i] > votos[ganador]) {
			ganador = i;
		}
	}

	return ganador;
}

vector<double> mult(vector<vector<double> > &a, vector<double> &b){
	vector<double> result (b.size(), 0);
	for (int i = 0; i < b.size(); ++i)
	{
		for (int j = 0; j < a[i].size(); ++j)
			result[i] += b[j]*a[i][j];
	}
	return result;
}

double norm(vector<double> &b){
	double sol = 0;
	for (int i = 0; i < b.size(); ++i)
		sol += b[i]*b[i];
	return sqrt(sol);
}

void normalizar(vector<double> &b){
	double norma = norm(b);
	for (int i = 0; i < b.size(); ++i)
		b[i] /= norma;
}

vector<double> pIteration(vector<vector<double> > &a, int n){
	vector<double> b;
	b.reserve(a.size());
	srand (time(NULL));
	for (int i = 0; i < a.size(); ++i)
		b.push_back((double)(rand() % 1007));
	while(n>0){
		normalizar(b);
		b = mult(a, b);
		n--;
	}
	return b;
}

#include <iostream>
#include <stdlib.h>
#include <list>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <math.h>

using namespace std;

void trainMatrix(string train, int** ans, int K);
double** toX(int** ans, int K);

int main(int argc, char * argv[]){

//ESTO ESTA COMENTADO PORQUE NO ME ARMÉ NINGUN CASO COMPLETO DE INPUT, SIN EMBARGO LA FUNCION QUE ABRE UN ARCHIVO QUE TIENE VECTORES DE IMÁGENES Y LOS PARSEA ANDA
	// string inputPath, outputPath; 
	// int metodo; 
	// cout << argc << endl;
	// if (argc < 4){ 
	// 	cout << "Input: ";
	// 	cin >> inputPath;
	// 	cout << "Output: ";
	// 	cin >> outputPath; 
	// 	cout << "Metodo (0|1|2): ";
	// 	cin >> metodo;
	// } else {
	// 	inputPath = argv[1];
	// 	outputPath = argv[2];
	// 	metodo = atoi(argv[3]);
	// 	cout << "Input: " << inputPath << endl;
	// 	cout << "Output: " << outputPath << endl;
	// 	cout << "Metodo: " << metodo << endl;
	// 	if (!(metodo == 0 || metodo == 1 || metodo == 2))
	// 		return 1;
	// }

	// ifstream input;
	// ofstream output;
	// input.open(inputPath);
	// output.open(outputPath);

	// string train;
	// string test;
	// int kappa;
	// int alpha;
	// int gamma;
	// int crossK;

	// input >> train;
	// input >> test;
	// input >> kappa;
	// input >> alpha;
	// input >> gamma;
	// input >> crossK;

	// bool partitions[crossK][42000];

	// for(int i = 0; i < crossK; i++){
	// 	for(int j = 0; j < 42000; j++){
	// 		input >> partitions[i][j];
	// 	}
	// }

	cout << "enter funcion" << endl;
	string train;
	cin >> train;

	int K = 42000;
	//trasformamos train en una matriz donde cada fila tiene el label del digito en la primer columna y 784 columnas más con los pixels
	int* ans[K];
	for(int i = 0; i < K; i++){
		ans[i] = new int[785];
	}
	trainMatrix(train, ans, K);

	double** x = toX(ans, K);

	for(int y = 0; y < K; y++){
		for(int z = 0; z < 784; z++){
			cout << x[y][z] << " ";
		}
		cout << endl << endl;
	}


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

double** toX(int** ans, int K){

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

	double** x = new double*[K];
	for(int c = 0; c < K; c++){
		x[c] = new double[784];
	}

	double n = sqrt(K - 1);
	for(int d = 0; d < K; d++){
		for(int e = 0; e < 784; e++){
			x[d][e] = (ans[d][e+1] - average[e]) / n;
		}
	}

	return x;

}
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
//vector<vector<double>> trasponer(vector<vector<double>> matrix, int n, int m);
vector<vector<double>> trasponer(vector<vector<double>> matrix);
vector<vector<double>> multiply(vector<vector<double>> x, vector<vector<double>> xt);

// int main(int argc, char * argv[]){

// //ESTO ESTA COMENTADO PORQUE NO ME ARMÉ NINGUN CASO COMPLETO DE INPUT, SIN EMBARGO LA FUNCION QUE ABRE UN ARCHIVO QUE TIENE VECTORES DE IMÁGENES Y LOS PARSEA ANDA
// 	string inputPath, outputPath; 
// 	 int metodo; 
// 	 cout << argc << endl;
// 	 if (argc < 4){ 
// 	 	cout << "Input: ";
// 	 	cin >> inputPath;
// 	 	cout << "Output: ";
// 	 	cin >> outputPath; 
// 	 	cout << "Metodo (0|1|2): ";
// 	 	cin >> metodo;
// 	 } else {
// 	 	inputPath = argv[1];
// 	 	outputPath = argv[2];
// 	 	metodo = atoi(argv[3]);
// 	 	cout << "Input: " << inputPath << endl;
// 	 	cout << "Output: " << outputPath << endl;
// 	 	cout << "Metodo: " << metodo << endl;
// 	 	if (!(metodo == 0 || metodo == 1 || metodo == 2))
// 	 		return 1;
// 	 }

// 	 ifstream input;
// 	 ofstream output;
// 	 input.open(inputPath);
// 	 output.open(outputPath);

// 	 string train;
// 	 string test;
// 	 int kappa;
// 	 int alpha;
// 	 int gamma;
// 	 int crossK;

// 	// Referencia a los archivos
// 	 input >> train;
// 	 test = train + "/test.csv";
// 	 train += "/train.csv";
	 
// 	// Variables de tuneo
// 	 input >> kappa;
// 	 input >> alpha;
// 	 input >> gamma;
// 	 input >> crossK;

// 	// Matriz de bools para ver cuales son los casos de train y test sobre train.csv
// 	// peligroso en memoria
// 	 bool partitions[crossK][42000];

// 	 for(int i = 0; i < crossK; i++){
// 	 	for(int j = 0; j < 42000; j++)
// 	 		input >> partitions[i][j];
// 	 }

// 	/*cout << "enter funcion" << endl;
// 	string train;
// 	cin >> train;*/

// 	int K = 42000;
// 	//trasformamos train en una matriz donde cada fila tiene el label del digito en la primer columna y 784 columnas más con los pixels
// 	// char * quizas sea mejor
// 	int* ans[K];
// 	for(int i = 0; i < K; i++){
// 		ans[i] = new int[785];
// 	}
// 	trainMatrix(train, ans, K);

// 	vector<vector<double>> images = toImageVector(ans, K);

// 	vector<vector<double>> x = toX(ans, K);

// 	vector<vector<double>> xt = trasponer(x);

// 	vector<vector<double>> xtx = multiply(x, xt);

// 	// vector<vector<double>> tcpca = characteristic_transformation(, ans);

// 	// for(int y = 0; y < K; y++){
// 	// 	for(int z = 0; z < 784; z++){
// 	// 		cout << x[y][z] << " ";
// 	// 	}
// 	// 	cout << endl << endl;
// 	// }


// }

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

vector<vector<double>> toImageVector(int** matrix, int K){
	vector<vector<double>> ans (K, vector<double> (784));

	for(int i = 0; i < K; i++){
		for(int j = 0; j < 784; j++){
			ans[i][j] = matrix[i][j + 1];
		}
	}

	return ans;

}

//vector<vector<double>> trasponer(vector<vector<double>> matrix, int n, int m){
vector<vector<double>> trasponer(vector<vector<double>> matrix){
	size_t n = matrix.size();
	size_t m = (matrix[0]).size();

	vector<vector<double>> ans (m, vector<double> (n, 0));

	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			ans[j][i] = matrix[i][j];
		}
	}

	return ans;
}

vector<vector<double> > X_K(const int ** const ans, const int K, const bool ** const partition){
	// K es la linea de partition a tener en cuenta
	int image_size = 784;
	int db_size = 42000;
	int count_train= 0;
	double average[image_size];
	for (int i = 0; i < image_size; i++){
		average[i] = 0.0;
	}
	for (int i = 0; i < image_size; i++){
		for (int j = 0; j < db_size; j++){
			if (partition[K][j] == true){
				average[i] += (double) ans[j][i+1];
				++count_train;
			}
		}
	}
	for (int i = 0; i < image_size; i++){
		average[i] /= (double) count_train;
	}
	vector<vector<double>> x (count_train, vector<double> (image_size,0));
	int added = 0;
	for (int j = 0; j < image_size; j++){
		for (int i = 0; i < db_size && added < count_train; i++){
			if (partition[K][added] == true){
				// j+1 para descartar el label
				x[added][j] = ans[i][j+1] - average[j];
				++added;
			}
		}
	}
	return x;
	// Devuelve x
	// n es x.size()
	// M_K = 1/(n-1) * (x' * x) 
	// o tambien:
	// M_K = (x' * x) / ((x.size()) - 1)
	
}

vector<vector<double> > PCA_M_K(vector<vector<double> > X_K){
	// Asumiendo X_K correcta, multiply correcta y trasponer correcta
	vector<vector<double> > M;
	//M = multiply(trasponer(X_K, X_K.size(), (X_K[0]).size), X_K);
	M = multiply(trasponer(X_K), X_K);
	int n = X_K.size() - 1;
	for (int i = 0; i < M.size(); i++){
		for (int j = 0; j < M.size(); j++){
			M[i][j] /= (double) n;
		}
	}

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

vector<vector<double>> multiply(vector<vector<double>> x, vector<vector<double>> xt){
	int n = x.size();
	int m = x[0].size();

	vector<vector<double>> ans (m, vector<double> (m, 0));

	for(int i = 0; i < m; i++){
		for(int j = 0; j < m; j++){
			for(int k = 0; k < n; k++){
				ans[i][j] += xt[i][k] * x[k][j];
			}
		}
		cout << "una linea menos: " << i << endl;
	}

	return ans;

}

vector<vector<double>> characteristic_transformation(vector<vector<double>> eigenvectors, vector<vector<double>> images){
	int n = images.size();
	int alpha = eigenvectors.size();

	vector<vector<double>> ans (n, vector<double> (alpha, 0));

	for(int i = 0; i < n; i++){
		for(int j = 0; j < alpha; j++){
			for(int a = 0; a < 784; a++){
				ans[i][j] += images[i][a] * eigenvectors[j][a];
			}
		}
	}

	return ans;

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

vector<vector<double> > xxt(vector<double> &v){
	vector<vector<double> > sol (v.size(), vector<double> (v));
	for (int i = 0; i < v.size(); ++i){
		for (int j = 0; j < v.size(); ++j)
			sol[i][j]*=v[i];
	}
	return sol;
}

void matSub(vector<vector<double> > &a, vector<vector<double> > &b){
	for (int i = 0; i < a.size(); ++i){
		for (int j = 0; j < a[j].size(); ++j)
			a[i][j] -= b[i][j];
	}
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
		b.push_back((double)(rand() % 1009));
	while(n>0){
		normalizar(b);
		b = mult(a, b);
		n--;
	}
	return b;
}

void multConst(vector<vector<double> > &a, double n){
    for (int i = 0; i < a.size(); ++i){
        for (int j = 0; j < a.size(); ++j)
            a[i][j]*=n;
    }
}
 
vector<vector<double> > deflate(vector<vector<double> > mat, int alpha){
    vector<vector<double> > sol (alpha, vector<double> (mat.size()));
    vector<vector<double> > transp (mat.size(), vector<double> (mat.size()));
    for (int i = 0; i < alpha; ++i)
    {
        std::vector<double> autov = pIteration(mat, 2048);
        double norma = norm(autov);
        normalizar(autov);
        for(int o = 0; o < mat.size(); o++){
        	cout << " " << autov[o];
        }
        sol[i] = autov;
        transp = xxt(autov);
        multConst(transp, norma);
        cout << norma << endl;
        matSub(mat, transp);
    }
    return sol;
}
 
vector<vector<double>> pls(vector<vector<double>> x, vector<vector<double>> y, int gama) {
    vector<vector<double>> w(x.size());
    for (int i = 0; i<gama; ++i) {
        //vector<vector<double>> m_i = multiply(multiply(multiply(trasponer(x, x.size(), x[0].size()), y), trasponer(y, y.size(), y[0].size())), x);
        vector<vector<double>> m_i = multiply(multiply(multiply(trasponer(x), y), trasponer(y)), x);
        w[i] = pIteration(m_i, 100);
        normalizar(w[i]);
        vector<double> t_i = mult(x, w[i]);
        normalizar(t_i);
        vector<vector<double>> ttt = xxt(t_i);
        vector<vector<double>> xt = multiply(ttt, x);
        matSub(x, xt);
        vector<vector<double>> yt = multiply(ttt, y);
        matSub(y, yt);
    }
    return w;
}

int main(){
 
    int n;
    cin >> n;
    std::vector<std::vector<double> > asd (n, std::vector<double> (n));
    double tmp;
 
 
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            cin >> tmp;
            asd[i][j] = tmp;
        }
    }
 
 
    std::vector<std::vector<double> > sol = deflate(asd, n);
 
    cout << endl << 3 << endl;
 
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            cout << sol[i][j] << " ";
        }
        cout << endl;
    }
 
    return 0;
}

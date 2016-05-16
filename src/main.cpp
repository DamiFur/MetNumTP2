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
#include <cassert>
// Para comparar con matlab
#include <iomanip>

using namespace std;

//const int db_size = 42000;
#define DB_SIZE 42000
#define IMAGE_SIZE 784
int image_size = IMAGE_SIZE;
int db_size = DB_SIZE;
const int K_DE_KNN = 10;

void trainMatrix(string train, vector<vector<double>>& ans, int K);
vector<vector<double>> toX(vector<vector<double>>& ans, int K);
//vector<vector<double>> trasponer(vector<vector<double>> matrix, int n, int m);
vector<vector<double>> trasponer(vector<vector<double>> matrix);
vector<vector<double>> multiply(vector<vector<double>> x, vector<vector<double>> y);
//vector<vector<double> > toX_K(const int ** const ans, const int K, const bool ** const partition);
vector<vector<double> > toX_K(vector<vector<double>>& ans, const int K, vector<vector<bool>>& partition);
vector<vector<double> > PCA_M_K(vector<vector<double> > X_K);
void print(vector<vector<int> >& M , ostream& out, const string caption, const char sep);
void print(vector<vector<double> >& M, ostream& out, const string caption, const char sep);
void print(int ** M, int m, int n, ostream& out, const string caption, const char sep);
vector<vector<double>> filtrarPartition(const vector<vector<double>>& x, const vector<vector<bool>>& partition, int k);
int knn(const vector<vector<double>>& train, const vector<double>& adivinar, int k);

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
		// Para testeos con db_size < 42000
		cout << "db_size: (-1 for default - " << db_size << " )";
		cin >> db_size;
		if (db_size == -1){
			db_size = DB_SIZE;
		}
		cout << "image_size: (-1 for default - " << image_size << " )";
		cin >> image_size;
		if (image_size == -1){
			image_size = IMAGE_SIZE;
		}
	} else {
		inputPath = argv[1];
		outputPath = argv[2];
		metodo = atoi(argv[3]);
		if (argc > 4) 
			db_size = atoi(argv[4]);
		if (argc > 5)
			image_size = atoi(argv[5]);
		cout << "Input: " << inputPath << endl;
		cout << "Output: " << outputPath << endl;
		cout << "Metodo: " << metodo << endl;
		cout << "db_size: " << db_size << endl;
		cout << "image_size: " << image_size << endl;
		if (!(metodo == 0 || metodo == 1 || metodo == 2 || metodo == 3)) // Metodo 3 = para pruebas - temporal
			return 1;
	}

	 ifstream input;
	 ofstream output;
	 ofstream reconocimiento;
	 reconocimiento.open("reconocimiento.txt");
	 input.open(inputPath);
	 output.open(outputPath);

	 string train;
	 string test;
	 int kappa;
	 int alpha;
	 int gamma;
	 int crossK;

	// Referencia a los archivos
	 input >> train;
	 test = train + "/test.csv";
	 train += "/train.csv";
	 
	// Variables de tuneo
	 input >> kappa;
	 input >> alpha;
	 input >> gamma;
	 input >> crossK;

	// Matriz de bools para ver cuales son los casos de train y test sobre train.csv
	vector<vector<bool>> partitions(crossK, vector<bool>(db_size));

	for(int i = 0; i < crossK; i++){
		for(int j = 0; j < db_size; j++){
			string str_bool;
			input >> str_bool;
			partitions[i][j] = (str_bool == "1");
		}
	}

	/*cout << "enter funcion" << endl;
	string train;
	cin >> train;*/

	int K = db_size;
	//trasformamos train en una matriz donde cada fila tiene el label del digito en la primer columna y 784 columnas más con los pixels
	// char * quizas sea mejor
	vector<vector<double>> ans(K, vector<double>(image_size + 1));

	trainMatrix(train, ans, K);

	if (metodo == 0) {
		vector<vector<double>> x = filtrarPartition(ans, partitions, K);
		double acertados = 0, total = 0;
		for(int i = 0; i<partitions.size(); ++i) {
			if (!partitions[K][i]) {
				int guess = knn(ans, x[i], K_DE_KNN);
				total++;
				if (guess == x[i][0]) {
					acertados++;
				}
			}
		}
		reconocimiento << acertados / total << endl;
	}else if (metodo == 3){
		vector<vector<double>> x = toX(ans, K);
		//const char sep = ';';
		//print(x, cout, caption, sep);
		//print(x, cout, "Matriz x", ';');
		ostringstream oss;

		print(ans, output, "Matriz Ans", ' ');
		print(x, output, "Matriz x", ' ');
		//vector<vector<double> > M;
		for (int i = 0; i < crossK; i++){
			vector<vector<double> > M = toX_K(ans, i, partitions);
			vector<vector<double> > N = trasponer(M);
			vector<vector<double> > NM = multiply(N, M);
			vector<vector<double> > P = PCA_M_K(M);
			string caption = "Matriz x_k" + to_string(i);
			print (M, output, caption, ' ');
			caption = "Matriz x_k" + to_string(i) + " traspuesta";
			print(N, output, caption, ' ');
			caption = "Matriz (x_k" + to_string(i) + ")t * x_k" + to_string(i);
			print(NM, output, caption, ' ');
			//print(P, output, caption, ' ');
			
		}
		return 0;
	} else {
		cout << "Aun no implementado" << endl;
		return 1;
	}

/*

	vector<vector<double>> xt = trasponer(x);

	vector<vector<double>> xtx = multiply(xt, x);
*/

	// for(int y = 0; y < K; y++){
	// 	for(int z = 0; z < 784; z++){
	// 		cout << x[y][z] << " ";
	// 	}
	// 	cout << endl << endl;
	// }


}

vector<vector<double>> filtrarPartition(const vector<vector<double>>& x, const vector<vector<bool>>& partition, int k) {
	// Filtra por partition y ademas convierte a double
	vector<vector<double>> ret;

	for (int i = 0; i < x.size(); ++i) {
		if (partition[k][i]) {
			ret.push_back(x[i]);
		}
	}
	return ret;
}

void trainMatrix(string train, vector<vector<double>>& ans, int K){

	ifstream input;
	input.open(train);

	// //Eliminamos la primer fila que tiene los nombres de las columnas que no nos sirven
	string deleteFirstRow;
	getline(input, deleteFirstRow);
	string row;

	for(int i = 0; getline(input, row); i++){
		replace(row.begin(), row.end(), ',', ' ');
		stringstream ss;
		ss << row;
		// Con Labels
		for(int j = 0; j < image_size + 1; j++){
			ss >> ans[i][j];
		}
	}

}

//vector<vector<double>> trasponer(vector<vector<double>> matrix, int n, int m){
vector<vector<double>> trasponer(vector<vector<double>> matrix){

	int m = (matrix[0]).size();
	int n = matrix.size();
	vector<vector<double>> ans (m, vector<double> (n, 0));

	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			ans[i][j] = matrix[j][i];
		}
	}

	return ans;
}

vector<vector<double> > toX_K(vector<vector<double>>& ans, int K, vector<vector<bool>>& partition){
	// K es la linea de partition a tener en cuenta
	// partition, la matriz de bool
	//int image_size = 784;
	//int db_size = 42000;
	int count_train= 0;
	double average[image_size];
	for (int j = 0; j < image_size; j++){
		average[j] = 0.0;
	}

	// Las columnas de partition representan las filas de ans
	// Average lo tomamos como vector fila
	for (int i = 0; i < db_size ; i++){
		if (partition[K][i] == true){
			++count_train; 
			for (int j = 0; j < image_size; j++){
				average[j] += (double) ans[i][j+1];
			}
		}
	}

	// Sacamos el promedio sobre la cantidad de casos tenidos en cuenta
	for (int j = 0; j < image_size; j++){
		average[j] /= (double) count_train;
	}

	vector<vector<double>> x (count_train, vector<double> (image_size,0));
	int added = 0;
	for (int i = 0; i < db_size; i++){
		if (partition[K][i] == true){
			for (int j = 0; j < image_size; j++){
				// j+1 para descartar el label
				x[added][j] = ans[i][j+1] - average[j];
			}
			++added; 
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
	M = multiply(trasponer(X_K), X_K);
	// Dividir matriz por escalar
	int n = X_K.size() - 1;
	for (int j = 0; j < (M[0]).size(); j++){
		for (int i = 0; i < M.size(); i++){
			M[i][j] /= (double) n;
		}
	}
	return M;
}
vector<vector<double>> toX(vector<vector<double>>& ans, int K){

	double average[image_size];

	// Inicializar array "average"
	for(int a = 0; a < image_size; a++){
		average[a] = 0.0;
	}

//	for(int i = 0; i < image_size; i++){
//		for(int j = 0; j < K; j++){
//			average[i] += (double) ans[j][i+1];
//		}
//	}
	for(int j=0; j < image_size; j++)
		for (int i = 0; i < K; i++)
			average[j] += (double) ans[i][j+1]; // Skip label

	cout << "average: " << endl;
	for(int j = 0; j < image_size; j++){
		average[j] /= (double) K;
		cout << average[j] << endl;
	}

	vector<vector<double>> x (K, vector<double> (image_size, 0));
	// for(int c = 0; c < K; c++){
	// 	x[c] = new vector<double>();
	// }

	// Comentado dado que nos ahorramos un sqrt si lo hacemos al hacer x' * x
	//double n = sqrt(K - 1);
	for(int i = 0; i < K; i++){
		for(int j = 0; j < image_size; j++){
			//x[i][j] = (ans[i][j+1] - average[j]) / n;
			x[i][j] = (ans[i][j+1] - average[j]);
		}
	}
	return x;
}

vector<vector<double>> multiply(vector<vector<double>> x, vector<vector<double>> y){
	int m = x.size();
	int n = x[0].size();
	// Verificar compatibilidad de dimensiones
	assert(x[0].size() == y.size());

	vector<vector<double>> ans (m, vector<double> (m, 0));

	for(int i = 0; i < m; i++){
		for(int j = 0; j < m; j++){
			for(int k = 0; k < n; k++){
				ans[i][j] += x[i][k] * y[k][j];
			}
		}
		// cout << "una linea menos: " << i << endl;
	}

	return ans;

}

#define cuad(x) ((x)*(x))
double distancia(const vector<double>& v1, const vector<double>& v2) {
	// en v1[0] esta el label asi que hay que comparar v1[i+1] con v2[i]
	double ret = 0.0;
	for (int i = 0; i<v1.size(); ++i) {
		ret += cuad(v1[i+1]-v2[i]);
	}
	return sqrt(ret);
}

int knn(const vector<vector<double>>& train, const vector<double>& adivinar, int k) {

	set<pair<double, int>> dist_index;

	for (int i = 0; i<train.size(); ++i) {
		dist_index.insert(make_pair(distancia(train[i], adivinar), train[i][0]));
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
			sol[i][j]+=v[i];
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

vector<vector<double>> pls(vector<vector<double>> x, vector<vector<double>> y, int gama) {
	vector<vector<double>> w(x.size());
	for (int i = 0; i<gama; ++i) {
		vector<vector<double>> m_i = multiply(multiply(multiply(trasponer(x),y),trasponer(y)),x);
		//vector<vector<double>> m_i = multiply(x, multiply(trasponer(y), multiply(y, trasponer(x))));
		w[i] = pIteration(m_i, 100);
		normalizar(w[i]);
		vector<double> t_i = mult(x, w[i]);
		normalizar(t_i);
		vector<vector<double>> ttt = xxt(t_i);
		//vector<vector<double>> xt = multiply(ttt, x);
		vector<vector<double>> xt = multiply(x, ttt);
		matSub(x, xt);
		//vector<vector<double>> yt = multiply(ttt, y);
		vector<vector<double>> yt = multiply(y, ttt);
		matSub(y, yt);
	}
	return w;
}

void print(vector<vector<int> >& M  , ostream& out, const string caption = "<Empty caption>", const char sep = ' '){
	int m = (M[0]).size();
	int n = M.size();

	out << caption << endl;
	for (int i = 0; i < m; i++){
		for (int j=0; j < n; j++){
			out << M[i][j] << sep;
		}
		out << endl;
	}
	out << endl;
	out << endl;
}
void print(vector<vector<double> >& M , ostream& out, const string caption = "<Empty caption>", const char sep = ' '){
	int m = M.size();
	int n = (M[0]).size();

	out << caption << endl;
	for (int i = 0; i < m; i++){
		for (int j=0; j < n -1; j++){
			out << setprecision(5) << M[i][j] << sep;
		}
		out << setprecision(5) << M[i][n-1] << endl;
	}
	out << endl;
}

void print(int ** M, int m, int n, ostream& out, const string caption = "<Empty caption>", const char sep = ' '){
	out << caption << endl;
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n - 1; j++){
			out << setprecision(5) << M[i][j] << sep;
		}
		out << setprecision(5) << M[i][n-1] << endl;
	}
	out << endl;
}


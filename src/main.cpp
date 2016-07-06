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
// Tiempos
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;


using namespace std;

//const int db_size = 42000;
#define DB_SIZE 42000
#define IMAGE_SIZE 784
#define TEST_SIZE 28000
int image_size = IMAGE_SIZE;
int db_size = DB_SIZE;
const unsigned int K_DE_KNN = 10;
unsigned int test_size = TEST_SIZE;


void dividirMatriz(vector<vector<double> > &m, double c);
vector<vector<int>> toImageVector(vector<vector<int>> matrix);
vector<vector<double>> labelImg(vector<vector<double>> toLabel, vector<vector<int>> labels, int alpha);
vector<vector<double> > deflate(vector<vector<double> > &mat, unsigned int alpha, vector<double> &autovalores, ostream& debug = std::cout);
//vector<vector<double>> toX(vector<vector<double>>& ans, int K);
vector<vector<double>> characteristic_transformation(vector<vector<double>> eigenvectors, vector<vector<int>> images);
void trainMatrix(string train, vector<vector<int>>& ans, int K);
void testMatrix(string test, vector<vector<int>>& ans, int K);
vector<vector<double>> toX(vector<vector<int>>& imagenes, int db_size = DB_SIZE);
vector<vector<double>> pls(vector<vector<double>> x, vector<vector<double>> y, int gama, vector<double> &autovals, ostream& debug = cout);
void toY(vector<vector<double>>& matrix);
vector<vector<double>> trasponer(vector<vector<double>> matrix);
vector<vector<double>> multiply(vector<vector<double>> x, vector<vector<double>> y);
vector<vector<double> > toX_K(vector<vector<int>>& original, const unsigned int K, vector<vector<bool>>& partition);
vector<vector<double> > PCA_M_K(vector<vector<double> > X_K);
vector<double> pIteration(vector<vector<double> > &a, int n, double &e, ostream& debug = std::cout);
void print(vector<vector<int> >& M , ostream& out, const string caption = "<empty caption>", const char sep = ' ');
void print(vector<vector<double> >& M, ostream& out, const string caption = "<empty caption>", const char sep = ' ');
void print(int ** M, int m, int n, ostream& out, const string caption = "<empty caption>", const char sep = ' ');
void print(vector<double> & M, ostream& out, const string caption = "<empty caption>", const char sep = ' ');
void printAsRow(vector<double>& v, ostream& out, const string caption = "<empty caption>", const char sep = ' ');
vector<vector<int>> filtrarPartition(const vector<vector<int>>& x, const vector<vector<bool>>& partition, int k, bool b);
template<typename T>
int knn(const vector<vector<T>>& train, const vector<T>& adivinar, int k);
double precision(int t_pos[], int f_pos[], int size);
double precision_(int t_pos[], int f_pos[], double prec[], int size);
double recall(int t_pos[], int f_neg[], int size);
double recall_(int t_pos[], int f_neg[], double rec[], int size);
double f1_score(double prec, double rec);
double f1_score_(double prec[], double rec[], double f1[], int size);
void trainMatrixDouble(string train, vector<vector<double>>& ans, int K, bool conLabels = true);
vector<vector<double>> preY_K(vector<vector<int>> matrix, int part, vector<vector<bool>> partitions);
void inplace_matrix_mult_by_scalar(vector<vector<double> >& mat, double scalar);
void inplace_matrix_div_by_scalar(vector<vector<double> >& M, double scalar);

vector<vector<double>> fullPCA(vector<vector<int>>& ans, vector<double>& autovals, int alpha, vector<vector<bool>>& partition, int i, ostream& debug = cout); // i = -1: sin particiones
vector<vector<double>> fullPLS(vector<vector<int>>& ans, vector<double>& autovals, int gamma, vector<vector<bool>>& partition, int i, ostream& debug = cout); // i = -1: sin particiones
void acum_metrics(int t_pos[10], int f_pos[10], int f_neg[10], unsigned int& acertados, int guess, int label);
void gen_metrics(int t_pos[10], int f_pos[10], int f_neg[10] , unsigned int p_acertados, unsigned int p_total, ostream& main_out, ostream& secondary_out = cout);

int main(int argc, char * argv[]){

	string inputPath, outputPath; 
	int metodo; 
	ofstream debug;
	debug.open("debug.out");

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
		if (!(metodo == 0 || metodo == 1 || metodo == 2 || metodo == 3 || metodo == 4)) // Metodo 3 = para pruebas - temporal
			return 1;
	}

	// Archivo con parametros y crossK particiones
	ifstream input; 
	input.open(inputPath);

	// Archivo de salida para autovalores
	ofstream output;
	output.open(outputPath);
	
	// Archivo de salida para kaggle
	string kaggle_s (outputPath);
	kaggle_s.append(".csv");

	// Archivo de salida para hit-rate y otras metricas 
	string metricas_s (outputPath);
	metricas_s.append(".mtx");
	string vmetricas_s(outputPath);
	vmetricas_s.append(".vmtx");

	string train_s;
	string test_s;
	int kappa;
	int alpha;
	int gamma;
	int crossK;


	// Referencia a los archivos
	input >> train_s;
	test_s = train_s + "test.csv";
	train_s += "train.csv";
	 
	// Variables de tuneo
	input >> kappa;
	input >> alpha;
	input >> gamma;
	input >> crossK;
	// Mantenemos input abierto para cargar las particiones

	if (crossK == 0){ // Test = test.csv
		// En este caso ya no necesitamos input abierto
		input.close();
		// Generacion archivo kaggle
		ofstream kaggle;
		kaggle.open(kaggle_s);
		// Levanto las imagenes
		vector<vector<int>> Train (db_size, vector<int>(image_size + 1));
		vector<vector<int>> Test (test_size, vector<int>(image_size));
    	trainMatrix(train_s, Train, db_size);
		testMatrix(test_s, Test, test_size);
		// Autovalores de PCA
		vector<vector<bool>> nil (1, vector<bool>(1, false));
		vector<double> PCA_evals;
		PCA_evals.reserve(alpha);
		vector<vector<double> > PCA_evec = fullPCA(Train, PCA_evals, alpha, nil, -1, debug);
		// Autovalores de PLS-DA
		vector<double> PLS_evals;
		PLS_evals.reserve(gamma);
		vector<vector<double> > PLS_evec = fullPLS(Train, PLS_evals, gamma, nil, -1, debug);
		// Escritura Autovalores en output
		for (int i = 0; i < alpha; ++i){
			output << PCA_evals[i] << endl;
		}
		for (int i = 0; i < gamma; ++i){
			output << PLS_evals[i] << endl;
		}
		output.close();

		// Aplicacion del metodo
		kaggle << "ImageId,Label" << endl;
		if (metodo == 0){ // knn
			for (unsigned int i = 0; i < Test.size(); ++i){
				// knn esta diseñando para que trabaje tanto con tests reales como particionados
				kaggle << i+1 << "," << knn(Train, Test[i], kappa) << endl;
			}
		} else if (metodo == 1){
			vector<vector<int>> trainImg = toImageVector(Train);
			vector<vector<double>> tcpca_train = characteristic_transformation(PCA_evec, trainImg);
			vector<vector<double>> tcpca_test = characteristic_transformation(PCA_evec, Test);
			vector<vector<double>> train_labeled = labelImg(tcpca_train, Train, alpha);
			//vector<vector<double>> testLabeled = labelImg(tcpca_test, Test, alpha);
			for (unsigned int i = 0; i < Test.size(); ++i){
				kaggle << i+1 << "," << knn(train_labeled, tcpca_test[i], kappa) << endl;
			}
		} else if (metodo == 2){
			vector<vector<int>> trainImg = toImageVector(Train);
			vector<vector<double>> tcpls_train = characteristic_transformation(PLS_evec, trainImg);
			vector<vector<double>> tcpls_test = characteristic_transformation(PLS_evec, Test);
			vector<vector<double>> train_labeled = labelImg(tcpls_train, Train, gamma);
			//vector<vector<double>> testLabeled = labelImg(tcpls_test, Test, gamma);
			for (unsigned int i = 0; i < Test.size(); ++i){
				kaggle << i+1 << "," << knn(train_labeled, tcpls_test[i], kappa) << endl;
			}
		}
		kaggle.close();
	} else { // Tests provienen de particionar train.csv
		ofstream metricas;
		metricas.open(metricas_s);
		ofstream vmetricas;
		vmetricas.open(vmetricas_s);

		int acertados_tot = 0, total_tot = 0;
		// Levanto partition
		vector<vector<bool>> partitions(crossK, vector<bool>(db_size));
		for (int i = 0; i < crossK; ++i){
			for (int j = 0; j < db_size; ++j){
				string str_bool;
				input >> str_bool;
				partitions[i][j] = (str_bool == "1");
			}
		}
		input.close();
		// Levanto imagenes
		vector<vector<int>> Train (db_size, vector<int>(image_size + 1));
    	trainMatrix(train_s, Train, db_size);

		// Itero sobre las particiones
		for (unsigned int i = 0; i < partitions.size(); ++i){
			unsigned int p_acertados = 0, p_total = 0;
			int t_pos[10] = {0}, f_pos[10] = {0}, f_neg[10] = {0};
			vector<vector<int>> Train_i = filtrarPartition(Train, partitions, i, true);
			vector<vector<int>> Test_i = filtrarPartition(Train, partitions, i, false);

			// Calcular autovals de los dos metodos
			// PCA
			vector<double> PCA_evals;
			PCA_evals.reserve(alpha);
			vector<vector<double> > PCA_evec = fullPCA(Train, PCA_evals, alpha, partitions, i, debug);
			// PLS-DA
			vector<double> PLS_evals;
			PLS_evals.reserve(gamma);
			vector<vector<double> > PLS_evec = fullPLS(Train, PLS_evals, gamma, partitions, i, debug);
			// Escritura Autovalores en output
			for (int j = 0; j < alpha; ++j){
				output << PCA_evals[j] << endl;
			} 
			for (int j = 0; j < gamma; ++j){ 
				output << PLS_evals[j] << endl; 
			}
			// mantengo output abierto para siguiente iteracion

			// aplicacion del metodo para la particion actual
			if (metodo == 0){
				for (p_total = 0; p_total < Test_i.size(); p_total++){ // itera sobre los vectores de "Test_i" para particion actual
					// knn esta diseñando para que trabaje tanto con tests reales como particionados (con o sin label)
					int guess = knn(Train_i, Test_i[p_total], kappa);
					acum_metrics(t_pos, f_pos, f_neg, p_acertados, guess, Test_i[p_total][0]);
				}
			} else if (metodo == 1){
				vector<vector<int>> trainImg = toImageVector(Train_i);
				vector<vector<int>> testImg = toImageVector(Test_i);
				vector<vector<double>> tcpca_train = characteristic_transformation(PCA_evec, trainImg);
				vector<vector<double>> tcpca_test = characteristic_transformation(PCA_evec, testImg);
				vector<vector<double>> train_labeled = labelImg(tcpca_train, Train_i, alpha);
				vector<vector<double>> test_labeled = labelImg(tcpca_test, Test_i, alpha); // No hace falta
				for (p_total = 0; p_total < Test_i.size(); ++p_total){
					int guess = knn(train_labeled, test_labeled[p_total], kappa);
					acum_metrics(t_pos, f_pos, f_neg, p_acertados, guess, Test_i[p_total][0]);
				}
			} else if (metodo == 2){
				vector<vector<int>> trainImg = toImageVector(Train_i);
				vector<vector<int>> testImg = toImageVector(Test_i);
				vector<vector<double>> tcpls_train = characteristic_transformation(PLS_evec, trainImg);
				vector<vector<double>> tcpls_test = characteristic_transformation(PLS_evec, testImg);
				vector<vector<double>> train_labeled = labelImg(tcpls_train, Train_i, gamma);
				vector<vector<double>> test_labeled = labelImg(tcpls_test, Test_i, gamma);
				for (p_total = 0; p_total < Test_i.size(); ++p_total){
					int guess = knn(train_labeled, test_labeled[p_total], kappa);
					acum_metrics(t_pos, f_pos, f_neg, p_acertados, guess, Test_i[p_total][0]);
				}
			}
			// genera y escribe metricas
			gen_metrics(t_pos, f_pos, f_neg, p_acertados, p_total, metricas, vmetricas);
		}
		output.close();
		metricas.close();
		vmetricas.close();
	}

	debug.close();
	return 0;
	/*
	*/
	
}


vector<vector<int>> filtrarPartition(const vector<vector<int>>& x, const vector<vector<bool>>& partition, int k, bool b) {
	// Filtra por partition y ademas convierte a double
	vector<vector<int>> ret;

	for (unsigned int i = 0; i < x.size(); ++i) {
		if (partition[k][i] == b) {
			ret.push_back(x[i]);
		}
	}
	return ret;
}

void trainMatrix(string train, vector<vector<int>>& ans, int K){

	ifstream input;
	input.open(train);

	// //Eliminamos la primer fila que tiene los nombres de las columnas que no nos sirven
	string deleteFirstRow;
	getline(input, deleteFirstRow);
	string row;

	for(unsigned int i = 0; getline(input, row); i++){
		replace(row.begin(), row.end(), ',', ' ');
		stringstream ss;
		ss << row;
		// Con Labels
		for(int j = 0; j < image_size + 1; j++){
			ss >> ans[i][j];
		}
	}

}

void testMatrix(string test, vector<vector<int>>& ans, int i){

    ifstream input;
    input.open(test);

    // //Eliminamos la primer fila que tiene los nombres de las columnas que no nos sirven
    string deleteFirstRow;
    getline(input, deleteFirstRow);
    string row;

    for(unsigned int i = 0; getline(input, row); i++){
        replace(row.begin(), row.end(), ',', ' ');
        stringstream ss;
        ss << row;
        for(int j = 0; j < image_size; j++){
            ss >> ans[i][j];
        }
    }

}

void trainMatrixDouble(string train, vector<vector<double>>& ans, int image_size, bool conLabels){

	ifstream input;
	input.open(train);

	string row;

	for(int i = 0; getline(input, row); i++){
		replace(row.begin(), row.end(), ',', ' ');
		stringstream ss;
		ss << row;
		if (conLabels){
		// Con Labels
			for(int j = 0; j < image_size + 1; j++){
				ss >> ans[i][j];
			}
		} else {
			for(int j = 0; j < image_size + 1; j++){
				ss >> ans[i][j+1];
			}
		}		
	}
	input.close();
}

vector<vector<int>> toImageVector(vector<vector<int>> matrix){
	unsigned int cantImag = matrix.size();
	vector<vector<int>> ans (cantImag, vector<int> (image_size));

	for(unsigned int i = 0; i < cantImag; i++){
		for(int j = 0; j < image_size; j++){
			ans[i][j] = matrix[i][j + 1];
		}
	}

	return ans;

}

// Agarra matriz "toLabel", y matriz "labels" devuelve la matriz "toLabel" con los labels
// Si no la deserdeno puede hacerse con un vector de labels
vector<vector<double>> labelImg(vector<vector<double>> toLabel, vector<vector<int>> labels, int alpha){
	int s = toLabel.size();
	vector<vector<double>> resp (s, vector<double> (alpha + 1));

	for(int i = 0; i < s; i++){
		resp[i][0] = labels[i][0];
		for(int j = 1; j < alpha + 1; j++){
			resp[i][j] = toLabel[i][j - 1];
		}
	}

	return resp;

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

// filtra "original" por medio de "partition" y lo convierte a double centrandolo en X_0 (average)
// NO LO DEVUELVE DIVIDIDO POR sqrt(n-1)!!!
vector<vector<double> > toX_K(vector<vector<int>>& original, unsigned int K, vector<vector<bool>>& partition){
	// K es la linea de partition a tener en cuenta
	// partition, la matriz de bool
	//int image_size = 784;
	//int db_size = 42000;
	int count_train= 0;
	double average[image_size];
	for (int j = 0; j < image_size; j++){
		average[j] = 0;
	}

	// Las columnas de partition representan las filas de original
	// Average lo tomamos como vector fila
	for (int i = 0; i < db_size ; i++){
		if (partition[K][i] == true){
			++count_train; 
			for (int j = 0; j < image_size; j++){
				average[j] += (double) original[i][j+1];
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
				x[added][j] = original[i][j+1] - average[j];
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
	inplace_matrix_div_by_scalar(M, n);
	
	/*
	for (unsigned int j = 0; j < (M[0]).size(); j++){
		for (unsigned int i = 0; i < M.size(); i++){
			M[i][j] /= (double) n;
		}
	}
	*/
	return M;
}
vector<vector<double>> toX(vector<vector<int>>& ans, int db_size){

	double average[image_size];

	// Inicializar array "average" (lo tomamos como vector fila)
	for(int j = 0; j < image_size; j++){
		average[j] = 0.0;
	}

	for(int i=0; i < db_size; i++){
		for (int j = 0; j < image_size; j++){
			average[j] += (double) ans[i][j+1]; // Skip label			
		}
	}

	// Sacamos el promedio sobre todos los casos
	for(int j = 0; j < image_size; j++){
		average[j] /= (double) db_size;
	}

	vector<vector<double>> x (db_size, vector<double> (image_size, 0));
	// Comentado dado que nos ahorramos un sqrt si lo hacemos al hacer x' * x
	//double n = sqrt(K - 1);
	for(int i = 0; i < db_size; i++){
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
    int r = y[0].size();
    // Verificar compatibilidad de dimensiones
    assert(x[0].size() == y.size());

    vector<vector<double>> ans (m, vector<double> (r, 0));

// Paraleliza, no importa como... requiere flag de compilador -fopenmp
// Baja el calculo de 42 minutos a 35 para 768x42000 * 420000x768
#pragma omp parallel for
    for(int i = 0; i < m; i++){
        // itero por k antes que por j, por cuestiones de cache... baja de 35 minutos a 10 el calculo para 768x42000 * 42000x768
        for(int k = 0; k < n; k++){
            if (x[i][k] == 0){
            //if (!x[i][k]){
                continue;
            } else {
                for(int j = 0; j < r; j++){
                    //ans[i][j] += ((y[k][j] != 0) ? (x[i][k] * y[k][j]) : 0);
                    ans[i][j] += x[i][k] * y[k][j];
                }
            }
        }
    }


    return ans;

}

vector<vector<double>> characteristic_transformation(vector<vector<double>> eigenvectors, vector<vector<int>> images){
	unsigned int m = images.size(); 	// Cantidad imagenes
							// images tiene x_i como vectores fila
	unsigned int n = images[0].size();
	int alpha = eigenvectors.size(); // Dimensiones a considerar

	vector<vector<double>> dimages(m, vector<double> (n));
	for (unsigned int i = 0; i < m; ++i){
		for (unsigned int j = 0; j < n; ++j){
			dimages[i][j] = (double) images[i][j];
		}
	}
	vector<vector<double>> ans (m, vector<double> (alpha, 0));
	ans = multiply(dimages, trasponer(eigenvectors));
	/*
	for(int i = 0; i < n; i++){
		for(int j = 0; j < alpha; j++){
			for(int a = 0; a < 784; a++){
				ans[i][j] += images[i][a] * eigenvectors[j][a];
			}
		}
	}
	*/

	return ans;

}

#define cuad(x) ((x)*(x))
template<typename T>
int distancia(const vector<T>& v1, const vector<T>& v2) {
	// en v1[0] esta el label asi que hay que comparar v1[i+1] con v2[i]
        double ret = 0.0;
        // Compara los tamaños - 
        //      si v2 es mas chico ("uno" mas chico) que v1 (cuando v2 pertenece a test y v1 a train
	assert( (v2.size() == v1.size() ) || (v2.size() == (v1.size() - 1) ) );
        if (v2.size() == (v1.size() -1)){
                for (int i = 0; i < v2.size(); ++i) {
                        ret += cuad((double)(v1[i+1])- (double)(v2[i]));
                }
        } else {
        // Sino compara desde el indice 1 - v1 y v2 pertenecen a train
                for (int i = 1; i < v2.size(); ++i){
                        ret += cuad((double)(v1[i]) - (double)(v2[i]));
                }
        }
        return ret;
}

template<typename T>
int knn(const vector<vector<T>>& train, const vector<T>& adivinar, int k) {
	multiset<pair<int, int>> dist_index;

	for (int i = 0; i<train.size(); ++i) {
		dist_index.insert(make_pair(distancia(train[i], adivinar), train[i][0]));
		if (dist_index.size() > k) {
			auto ulti = dist_index.end();
			ulti--;
			dist_index.erase(ulti);
		}
	}

	int votos[10];
	for (int i = 0; i<10; ++i) votos[i] = 0;
	for (const auto& v : dist_index) {
		votos[v.second]++;
	}

	int ganador = 0;
	for (int i = 1; i<10; ++i) {
		if (votos[i] > votos[ganador]) {
			ganador = i;
		} else if (votos[i] == votos[ganador]) {
            // me fijo el que tenga el voto mas cercano
            for (const auto& v : dist_index) {
                if (v.second == i) {
                    ganador = i;
                    break;
                } else if (v.second == ganador) {
                    break;
                }
            }
        }
	}
	return ganador;
}

vector<double> tmult(vector<double> &b, vector<vector<double> > &a){
	vector<double> result (a[0].size(), 0);
	assert(b.size() == a.size());
	for (unsigned int i = 0; i < a[0].size(); ++i)
	{
		for (unsigned int j = 0; j < b.size(); ++j)
			result[i] += b[j]*a[j][i];
	}
	return result;
}

vector<double> mult(vector<vector<double> > &a, vector<double> &b){
	vector<double> result (a.size(), 0);
	assert(b.size() == a[0].size());
	for (unsigned int i = 0; i < a.size(); ++i)
	{
		for (unsigned int j = 0; j < b.size(); ++j)
			result[i] += b[j]*a[i][j];
	}
	return result;
}

vector<vector<double> > vectorMult(vector<double> &a, vector<double> &b){
	vector<vector<double> > sol (a.size(), vector<double> (b.size()));
	for (unsigned int i = 0; i < a.size(); ++i){
		for (unsigned int j = 0; j < b.size(); ++j)
			sol[i][j]= a[i]*b[j];
	}
	return sol;
}

vector<vector<double> > xxt(vector<double> &v){
	vector<vector<double> > sol (v.size(), vector<double> (v));
	for (unsigned int i = 0; i < v.size(); ++i){
		for (unsigned int j = 0; j < v.size(); ++j)
			sol[i][j]*=v[i];
	}
	return sol;
}

void matSub(vector<vector<double> > &a, vector<vector<double> > &b){
	unsigned int tam = a.size();
	unsigned int tam2 = a[0].size();
	for (unsigned int i = 0; i < tam; i++){
		for (unsigned int j = 0; j < tam2; ++j){
			a[i][j] -= b[i][j];		
		}
	}
}

double productoInterno(vector<double> &v, vector<double> &w){
    assert(v.size() == w.size());
    double sol = 0;
    for (unsigned int i = 0; i < v.size(); ++i)
        sol += v[i]*w[i];
    return sol;
}

double norm(vector<double> &b, int metodo = 2){
    // metodo 1 = norma 1
    // metodo 2 (default) = norma 2
    // metodo -1 = norma infinito

    assert(metodo == 2 || metodo == 1 || metodo == -1);
    if (metodo == 2){
        return sqrt(productoInterno(b, b));
    }
    if (metodo == 1){
        double res = 0;
        for (unsigned int i = 0; i < b.size(); i++){
            res += abs(b[i]);
        }
        return res;
    }
    if (metodo == -1){
        double max = 0;
        for (unsigned int i = 0; i < b.size(); i++){
            if (abs(b[i]) > max){
                max = abs(b[i]);
            }
        }
        return max;
    }
    return -1.0;
}



void normalizar(vector<double> &b){
	double norma = norm(b);
	for (unsigned int i = 0; i < b.size(); ++i){
		b[i] /= norma;		
	}
}

double prod(std::vector<double> &v1, std::vector<double> v2){
    double sol =0;
    for (unsigned int i = 0; i < v1.size(); ++i)
    {
        sol+=(v1[i]*v2[i]);
    }
    return sol;
}

vector<double> pIteration(vector<vector<double> > &A, int n, double &e, ostream& debug){
	// Declara e inicializa autovector de salida
    vector<double> v;
    v.reserve(A.size());
    srand (time(NULL));
    for (unsigned int i = 0; i < A.size(); ++i){
		v.push_back((double)(rand() % 1009));
    }
	// Inicializa autovalor para comparar
	//double e0 = norm(v);
	//int j=0;
    for(int i=0;i < n;i++){ // Al menos 300 iteraciones
        v = mult(A, v);
        //e = prod(v, mult(A,v));
 		//e /= productoInterno(v,v);
        e = norm(v);
        for (int l = 0; l < v.size(); ++l)
        	v[l] /= e;

        //normalizar(v);

		//double d = e - e0;
		//if (d<0.000000001 && d>-0.000000001) // resetea el contador de corte
		//	j++;
		//else // incrementa el contador de corte
	//		j=0;
	//	e0 = e; // Setea autovalor de referencia para siguiente iteracion
    }
    return v;
}

/*vector<double> pIteration(vector<vector<double> > &A, int n, double &e, ostream& debug){
	// Declara e inicializa autovector de salida
    vector<double> v;
    v.reserve(A.size());
    srand (time(NULL));
    for (unsigned int i = 0; i < A.size(); ++i){
		v.push_back((double)(rand() % 1009));
    }

	double e0; // Autovalor
	double d0 = 1000; // Distancia
	int d; // Distancia
	
	// Inicializa autovalor para comparar

	e0 = prod(v, mult(A,v));
	e0 /= productoInterno(v,v);

	// Itera
	int i=0, j=0;
    while(i < n && j < 300){ // Al menos 300 iteraciones
        vector<double> c = mult(A, v);
        normalizar(c);
        v = c; // Autovector en esta iteracion;

		e = prod(v, mult(A,v)); 
		e /= productoInterno(v,v); // Autovalor en esta iteracion

		d = abs(e - e0);
		if (d  > d0){ // resetea el contador de corte
			j = 0;
		} else if (d < 0.000001){ // incrementa el contador de corte
			++j;
		}
		d0 = d; // Setea distancia minima para siguiente iteracion
		e0 = e; // Setea autovalor de referencia para siguiente iteracion
		
        ++i;
    }

	// Trunca errores despreciables en las componentes del autovector
    for (unsigned int k = 0; k < v.size(); ++k)  {d
        if(v[k]<0.000001 && v[k]>(-0.000001))
            v[k]=0;
    }

	// No hace falta, lo calcula en cada iteracion
    //e = prod(v, mult(A,v));
    //e /= productoInterno(v, v);

	debug << i << " iteraciones" << endl;
    return v;
}*/

void multConst(vector<vector<double> > &a, double n){
	for (unsigned int i = 0; i < a.size(); ++i){
		for (unsigned int j = 0; j < a.size(); ++j)
			a[i][j]*=n;
	}
}

vector<vector<double> > deflate(vector<vector<double> > &mat, unsigned int alpha, vector<double> &autovalores, ostream& debug){
	vector<vector<double> > sol ;
	for (unsigned int i = 0; i < alpha; ++i)
	{
		double eigenvalue;
		debug << i << " eigenvalue" << endl;
		vector<double> autovect = pIteration(mat, 2000, eigenvalue, debug);
		sol.push_back(autovect);
		autovalores.push_back(eigenvalue);
		for (int i = 0; i < mat.size(); ++i){
			for (int j = 0; j < mat.size(); ++j){
				mat[i][j] -= autovect[i]*autovect[j]*eigenvalue;
			}
		}
		
		debug << eigenvalue << endl;
	}
	return sol;
}

vector<vector<double>> pls(vector<vector<double>> x, vector<vector<double>> y, int gama, vector<double> &autovals, ostream& debug) {
	debug << "Entro al pls" << endl;
	vector<vector<double>> w (gama);
	autovals.resize(gama);
	double eigenvalue; 
	for (int i = 0; i<gama; ++i) {
		vector<vector<double>> aux = multiply(trasponer(x), y);
		vector<vector<double>> m_i = multiply(aux,trasponer(aux));
		//vector<vector<double>> m_i = multiply(x, multiply(trasponer(y), multiply(y, trasponer(x))));
		w[i] = pIteration(m_i, 2000, eigenvalue, debug);
		autovals[i] = eigenvalue;
		// no vuelve normalizado de pIteration???
		normalizar(w[i]);
		vector<double> t_i = mult(x, w[i]);
		normalizar(t_i);
		vector<double> ttx = tmult(t_i, x);
		vector<vector<double> > xt = vectorMult(t_i, ttx);
		matSub(x, xt);
		vector<double> tty  = tmult(t_i, y);
		vector<vector<double> > ty  = vectorMult(t_i, tty);
		matSub(y, ty);
		//ttx  = tmult(t_i, y);
		//xt  = vectorMult(t_i, ttx);
		//matSub(y, xt);

	}
	debug << "salgo del pls" << endl;
	return w;

}

vector<vector<double>> preY_K(vector<vector<int>> matrix, int part, vector<vector<bool>> partitions){
	vector<double> aux (10, -1.0);
	vector<vector<double>> resp;
	int added = 0; 
	if (part != -1){ 
		for(unsigned int i = 0; i < matrix.size(); i++){ 
			if (partitions[part][i] == true){ 
				resp.push_back(vector<double> (aux)); 
				resp[added][matrix[i][0]] = 1.0; 
				added++; 
			} 
		}
	} else { // Sin particiones
		for (unsigned int i = 0; i < matrix.size(); i++){
			resp.push_back(vector<double> (aux));
			resp[added][matrix[i][0]] = 1.0;
			added++; 
		}
	}
	return resp;
}

void toY(vector<vector<double>>& matrix){
	int n = matrix.size();
	int m = 10;
	vector<int> average(m, 0.0);

	double sq = sqrt(n - 1);

	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			average[j] += (double)matrix[i][j];
		}
	}

	for(int j = 0; j < m; j++){
		average[j] /= n;
	}

	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			matrix[i][j] -= average[j];
			matrix[i][j] /= sq;
		}
	}
}

void print(vector<vector<int> >& M  , ostream& out, const string caption , const char sep){
	int m = M.size();
	int n = (M[0]).size();

	out << caption << endl;
	for (int i = 0; i < m; i++){
		for (int j=0; j < n -1; j++){
			out << M[i][j] << sep;
		}
		out << M[i][n-1] << endl;
	}
	// out << endl;
}

void print(vector<vector<double> >& M , ostream& out, const string caption , const char sep ){
	int m = M.size();
	int n = (M[0]).size();

	// out << caption << endl;
	for (int i = 0; i < m; i++){
		for (int j=0; j < n -1; j++){
			out << setprecision(5) << M[i][j] << sep;
		}
		out << setprecision(5) << M[i][n-1] << endl;
	}
	// out << endl;
}

void print(vector<double> & M , ostream& out, const string caption , const char sep){
	int m = M.size();

	for (int i = 0; i < m; i++){
		out << setprecision(5) << M[i] << endl;
	}
}

void printAsRow(vector<double>& v, ostream& out, const string caption, const char sep){
	int m = v.size();
	out << caption << endl;
	for (int i = 0; i < m -1; i++){
		out << setprecision(5) << v[i] << sep;
	}
	out << setprecision(5) << v[m-1] << endl;
}

void print(int ** M, int m, int n, ostream& out, const string caption , const char sep ){
	// out << caption << endl;
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n - 1; j++){
			out << setprecision(5) << M[i][j] << sep;
		}
		out << setprecision(5) << M[i][n-1] << endl;
	}
	// out << endl;
}

void dividirMatriz(vector<vector<double> > &m, double c){
	for (unsigned int i = 0; i < m.size(); ++i){
		for (unsigned int j = 0; j < m[0].size(); ++j)
			m[i][j] /= c;
	}
}


double precision_(int t_pos[], int f_pos[], double prec[], int size){
	
	double ret = 0;
	for (int i = 0; i < size; i++){
		// Para evitar los nan si da 0/0
		if (t_pos[i] == 0){
			prec[i] = 0;
		} else {
			// asegurado denominador != 0
			prec[i] = (double) (t_pos[i]) / ((double)(t_pos[i]) + (double)(f_pos[i]));
		}
		ret += prec[i];
	}
	ret /= (double) size;
	return ret;
}
double precision(int t_pos[], int f_pos[], int size){
	double ret = 0;
	for (int i = 0; i < size; i++){
		if (t_pos[i] != 0){
			ret += (double) (t_pos[i]) / ((double)(t_pos[i]) + (double)(f_pos[i]));
		}
	}
	ret /= (double) size;
	return ret;
}

double recall(int t_pos[], int f_neg[], int size){
	double ret = 0;
	for (int i = 0; i < size; i++){
		if (t_pos[i] != 0){
			ret += (double) (t_pos[i]) / ((double)(t_pos[i]) + (double)(f_neg[i]));
		}
	}
	ret /= (double) size;
	return ret;
}
double recall_(int t_pos[], int f_neg[], double rec[], int size){
	double ret = 0;
	for (int i = 0; i < size; i++){
		if (t_pos[i] == 0){
			rec[i] = 0;
		} else {
			rec[i] = (double) (t_pos[i]) / ((double)(t_pos[i]) + (double)(f_neg[i]));
		}
		ret += rec[i];
	}
	ret /= (double) size;
	return ret;
}
double f1_score(double prec, double rec){
	double ret = 0;
	if ( prec == 0 || rec == 0){
		ret = 0;
	} else {
		ret = ((2 * prec * rec)/(prec + rec));
	}
	return ret;
}
double f1_score_(double prec[], double rec[], double f1[], int size){
	double ret = 0;
	for (int i = 0; i < size; i++){
		if ( prec[i] == 0 || rec[i] == 0 ){
			f1[i] = 0;
		} else {
			f1[i] = ((2 * prec[i] * rec[i])/(prec[i] + rec[i]));
			ret += f1[i];
		}
	}
	ret /= size;
	return ret;
}

void inplace_matrix_mult_by_scalar(vector<vector<double> >& M, double scalar){
    int m = M.size();
    int n = (M[0]).size();

    int i, j;
    for (i = 0; i < m; ++i){
        for (j = 0; j < n; ++j){
            M[i][j] *= scalar;
        }
    }
}

void inplace_matrix_div_by_scalar(vector<vector<double> >& M, double scalar){
    int m = M.size();
    int n = (M[0]).size();

    int i, j;
    for (i = 0; i < m; ++i){
        for (j = 0; j < n; ++j){
            M[i][j] /= scalar;
        }
    }
}

//vector<vector<double>> fullPCA(vector<vector<int>>& ans, vector<vector<bool>>& partition, int i, vector<double> autovals, vector<vector<int>> train, vector<vector<int>> test, int alpha){
vector<vector<double>> fullPCA(vector<vector<int>>& ans, vector<double>& autovals, int alpha, vector<vector<bool>>& partition, int i, ostream& debug){

	if (i != -1){ 
		vector<vector<double>> x = toX_K(ans, i, partition); 
		vector<vector<double>> M = PCA_M_K(x); 
		vector<vector<double>> eigenvectors = deflate(M, alpha, autovals, debug);
		return eigenvectors;
	} else {
		vector<vector<double>> x = toX(ans);
		vector<vector<double>> M = PCA_M_K(x);
		vector<vector<double>> eigenvectors = deflate(M, alpha, autovals, debug);
		return eigenvectors;
	}

}

vector<vector<double>> fullPLS(vector<vector<int>>& ans, vector<double>& autovals, int gamma, vector<vector<bool>>& partition, int i, ostream& debug){
	if (i != -1){
		vector<vector<double>> X = toX_K(ans, i, partition);
		double factor = 1.0 / (sqrt(X.size()));
		inplace_matrix_mult_by_scalar(X, factor);
		
		vector<vector<double>> Y = preY_K(ans, i, partition);
		toY(Y);

		vector<vector<double>> Ws = pls(X, Y, gamma, autovals, debug);
		return Ws;
	} else {
		vector<vector<double>> X = toX(ans);
		double factor = 1.0 / (sqrt(X.size()));
		inplace_matrix_mult_by_scalar(X, factor);

		vector<vector<double>> Y = preY_K(ans, i, partition);
		toY(Y);

		vector<vector<double>> Ws = pls(X, Y, gamma, autovals, debug);
		return Ws;
	}
}

void acum_metrics(int t_pos[10], int f_pos[10], int f_neg[10], unsigned int& acertados, int guess, int label){
	if (guess == label){
		++t_pos[label]; 
		++acertados;
	} else {
		++f_pos[guess];
		++f_neg[label];
	}
}

void gen_metrics(int t_pos[10], int f_pos[10], int f_neg[10], unsigned int p_acertados, unsigned int p_total, ostream& main_out, ostream& secondary_out){
	double prec[10] = {0}, rec[10] = {0}, f1[10] = {0};
	double prec_res, rec_res, f1_res;
	double hitrate = (double) p_acertados / (double) p_total;

	prec_res = precision_(t_pos, f_pos, prec, 10);
	rec_res = recall_(t_pos, f_neg, rec, 10);
	f1_res = f1_score_(prec, rec, f1, 10);

	main_out << hitrate << "," << prec_res << "," << rec_res << "," << f1_res << endl;
	for (int i = 0; i < 10; i++){
		secondary_out << i << "," << prec[i] << "," << rec[i] << "," << f1[i] << endl;
	}
}

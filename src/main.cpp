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

vector<vector<int>> toImageVector(vector<vector<int>> matrix);
vector<vector<double>> labelImg(vector<vector<double>> toLabel, vector<vector<int>> labels, int alpha);
vector<vector<double> > deflate(vector<vector<double> > &mat, unsigned int alpha, vector<double> &autovalores);
vector<vector<double>> characteristic_transformation(vector<vector<double>> eigenvectors, vector<vector<int>> images);
void trainMatrix(string train, vector<vector<int>>& ans, int K);
void testMatrix(string test, vector<vector<int>>& ans);
vector<vector<double>> toX(vector<vector<int>>& imagenes, int db_size = DB_SIZE);
vector<vector<double>> pls(vector<vector<double>> x, vector<vector<double>> y, int gama, vector<double> &autovals);
void toY(vector<vector<double>>& matrix);
vector<vector<double>> trasponer(vector<vector<double>> matrix);
vector<vector<double>> multiply(vector<vector<double>> x, vector<vector<double>> y);
vector<vector<double> > toX_K(vector<vector<int>>& original, const unsigned int K, vector<vector<bool>>& partition);
vector<vector<double> > PCA_M_K(vector<vector<double> > X_K);
vector<double> pIteration(vector<vector<double> > &a, int n, double &e);
vector<vector<int>> filtrarPartition(const vector<vector<int>>& x, const vector<vector<bool>>& partition, int k, bool b);
template<typename T>
int knn(const vector<vector<T>>& train, const vector<T>& adivinar, int k);
double precision(int t_pos[], int f_pos[], int size);
double precision_(int t_pos[], int f_pos[], double prec[], int size);
double recall(int t_pos[], int f_neg[], int size);
double recall_(int t_pos[], int f_neg[], double rec[], int size);
double f1_score(double prec, double rec);
double f1_score_(double prec[], double rec[], double f1[], int size);
vector<vector<double>> preY_K(vector<vector<int>> matrix, int part, vector<vector<bool>> partitions);
void inplace_matrix_div_by_scalar(vector<vector<double> >& M, double scalar);

vector<vector<double>> fullPCA(vector<vector<int>>& ans, vector<double>& autovals, int alpha, vector<vector<bool>>& partition, int i); // i = -1: sin particiones
vector<vector<double>> fullPLS(vector<vector<int>>& ans, vector<double>& autovals, int gamma, vector<vector<bool>>& partition, int i); // i = -1: sin particiones
void acum_metrics(int t_pos[10], int f_pos[10], int f_neg[10], unsigned int& acertados, int guess, int label);
void gen_metrics(int t_pos[10], int f_pos[10], int f_neg[10] , unsigned int p_acertados, unsigned int p_total, ostream& main_out, ostream& secondary_out = cout);

int main(int argc, char * argv[]){

	string inputPath, outputPath; 
	int metodo; 

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
	//	cout << "Input: " << inputPath << endl;
	//	cout << "Output: " << outputPath << endl;
	//	cout << "Metodo: " << metodo << endl;
	//	cout << "db_size: " << db_size << endl;
	//	cout << "image_size: " << image_size << endl;
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
		testMatrix(test_s, Test);
		// Autovalores de PCA
		vector<vector<bool>> nil (1, vector<bool>(1, false));
		vector<double> PCA_evals;
		PCA_evals.reserve(alpha);
		vector<vector<double> > PCA_evec = fullPCA(Train, PCA_evals, alpha, nil, -1);
		// Autovalores de PLS-DA
		vector<double> PLS_evals;
		PLS_evals.reserve(gamma);
		vector<vector<double> > PLS_evec = fullPLS(Train, PLS_evals, gamma, nil, -1);
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

		vector<vector<int>> Train (db_size, vector<int>(image_size + 1));
    	trainMatrix(train_s, Train, db_size);
    	cout << "Archivo :" << outputPath << endl;
    	vector<double> PCA_evals;
    	PCA_evals.reserve(alpha);
		vector<vector<double> > PCA_evec;
		vector<double> PLS_evals;
		PLS_evals.reserve(gamma);
		vector<vector<double> > PLS_evec;
		vector<vector<int>> trainImg;
		vector<vector<int>> testImg;
    	
		// Itero sobre las particiones
		for (unsigned int i = 0; i < partitions.size(); ++i){
			printf("Calculando partition %i\n", i);
			unsigned int p_acertados = 0, p_total = 0;
			int t_pos[10] = {0}, f_pos[10] = {0}, f_neg[10] = {0};

			unsigned int p_acertados_pls = 0, p_total_pls = 0;
			int t_pos_pls[10] = {0}, f_pos_pls[10] = {0}, f_neg_pls[10] = {0};
			vector<vector<int>> Train_i = filtrarPartition(Train, partitions, i, true);
			vector<vector<int>> Test_i = filtrarPartition(Train, partitions, i, false);

			
			// Calcular autovals de los dos metodos, solo si el metodo es distinto
			// a 0
			if (metodo != 0) {
				// PCA
				auto t0 = Clock::now();
				PCA_evec = fullPCA(Train, PCA_evals, alpha, partitions, i);
				// PLS-DA
				auto t1 = Clock::now();
				PLS_evec = fullPLS(Train, PLS_evals, gamma, partitions, i);
				// Escritura Autovalores en output
				auto t2 = Clock::now();
				
				for (int j = 0; j < alpha; ++j){
					output << std::scientific << PCA_evals[j] << endl;
				}
				
				for (int j = 0; j < gamma; ++j){ 
					output << std::scientific << PLS_evals[j] << endl; 
				}
				
				cout << "PCA tardo: " << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count() << "s, PLS tardo: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << 's' <<endl;
				
				trainImg = toImageVector(Train_i);
				testImg = toImageVector(Test_i);
			}
			// mantengo output abierto para siguiente iteracion

			// aplicacion del metodo para la particion actual
			if (metodo == 0){
				auto t1 = Clock::now();
				for (p_total = 0; p_total < Test_i.size(); p_total++){ // itera sobre los vectores de "Test_i" para particion actual
					// knn esta diseñando para que trabaje tanto con tests reales como particionados (con o sin label)
					int guess = knn(Train_i, Test_i[p_total], kappa);
					acum_metrics(t_pos, f_pos, f_neg, p_acertados, guess, Test_i[p_total][0]);
				}
				auto t2 = Clock::now();
				std::cout << "Tiempo en correr knn: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " minutos" << std::endl;
			} else if (metodo == 1) {
				
				vector<vector<double>> tcpca_train = characteristic_transformation(PCA_evec, trainImg);
				vector<vector<double>> tcpca_test = characteristic_transformation(PCA_evec, testImg);
				vector<vector<double>> pca_train_labeled = labelImg(tcpca_train, Train_i, alpha);
				vector<vector<double>> pca_test_labeled = labelImg(tcpca_test, Test_i, alpha); // No hace falta
				auto t0 = Clock::now();
				for (p_total = 0; p_total < Test_i.size(); ++p_total){
					int guess = knn(pca_train_labeled, pca_test_labeled[p_total], kappa);
					acum_metrics(t_pos, f_pos, f_neg, p_acertados, guess, Test_i[p_total][0]);
				}
				auto t1 = Clock::now();
				cout << "kNN en PCA tardo: " << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count() << 's' << endl;

				gen_metrics(t_pos, f_pos, f_neg, p_acertados, p_total, metricas, vmetricas);
			}else {
				vector<vector<double>> tcpls_train = characteristic_transformation(PLS_evec, trainImg);
				vector<vector<double>> tcpls_test = characteristic_transformation(PLS_evec, testImg);
				vector<vector<double>> pls_train_labeled = labelImg(tcpls_train, Train_i, gamma);
				vector<vector<double>> pls_test_labeled = labelImg(tcpls_test, Test_i, gamma);
				auto t0 = Clock::now();
				for (p_total = 0; p_total < Test_i.size(); ++p_total){
					int guess = knn(pls_train_labeled, pls_test_labeled[p_total], kappa);
					acum_metrics(t_pos, f_pos, f_neg, p_acertados, guess, Test_i[p_total][0]);
				}
				auto t1 = Clock::now();
				cout << "kNN en PLS tardo: " << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count() << 's' << endl;

				gen_metrics(t_pos, f_pos, f_neg, p_acertados, p_total, metricas, vmetricas);
			}
			// genera y escribe metricas
			
		}
		output.close();
		metricas.close();
		vmetricas.close();
	}

	return 0;

}


vector<vector<int>> filtrarPartition(const vector<vector<int>>& x, const vector<vector<bool>>& partition, int k, bool b) {
	// Filtra por partition y ademas convierte a double
	vector<vector<int>> ret;
	ret.reserve(x.size());

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

void testMatrix(string test, vector<vector<int>>& ans){

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

vector<vector<int>> toImageVector(vector<vector<int>> matrix){
	unsigned int cantImag = matrix.size();
	vector<vector<int>> ans (cantImag, vector<int> (image_size));

	#pragma omp parallel for
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
	vector<vector<double>> ans (m, vector<double> (n));
	#pragma omp parallel for
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
		average[j] = 0.0;
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
	double n = (double)X_K.size() - 1;
	inplace_matrix_div_by_scalar(M, n);
	
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



	vector<vector<double>> x (db_size, vector<double> (image_size));
	// Comentado dado que nos ahorramos un sqrt si lo hacemos al hacer x' * x
	//double n = sqrt(K - 1);
	for(int i = 0; i < db_size; i++){
		for(int j = 0; j < image_size; j++){
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
	        // itero por k antes que por ,j por cuestiones de cache... baja de 35 minutos a 10 el calculo para 768x42000 * 42000x768
	        for(int k = 0; k < n; k++){
	            if (x[i][k] == 0){
	                continue;
	            } else {
	                for(int j = 0; j < r; j++){
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
	#pragma omp parallel for
		for (unsigned int i = 0; i < m; ++i){
			for (unsigned int j = 0; j < n; ++j){
				dimages[i][j] = (double) images[i][j];
			}
		}

	return multiply(dimages, trasponer(eigenvectors));
}

#define cuad(x) ((x)*(x))
template<typename T>
double distancia(const vector<T>& v1, const vector<T>& v2) {
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
	multiset<pair<double, int>> dist_index;

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
	#pragma omp parallel for
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
	#pragma omp parallel for
		for (unsigned int i = 0; i < a.size(); ++i)
		{
			for (unsigned int j = 0; j < b.size(); ++j)
				result[i] += b[j]*a[i][j];
		}
	return result;
}

double productoInterno(vector<double> &v, vector<double> &w){
    assert(v.size() == w.size());
    double sol = 0;
    for (unsigned int i = 0; i < v.size(); ++i)
        sol += v[i]*w[i];
    return sol;
}

inline double norm(vector<double> &b){
        return sqrt(productoInterno(b, b));
}

void normalizar(vector<double> &b){
	double norma = norm(b);
	for (unsigned int i = 0; i < b.size(); ++i){
		b[i] /= norma;		
	}
}

vector<double> pIteration(vector<vector<double> > &A, int n, double &e){
	// Declara e inicializa autovector de salida
    vector<double> v;
    v.reserve(A.size());
    srand (time(NULL));
    for (unsigned int i = 0; i < A.size(); ++i){
		v.push_back((double)(rand() % 1009));
    }
	// Inicializa autovalor para comparar
	e = norm(v);
    for(int j=0;n > 0 && j<350 ;n--){ // Al menos 350 iteraciones
    	double e0 = e; // Setea autovalor de la anterior iteracion
        v = mult(A, v);
        e = norm(v);
        for (int l = 0; l < v.size(); ++l)
        	v[l] /= e;

		double d = e - e0;
		if (d<0.000000001 && d>-0.000000001) // incrementa el contador de corte 
			j++;
		else // resetea el contador de corte
			j=0;
		e0 = e; // Setea autovalor de referencia para siguiente iteracion
    }
    return v;
}

vector<vector<double> > deflate(vector<vector<double> > &mat, unsigned int alpha, vector<double> &autovalores){
	vector<vector<double> > sol ;
	sol.reserve(alpha);
	vector<double> autovect;
	double eigenvalue;
	for (unsigned int i = 0; i < alpha; ++i)
	{
		autovect = pIteration(mat, 3000, eigenvalue);
		sol.push_back(autovect);
		autovalores[i] = eigenvalue;
		#pragma omp parallel for
			for (int i = 0; i < mat.size(); ++i){
				for (int j = 0; j < mat.size(); ++j){
					mat[i][j] -= autovect[i]*autovect[j]*eigenvalue;
				}
			}
	}
	return sol;
}

vector<vector<double>> pls(vector<vector<double>> x, vector<vector<double>> y, int gama, vector<double> &autovals) {

	vector<vector<double>> w (gama);
	autovals.resize(gama);
	double eigenvalue;
	vector<double> t_i;
	vector<double> ttx;
	vector<double> tty; 
	for (int i = 0; i<gama; ++i) {
		vector<vector<double>> aux = multiply(trasponer(x), y);
		vector<vector<double>> m_i = multiply(aux,trasponer(aux));

		w[i] = pIteration(m_i, 3000, eigenvalue);
		autovals[i] = eigenvalue;
		t_i = mult(x, w[i]);
		normalizar(t_i);
		ttx = tmult(t_i, x);
		#pragma omp parallel for
			for (int i = 0; i < x.size(); ++i){
				for (int j = 0; j < x[0].size(); ++j)
					x[i][j] -= t_i[i]*ttx[j];
			}
		tty = tmult(t_i, y);
		#pragma omp parallel for
			for (int i = 0; i < y.size(); ++i){
				for (int j = 0; j < y[0].size(); ++j)
					y[i][j] -= t_i[i]*tty[j];
			}
	}

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
	double n = matrix.size();
	int m = 10;
	vector<double> average(m, 0);

	double sq = sqrt(n - 1);

	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			average[j] += matrix[i][j];
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

void inplace_matrix_div_by_scalar(vector<vector<double> >& M, double scalar){
    int m = M.size();
    int n = (M[0]).size();

    #pragma omp parallel for
	    for (int i = 0; i < m; ++i){
	        for (int j = 0; j < n; ++j){
	            M[i][j] /= scalar;
	        }
	    }
}

vector<vector<double>> fullPCA(vector<vector<int>>& ans, vector<double>& autovals, int alpha, vector<vector<bool>>& partition, int i){

	if (i != -1){ 
		vector<vector<double>> x = toX_K(ans, i, partition); 
		vector<vector<double>> M = PCA_M_K(x); 
		vector<vector<double>> eigenvectors = deflate(M, alpha, autovals);
		return eigenvectors;
	} else {
		vector<vector<double>> x = toX(ans);
		vector<vector<double>> M = PCA_M_K(x);
		vector<vector<double>> eigenvectors = deflate(M, alpha, autovals);
		return eigenvectors;
	}

}

vector<vector<double>> fullPLS(vector<vector<int>>& ans, vector<double>& autovals, int gamma, vector<vector<bool>>& partition, int i){
	if (i != -1){
		vector<vector<double>> X = toX_K(ans, i, partition);
		
		inplace_matrix_div_by_scalar(X,sqrt(X.size()-1));

		vector<vector<double>> Y = preY_K(ans, i, partition);
		toY(Y);

		vector<vector<double>> Ws = pls(X, Y, gamma, autovals);
		return Ws;
	} else {
		vector<vector<double>> X = toX(ans);

		inplace_matrix_div_by_scalar(X,sqrt(X.size()-1));

		vector<vector<double>> Y = preY_K(ans, i, partition);
		toY(Y);

		vector<vector<double>> Ws = pls(X, Y, gamma, autovals);
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

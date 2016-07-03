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
int image_size = IMAGE_SIZE;
int db_size = DB_SIZE;
const int K_DE_KNN = 10;


vector<vector<int>> toImageVector(vector<vector<int>> matrix);
vector<vector<double>> labelImg(vector<vector<double>> toLabel, vector<vector<int>> labels, int alpha);
vector<vector<double> > deflate(vector<vector<double> > &mat, int alpha, vector<double> &autovalores);
vector<vector<double>> toX(vector<vector<double>>& ans, int K);
vector<vector<double>> characteristic_transformation(vector<vector<double>> eigenvectors, vector<vector<int>> images);
void trainMatrix(string train, vector<vector<int>>& ans, int K);
void testMatrix(string test, vector<vector<int>>& ans);
vector<vector<double>> toX(vector<vector<int>>& ans, int K);
vector<vector<double>> pls(vector<vector<double>> x, vector<vector<double>> y, int gama);
void toY(vector<vector<double>>& matrix);
//vector<vector<double>> trasponer(vector<vector<double>> matrix, int n, int m);
vector<vector<double>> trasponer(vector<vector<double>> matrix);
vector<vector<double>> multiply(vector<vector<double>> x, vector<vector<double>> y);
//vector<vector<double> > toX_K(const int ** const ans, const int K, const bool ** const partition);
vector<vector<double> > toX_K(vector<vector<int>>& original, const int K, vector<vector<bool>>& partition);
vector<vector<double> > PCA_M_K(vector<vector<double> > X_K);
vector<double> pIteration(vector<vector<double> > &a, int n);
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
vector<vector<double>> fullPCA(vector<vector<int>>& ans, vector<vector<bool>>& partition, int i, vector<double> autovals, vector<vector<int>> train, vector<vector<int>> test, int alpha);

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
		if (!(metodo == 0 || metodo == 1 || metodo == 2 || metodo == 3 || metodo == 4)) // Metodo 3 = para pruebas - temporal
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
	 test = train + "test.csv";
	 train += "train.csv";
	 
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

	ofstream ext;
	ext.open("test1.results.out");
	//trasformamos train en una matriz donde cada fila tiene el label del digito en la primer columna y 784 columnas más con los pixels
	// char * quizas sea mejor
	//vector<vector<double>> ans(K, vector<double>(image_size + 1));
	vector<vector<int>> ans(K, vector<int>(image_size + 1));

	cout << "Levantando train" << endl;
    auto t1 = Clock::now();
    trainMatrix(train, ans, K);
    auto t2 = Clock::now();
    cout << "Train cargado" << endl;
    std::cout << "Delta t2-t1: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " nanoseconds" << std::endl;

    // escribimos train
    ofstream temp;
    temp.open("./precalc/train");
    print(ans, temp);
    temp.close();

	if (metodo == 0) {
		double acertados = 0, total = 0;
		for(int i = 0; i<partitions.size(); ++i) { // i itera particiones 
			cout << "Comienza particion " << i << endl;
            t1 = Clock::now();
			double p_acertados = 0, p_total = 0;
			// Para los experimentos - verdaderos/falsos positivos/negativos
			int t_pos[10] = {0}, f_pos[10] = {0}, f_neg[10] = {0};
			vector<vector<int>> x = filtrarPartition(ans, partitions, i, true);
			vector<vector<int>> v = filtrarPartition(ans, partitions, i, false);
			for (int j = 0; j < v.size(); j++) { // j itera sobre los test de train (v)
				int guess = knn(x, v[j], kappa);
				p_total++;
				if (guess == v[j][0]) {
					p_acertados++;
					t_pos[v[j][0]]++;
				} else {
					f_pos[guess]++;
					f_neg[v[j][0]]++;
				}
			}
			
			t2 = Clock::now();
            cout << "Tiempo en aplicar kNN a la particion: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " nanoseconds" << endl;
			reconocimiento << "Particion " << i << ": ";
			reconocimiento << p_acertados / p_total << endl;
			acertados += p_acertados;
			total += p_total;

			double prec[10] = {0}, prec_res;
			double rec[10] = {0}, rec_res;
			double f1[10] = {0}, f1_res;

			prec_res = precision_(t_pos, f_pos, prec, 10);
			rec_res = recall_(t_pos, f_neg, rec, 10);
			f1_res = f1_score_(prec, rec, f1, 10);

			output << "Particion" << i << ":" << " ";
			output << prec_res << ' ' << rec_res << ' ' << f1_res << endl;
	
		}
		reconocimiento << acertados / total << endl;
	}else if (metodo == 3){
		// Calculamos el factor para dividir xtx y conseguir M
        double scalar = 1.0 / (db_size -1);
        cout << "Scalar = " << scalar << endl;

        // Calculamos x
        cout << "Calculando X" << endl;
            t1 = Clock::now();
        vector<vector<double>> x = toX(ans, K);
            t2 = Clock::now();
        cout << "X calculado" << endl;
            std::cout << "Delta t2-t1: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " nanoseconds" << std::endl;

        // Escribimos x
        temp.open("./precalc/x");
        print(x, temp);
        temp.close();

        // Calculamos x traspuesta
        cout << "Trasponiendo x" << endl;
            t1 = Clock::now();
        vector<vector<double> > xt = trasponer(x);
            t2 = Clock::now();
        cout << "X' calculado" << endl;
            std::cout << "Delta t2-t1: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " nanoseconds" << std::endl;

        // Escribimos x traspuesta
        temp.open("./precalc/xt");
        print(xt, temp);
        temp.close();

        // Calculamos x traspuesta * x
        cout << "multiplicando x' por x" << endl;
            t1 = Clock::now();
        vector<vector<double> > xtx = multiply(xt, x);
            t2 = Clock::now();
        cout << "x' * x calculado" << endl;
            std::cout << "Delta t2-t1: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " nanoseconds" << std::endl;

		 // Escribimos x traspuesta * x
        temp.open("./precalc/xtx");
        print(xtx, temp);
        temp.close();

        // Calculamos M
        cout << "Convirtiendo xtx en M (multiplicando por escalar)" << endl;
            t1 = Clock::now();
        inplace_matrix_mult_by_scalar(xtx, scalar);
            t2 = Clock::now();
        cout << "M calculado" << endl;
            std::cout << "Delta t2-t1: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " nanoseconds" << std::endl;

        // Escribimos M
        temp.open("./precalc/M");
        print(xtx, temp);
        temp.close();


		// Calculamos autovectores y autovalores de M 

 /*
		vector<vector<double>> xtx(K, vector<double>(image_size));
    		cout << "Levantando M" << endl;
        	t1 = Clock::now();
		trainMatrixDouble("./precalc/M", xtx, image_size, false);
        	t2 = Clock::now();
    		cout << "M cargado" << endl;
        	std::cout << "Delta t2-t1: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " nanoseconds" << std::endl;
		
		// Escribimos M'
		temp.open("./precalc/Mprima");
		print(xtx, temp);
		temp.close();
		
*/
		// return 1;

		int cant = 784; // a cambiar por alpha
		vector<double> autovalores;
		autovalores.reserve(cant);
		
        cout << "Calculando " << cant << " autovalores y autovectores" << endl;
            t1 = Clock::now();
		vector<vector<double> > autovec = deflate(xtx, cant, autovalores);
            t2 = Clock::now();
        cout << "Calculados los autovalores y autovectores" << endl;
            std::cout << "Delta t2-t1: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " nanoseconds" << std::endl;
		
		temp.open("./precalc/autovalores");
			for (int i = 0; i < cant; i++){
				temp << autovalores[i] << endl;
			}
		temp.close();
		
		
		
		
        /*
        ofstream out_xtx;
        out_xtx.open("precalc/xtx");
        cout << "Escribiendo la matriz M" << endl;
            t1 = Clock::now();
        print(xtx, out_xtx);
            t2 = Clock::now();
        cout << "M escrita" << endl;
            std::cout << "Delta t2-t1: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " nanoseconds" << std::endl;
        out_xtx.close();
        */

        // Vuelta anticipada
        return 1;

		/* Prueba Vieja
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
			caption = "Matriz x_k" + to_string(i) + " PCA - Matriz(M)";
			print(P, output, caption, ' ');

			
		}
		return 0;
		Fin Prueba vieja */
	} else if (metodo == 1){

	// 	//trasformamos train en una matriz donde cada fila tiene el label del digito en la primer columna y 784 columnas más con los pixels
		// char * quizas sea mejor
		ofstream expected;
		expected.open("guess_results");

		// vector<vector<int>> images = toImageVector(ans, K, 1, partitions);

		// vector<vector<double>> x = toX(ans, K);
		for (int i = 0; i < crossK; i++){

			vector<double> autovals;

			autovals.reserve(alpha);

			vector<vector<int>> train = filtrarPartition(ans, partitions, i, true);

			vector<vector<int>> test = filtrarPartition(ans, partitions, i, false);

			vector<vector<double>> eigenvectors = fullPCA(ans, partitions, i, autovals, train, test, alpha);

			vector<vector<int>> trainImg = toImageVector(train);

			vector<vector<int>> testImg = toImageVector(test);

			vector<vector<double>> tcpca_train = characteristic_transformation(eigenvectors, trainImg);

			vector<vector<double>> tcpca_test = characteristic_transformation(eigenvectors, testImg);

			vector<vector<double>> trainLabeled = labelImg(tcpca_train, train, alpha);

			vector<vector<double>> testLabeled = labelImg(tcpca_test, test, alpha);

			double p_acertados = 0, p_total = 0;
			// Para los experimentos - verdaderos/falsos positivos/negativos
			int t_pos[10] = {0}, f_pos[10] = {0}, f_neg[10] = {0};

			for (int j = 0; j < testLabeled.size(); j++) { // j itera sobre los test de train (v)
				int guess = knn(trainLabeled, testLabeled[j], kappa);
				p_total++;
				if (guess == testLabeled[j][0]) {
					cout << "Indice " << j << " acertado: " << guess << endl;
					ext << "Indice " << j << " acertado: " << guess << endl;
					p_acertados++;
					t_pos[(int)testLabeled[j][0]]++;
				} else {
					cout << "Indice " << j << " errado. Da: " << guess << " - Esperado: " << testLabeled[j][0] << endl;
					ext << "Indice " << j << " errado. Da: " << guess << " - Esperado: " << testLabeled[j][0] << endl;
					f_pos[guess]++;
					f_neg[(int)testLabeled[j][0]]++;
				}
			}

			// print(tcpca, ext, "", ';');

			// print(eigenvectors, expected, "", '\n');

			print(autovals, output, "", '\n');



		}
		// for(int y = 0; y < K; y++){
		// 	for(int z = 0; z < 784; z++){
		// 		cout << x[y][z] << " ";
		// 	}
		// 	cout << endl << endl;
		// }

		return 0;
	} else if (metodo == 2){

		// vector<vector<int>> ans(K, vector<int>(image_size + 1));

		// trainMatrix(train, ans, K);

		// vector<vector<int>> images = toImageVector(ans, K);

		for (int i = 0; i < crossK; i++){

			vector<vector<int>> train = filtrarPartition(ans, partitions, i, true);

			vector<vector<int>> test = filtrarPartition(ans, partitions, i, false);

			vector<vector<double>> X = toX_K(ans, i, partitions);

			vector<vector<double>> Y = preY_K(ans, i, partitions);

			toY(Y);

			vector<vector<double>> Ws = pls(X, Y, gamma);

			cout << "Salgo de Pls" << endl;

			vector<vector<int>> trainImg = toImageVector(train);

			vector<vector<int>> testImg = toImageVector(test);

			cout << "toImageVector" << endl;

			vector<vector<double>> tcpls_train = characteristic_transformation(Ws, trainImg);

			vector<vector<double>> tcpls_test = characteristic_transformation(Ws, testImg);

			cout << "transformacion caracteristica" << endl;

			vector<vector<double>> trainLabeled = labelImg(tcpls_train, train, gamma);

			vector<vector<double>> testLabeled = labelImg(tcpls_test, test, gamma);

			// print(autovals, output, "", '\n');

		}





	} else if (metodo == 4){
		vector<vector<double>> ans(K, vector<double>(image_size));

		vector<double> autoval;

		trainMatrixDouble(train, ans, K);

		vector<vector<double>> deflated = deflate(ans, 5, autoval);

		print(deflated, cout, "", '\n');


	}

/*

	vector<vector<double>> xt = trasponer(x);

	vector<vector<double>> xtx = multiply(xt, x);
*/






 }

vector<vector<int>> filtrarPartition(const vector<vector<int>>& x, const vector<vector<bool>>& partition, int k, bool b) {
	// Filtra por partition y ademas convierte a double
	vector<vector<int>> ret;

	for (int i = 0; i < x.size(); ++i) {
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

void testMatrix(string test, vector<vector<int>>& ans){

    ifstream input;
    input.open(test);

    // //Eliminamos la primer fila que tiene los nombres de las columnas que no nos sirven
    string deleteFirstRow;
    getline(input, deleteFirstRow);
    string row;

    for(int i = 0; getline(input, row); i++){
        replace(row.begin(), row.end(), ',', ' ');
        stringstream ss;
        ss << row;
        for(int j = 0; j < image_size; j++){
            ss >> ans[i][j+1];
        }
        // Ponemos un -1 donde deberia estar el label

        ans[i][0] = -1;
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
	int cantImag = matrix.size();
	vector<vector<int>> ans (cantImag, vector<int> (image_size));

	for(int i = 0; i < cantImag; i++){
		for(int j = 0; j < image_size; j++){
			ans[i][j] = matrix[i][j + 1];
		}
	}

	return ans;

}

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

vector<vector<double> > toX_K(vector<vector<int>>& original, int K, vector<vector<bool>>& partition){
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
	int n = X_K.size() - 1;
	for (int j = 0; j < (M[0]).size(); j++){
		for (int i = 0; i < M.size(); i++){
			M[i][j] /= (double) n;
		}
	}
	return M;
}
vector<vector<double>> toX(vector<vector<int>>& ans, int K){

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
	for(int j=0; j < image_size; j++){
		for (int i = 0; i < K; i++){
			average[j] += (double) ans[i][j+1]; // Skip label			
		}
	}


	for(int j = 0; j < image_size; j++){
		average[j] /= (double) K;
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
    int k = y[0].size();

    cout << m << " " << n << " " << k << endl;
    // Verificar compatibilidad de dimensiones
    assert(x[0].size() == y.size());

    vector<vector<double>> ans (m, vector<double> (k, 0));

// Paraleliza, no importa como... requiere flag de compilador -fopenmp
// Baja el calculo de 42 minutos a 35 para 768x42000 * 420000x768
//#pragma omp parallel for
   /* for(int i = 0; i < m; i++){
        // itero por k antes que por j, por cuestiones de cache... baja de 35 minutos a 10 el calculo para 768x42000 * 42000x768
        for(int k = 0; k < n; k++){
            if (x[i][k] == 0){
            //if (!x[i][k]){
                continue;
            } else {
                for(int j = 0; j < m; j++){
                    //ans[i][j] += ((y[k][j] != 0) ? (x[i][k] * y[k][j]) : 0);
                    ans[i][j] += x[i][k] * y[k][j];
                }
            }
        }
        // cout << "una linea menos: " << i << endl;
    }*/
     for (int i = 0; i < m; ++i){
     	for (int j = 0; j < k; ++j){
     		for (int h = 0; h < n; ++h){
     			ans[i][j] += x[i][h] * y[h][j]; 
     		}
     	}
     }

    return ans;

}

vector<vector<double>> characteristic_transformation(vector<vector<double>> eigenvectors, vector<vector<int>> images){
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
        return sqrt(ret);

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
		}
	}
	return ganador;
}

vector<double> mult(vector<vector<double> > &a, vector<double> &b){
	vector<double> result (a.size(), 0);
	assert(b.size() == a[0].size());
	for (int i = 0; i < a.size(); ++i)
	{
		for (int j = 0; j < b.size(); ++j)
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
	int tam = a.size();
	for (int i = 0; i < tam; i++){
		for (int j = 0; j < tam; ++j){
			a[i][j] -= b[i][j];		
		}
	}
}

double productoInterno(vector<double> &v, vector<double> &w){
    assert(v.size() == w.size());
    double sol = 0;
    for (int i = 0; i < v.size(); ++i)
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
        for (int i = 0; i < b.size(); i++){
            res += abs(b[i]);
        }
        return res;
    }
    if (metodo == -1){
        double max = 0;
        for (int i = 0; i < b.size(); i++){
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
	for (int i = 0; i < b.size(); ++i){
		b[i] /= norma;		
	}
}

double prod(std::vector<double> v1, std::vector<double> v2){
    double sol =0;
    for (int i = 0; i < v1.size(); ++i)
    {
        sol+=(v1[i]*v2[i]);
    }
    return sol;
}

vector<double> pIteration(vector<vector<double> > &a, int n, double &e){
    vector<double> b;
    b.reserve(a.size());
    srand (time(NULL));
    for (int i = 0; i < a.size(); ++i){
		b.push_back((double)(rand() % 1009));
    }
    while(n>0){
        vector<double> c = mult(a, b);
        normalizar(c);
        b = c;
        n--;
    }
    for (int i = 0; i < b.size(); ++i)
    {
        if(b[i]<0.000001 && b[i]>(-0.000001))
            b[i]=0;
    }
    e = prod(b, mult(a,b));
    e /= productoInterno(b, b);
    return b;
}

void multConst(vector<vector<double> > &a, double n){
	for (int i = 0; i < a.size(); ++i){
		for (int j = 0; j < a.size(); ++j)
			a[i][j]*=n;
	}
}

vector<vector<double> > deflate(vector<vector<double> > &mat, int alpha, vector<double> &autovalores){
	vector<vector<double> > sol ;
	// cout << "eigenvalues" << endl;
	for (int i = 0; i < alpha; ++i)
	{
		double eigenvalue;
		vector<double> autovect = pIteration(mat, 800, eigenvalue);
		vector<vector<double> > transp = xxt(autovect);
		sol.push_back(autovect);
		autovalores.push_back(eigenvalue);
		cout << eigenvalue << endl;
		multConst(transp, eigenvalue);
		matSub(mat, transp);

	}
	return sol;
}

vector<vector<double>> pls(vector<vector<double>> x, vector<vector<double>> y, int gama) {
	cout << "Entro al pls" << endl;
	vector<vector<double>> w(x.size());
	double eigenvalue; 
	for (int i = 0; i<gama; ++i) {
		cout << "mult 1 - " << i <<  endl;
		vector<vector<double>> aux = multiply(trasponer(x), y);
		cout << "mult 2 - " << i <<   endl;
		vector<vector<double>> m_i = multiply(aux,trasponer(aux));
		//vector<vector<double>> m_i = multiply(x, multiply(trasponer(y), multiply(y, trasponer(x))));
		w[i] = pIteration(m_i, 800, eigenvalue);
		normalizar(w[i]);
		cout << x.size() << " " <<x[0].size() << " " << w[i].size() << endl;
		vector<double> t_i = mult(x, w[i]);
		cout << "size" <<  t_i.size() << endl;
		normalizar(t_i);
		vector<vector<double>> ttt = xxt(t_i);
		//vector<vector<double>> xt = multiply(ttt, x);
		cout << "mult 3 - " << i << endl;
		vector<vector<double>> xt = multiply(ttt, x);
		matSub(x, xt);
		//vector<vector<double>> yt = multiply(ttt, y);
		cout << "mult 4 - " << i <<  endl;
		vector<vector<double>> yt = multiply(ttt, y);
		matSub(y, yt);
	}
	cout << "salgo del pls" << endl;
	return w;

}

vector<vector<double>> preY_K(vector<vector<int>> matrix, int part, vector<vector<bool>> partitions){
	vector<double> aux (10, -1.0);
	vector<vector<double>> resp;
	int added = 0;
	for(int i = 0; i < matrix.size(); i++){
		if (partitions[part][i] == true){
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

vector<vector<double>> fullPCA(vector<vector<int>>& ans, vector<vector<bool>>& partition, int i, vector<double> autovals, vector<vector<int>> train, vector<vector<int>> test, int alpha){

	vector<vector<double>> x = toX_K(ans, i, partition);

	vector<vector<double>> M = PCA_M_K(x);

	vector<vector<double>> eigenvectors = deflate(M, alpha, autovals);

	return eigenvectors;
}


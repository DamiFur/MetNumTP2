
% Archivos
output = 'testmin/test1_met3.out.ml'
train = '../datamin/train.csv'
input = 'testmin/test1.in'

% Lectura y armado matrices
% Matrices de base
Ans = dlmread(train, ',', 1, 0)
M = dlmread(train, ',', 1, 1)
m = mean(M)
AVG = repmat(m, 15,1)
X = M - AVG
In = dlmread(input, ' ', 1, 0)

% Celdas de matrices definitivas
Ic = logical(ones(1,10))
for i = 1 : 3
	Ir_i{i} = logical(In(i,:))
	M_i{i} = M(Ir_i{i}, Ic)
	m_i{i} = mean(M_i{i})
	AVG_i{i} = repmat(m_i{i}, size(M_i{i},1), 1)
	X_i{i} = M_i{i} - AVG_i{i}
end

% Escribir Header Ans
fid = fopen(output, 'wt')
fprintf(fid, 'Matriz Ans\n')
fclose(fid)
% Escribir ans
dlmwrite(output, Ans, '-append', 'delimiter', ' ')

% Escribir Header "x"
fid = fopen(output, 'at')
fprintf(fid, '\nMatriz x\n')
fclose(fid)
% Escribir x
dlmwrite(output, X, '-append', 'delimiter', ' ')

% Ciclar por las 3 particiones
for i = 1 : 3
	% Escrbir Header x_k<i>
	fid = fopen(output, 'at')
	fprintf(fid, '\nMatriz x_k%i\n', i-1)
	fclose(fid)
	
	% Escribir x_k<i>
	dlmwrite(output, X_i{i}, '-append', 'delimiter', ' ') 
	
	% Escribir Header x_k<i> traspuesta
	fid = fopen(output, 'at')
	fprintf(fid, '\nMatriz x_k%i traspuesta\n', i-1)
	fclose(fid)
		
	% Escribir x_k<i> traspuesta
	dlmwrite(output, (X_i{i})', '-append', 'delimiter', ' ') 

	% Escribir Header x_k<i> traspuesta * x_k<i>
	fid = fopen(output, 'at')
	fprintf(fid, '\nMatriz (x_k%i)t * x_k%i\n', i-1, i-1)
	fclose(fid)

	% Escribir x_k<i> traspuesta
	dlmwrite(output, ((X_i{i})' * X_i{i}), '-append', 'delimiter', ' ') 
end
% Una linea extra para el diff
fid = fopen(output, 'at')
fprintf(fid, '\n')
fclose(fid)

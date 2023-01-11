#include <iostream>
#include <vector>
#include <immintrin.h>
#include <random>
#include <chrono>
#include <cassert>

using namespace std;


double mean(vector<double> vals) {
	double sum = 0.0;
	for (auto val : vals) {
		sum += val;
	}
	return sum / vals.size();
}

void _vec_sum(vector<float>& C, const vector<float>& A, const vector<float>& B) {
	assert(A.size() == B.size() && A.size() == C.size());
	size_t alignedN = C.size() - (C.size() % 8);
	for (int i = 0; i < alignedN; i += 8) {
		__m256 avec = _mm256_loadu_ps(&A[i]);
		__m256 bvec = _mm256_loadu_ps(&B[i]);

		__m256 cvec = _mm256_add_ps(avec, bvec);

		_mm256_storeu_ps(&C[i], cvec);
	}
	for (int i = alignedN; i < A.size(); ++i) {
		C[i] == A[i] + B[i];
	}
}

void sum(vector<float>& C, const vector<float>& A, const vector<float>& B) {
	assert(A.size() == B.size() && A.size() == C.size());
	for (int i = 0; i < A.size(); ++i) {
		C[i] = A[i] + B[i];
	}
}

void sum_mat(vector<vector<float> >& C, const vector<vector<float> >& A, const vector<vector<float> >& B) {
	assert(C.size() == A.size() && A.size() == B.size());
	assert(C[0].size() == A[0].size() && A[0].size() == B[0].size());
	size_t h = C.size(), w = C[0].size();
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			C[i][j] = A[i][j] + B[i][j];
		}
	}
}
void _vec_sum_mat(vector<vector<float> >& C, const vector<vector<float> >& A, const vector<vector<float> >& B) {
	assert(C.size() == A.size() && A.size() == B.size());
	assert(C[0].size() == A[0].size() && A[0].size() == B[0].size());
	size_t h = C.size(), w = C[0].size();
	size_t alignedN = (h * w) - (h * w % 8);
	
	for (int k = 0; k < alignedN; k += 8) {
		int i = k / w, j = k % w;
		__m256 avec = _mm256_loadu_ps(&A[i][j]);
		__m256 bvec = _mm256_loadu_ps(&B[i][j]);

		__m256 cvec = _mm256_add_ps(avec, bvec);

		_mm256_storeu_ps(&C[i][j], cvec);
	}
	for (int k = alignedN; k < h * w; ++k) {
		C[k / w][k % w] = A[k / w][k % w] + B[k / w][k % w];
	}

}

int main() {
	//size_t N = 100000000;
	//vector<float> A(N), B(N), C1(N), C2(N);
	cout << "Values from initialised vector" << endl;
	vector<double> inited(10);
	for (auto val : inited) {
		cout << val << endl;
	}
	size_t m = 10000, n = 10000;
	vector<vector<float> > X(m, vector<float>(n, 0)), Y(m, vector<float>(n, 0)), Z1(m, vector<float>(n, 0)), Z2(m, vector<float>(n, 0));
	default_random_engine rg(time(0));
	uniform_real_distribution<float> random(0, 1);
	//put in random floats
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			X[i][j] = random(rg);
			Y[i][j] = random(rg);
		}
	}
	vector<double> normal_times, simded_times;
	for (int round = 0; round < 20; round++) {
		auto normal0 = chrono::high_resolution_clock::now();
		sum_mat(Z1, X, Y);
		auto normal1 = chrono::high_resolution_clock::now();
		double normal_time = chrono::duration_cast<chrono::nanoseconds>(normal1 - normal0).count() * 1e-9;
		auto simded0 = chrono::high_resolution_clock::now();
		_vec_sum_mat(Z2, X, Y);
		auto simded1 = chrono::high_resolution_clock::now();
		double simded_time = chrono::duration_cast<chrono::nanoseconds>(simded1 - simded0).count() * 1e-9;
		if (round > 0) {
			normal_times.push_back(normal_time);
			simded_times.push_back(simded_time);
		}
	}

	cout << "Mean Normal time: " << mean(normal_times) << endl;
	cout << "Mean Simded time: " << mean(simded_times) << endl;

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (Z1[i][j] != Z2[i][j]) {
				cout << "Not same at " << i << ", " << j << endl;
				cout << Z1[i][j] << " " << Z2[i][j] << endl;
				break;
			}
		}
	}
	cout << "All values equal!!" << endl;

	
	return 0;
}
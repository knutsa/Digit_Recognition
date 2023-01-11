#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <random>
#include <immintrin.h>
#include <map>
#include <omp.h>



using namespace std;

void sum_vecs(vector<float>& sum, const vector<float>& A, const vector<float>& B) {
	assert(sum.size() == A.size() && A.size() == B.size());

	for (int i = 0; i < A.size(); ++i) {
		sum[i] = A[i] + B[i];
	}
}
//void _vectorized_sum(vector<float>& sum, const vector<float>& A,const vector<float>& B) {
//	assert(sum.size() == A.size() && A.size() == B.size());
//	size_t alignedN = A.size() - (A.size() % 8);
//	for (int i = 0; i < alignedN; i += 4) {
//		__m256 avec = _mm256_loadu_ps(&A[i]);
//		__m256 bvec = _mm256_loadu_ps(&B[i]);
//
//		__m256 sumvec = _mm256_add_ps(avec, bvec);
//
//		_mm256_storeu_ps(&sum[i], sumvec);
//	}
//	for (int i = alignedN; i < A.size(); ++i) {
//		sum[i] = A[i] + B[i];
//	}
//}

void sum_omp(vector<float>& sum, const vector<float>& A, vector<float>& B) {
#pragma omp parallel for
	for (unsigned int i = 0; i < sum.size(); ++i) {
		sum[i] = A[i] + B[i];
	}
}

double mean(vector<double> vals) {
	double sum = 0.0;
	for (auto val : vals) {
		sum += val;
	}
	return sum / vals.size();
}

int main() {
	
	int N = 100000000;

	//omp_set_num_threads(4);
	cout << "Max threads is: " << omp_get_max_threads() << endl;

	vector<float>  A(N);
	vector<float> B(N);
	vector<float> C1(N);
	vector<float> C2(N);
	for (int i = 0; i < A.size(); i++) {
		A[i] = (double)(rand() % 10);
		B[i] = (double)(rand() % 10);
	}
	assert(A.size() == B.size());

	vector<double> times_normal, times_omp;
	for (int i = 0; i < 10; ++i) {
		auto start = chrono::high_resolution_clock::now();
		sum_vecs(C1, A, B);
		auto stop1 = chrono::high_resolution_clock::now();
		sum_omp(C2, A, B);
		auto stop2 = chrono::high_resolution_clock::now();

		auto time_normal = chrono::duration_cast<chrono::nanoseconds>(stop1 - start).count();
		auto time_omp = chrono::duration_cast<chrono::nanoseconds>(stop2 - stop1).count();
		if (i > 0) {
			times_normal.push_back(time_normal * 1e-9);
			times_omp.push_back(time_omp * 1e-9);
		}
	}


	cout << "Time normal call " <<  mean(times_normal) << " s" << endl;
	cout << "Time with OMP is " << mean(times_omp) << " s" << endl;

	for (int i = 0; i < N; ++i) {
		if (C1[i] - C2[i]) {
			cout << "Not the same answer at " << i << endl;
			break;
		}
	}
	cout << "All answers same!!" << endl;

	return 0;
}